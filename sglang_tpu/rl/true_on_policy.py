# SPDX-License-Identifier: Apache-2.0
"""Batch-invariant kernels for true on-policy RL: a row's output bytes do not depend on the call's row count.

XLA picks its dot tiling and reduction order per shape, so an XLA matmul, log-softmax or RMSNorm returns
different bytes for the same row at different batch sizes. These Pallas kernels fix the tile of every reduction and
walk it in one order, so a decode step, a prefill chunk and the trainer's packed forward give the same
bytes. The trainer calls the same torch ops (registered below), which is what makes its log-probs equal the
engine's.

The contract is the K step of the matmul (``_MM_BK``) and the column tile of the log-prob (``_LP_COLS``): changing
either changes the bytes. Row and column block sizes do not.
"""

# NOTE: do *not* add ``from __future__ import annotations`` here: the torch_tpu
# Pallas bridge reads the wrapped functions' annotations (``jax.Array``).

import contextlib
import dataclasses
import functools

import jax
import jax.numpy as jnp
import torch
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from torch_tpu._internal import pallas as _torch_pallas

# Row and column blocks are sized for speed only (measured on v7x): large ones keep HBM traffic and grid steps down
# for a packed trainer batch, and _row_block shrinks the row block for a small decode batch.
_MM_BM, _MM_BN, _MM_BK = 1024, 2048, 512
# Scoped VMEM the 1024 x 2048 block was measured with (fp32 accumulator plus tiles, about 30 MiB).
_MM_VMEM_BYTES = 48 << 20
_LP_ROWS, _LP_COLS = 1024, 2048
# Elements per norm block (rows x row width); measured up to a 5120-wide row.
_NORM_BLOCK_ELEMS = 1 << 20


def _row_block(m, cap, floor):
    """Rows per block: the power of two covering m rows, within [floor, cap]."""
    return min(cap, max(floor, 1 << (m - 1).bit_length()))


# The last grid step is bound in Python: pl.num_programs inside a kernel function reused for another grid with
# the same block shapes returned the first grid's extent (v7x, jax 0.11), ending the K loop early.
def _mm_kernel(x_ref, w_ref, o_ref, acc_ref, *, nk):
    k = pl.program_id(2)

    @pl.when(k == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    acc_ref[...] += jax.lax.dot_general(
        x_ref[...],
        w_ref[...],
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )

    @pl.when(k == nk - 1)
    def _():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def _jax_matmul(x: jax.Array, w: jax.Array) -> jax.Array:
    """x [M, K] @ w [N, K]^T -> [M, N] in x's dtype; fp32 accumulation over K in ``_MM_BK`` steps, in order."""
    m, k = x.shape
    n = w.shape[0]
    if k % _MM_BK:
        raise ValueError(f"True on-policy matmul needs K % {_MM_BK} == 0, got K={k}")
    bm = _row_block(m, _MM_BM, 128)
    return pl.pallas_call(
        functools.partial(_mm_kernel, nk=k // _MM_BK),
        grid=(pl.cdiv(m, bm), pl.cdiv(n, _MM_BN), k // _MM_BK),
        in_specs=[
            pl.BlockSpec((bm, _MM_BK), lambda i, j, kk: (i, kk)),
            pl.BlockSpec((_MM_BN, _MM_BK), lambda i, j, kk: (j, kk)),
        ],
        out_specs=pl.BlockSpec((bm, _MM_BN), lambda i, j, kk: (i, j)),
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        scratch_shapes=[pltpu.VMEM((bm, _MM_BN), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            vmem_limit_bytes=_MM_VMEM_BYTES,
        ),
    )(x, w)


def _lp_kernel(
    logits_ref, temp_ref, target_ref, o_ref, m_ref, s_ref, pick_ref, *, vocab, nc
):
    j = pl.program_id(1)

    @pl.when(j == 0)
    def _():
        m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
        s_ref[...] = jnp.zeros_like(s_ref)
        pick_ref[...] = jnp.zeros_like(pick_ref)

    col = j * _LP_COLS + jax.lax.broadcasted_iota(jnp.int32, logits_ref.shape, 1)
    x = logits_ref[...].astype(jnp.float32) / temp_ref[...]
    x = jnp.where(col < vocab, x, -jnp.inf)
    m_new = jnp.maximum(m_ref[...], jnp.max(x, axis=1, keepdims=True))
    # While every logit so far is -inf the max is -inf: rescale against 0 there, not -inf - -inf = NaN.
    m_safe = jnp.where(m_new == -jnp.inf, 0.0, m_new)
    s_ref[...] = s_ref[...] * jnp.exp(m_ref[...] - m_safe) + jnp.sum(
        jnp.exp(x - m_safe), axis=1, keepdims=True
    )
    m_ref[...] = m_new
    pick_ref[...] += jnp.sum(
        jnp.where(col == target_ref[...], x, 0.0), axis=1, keepdims=True
    )

    @pl.when(j == nc - 1)
    def _():
        o_ref[...] = pick_ref[...] - (m_ref[...] + jnp.log(s_ref[...]))


def _jax_logprob(
    logits: jax.Array, temperature: jax.Array, target: jax.Array
) -> jax.Array:
    """log_softmax(logits.float() / temperature)[target] per row: logits [M, V], temperature [M] f32, target [M]."""
    m, v = logits.shape
    nc = pl.cdiv(v, _LP_COLS)
    rows = _row_block(m, _LP_ROWS, 16)
    row = lambda i, j: (i, 0)  # noqa: E731
    out = pl.pallas_call(
        functools.partial(_lp_kernel, vocab=v, nc=nc),
        grid=(pl.cdiv(m, rows), nc),
        in_specs=[
            pl.BlockSpec((rows, _LP_COLS), lambda i, j: (i, j)),
            pl.BlockSpec((rows, 1), row),
            pl.BlockSpec((rows, 1), row),
        ],
        out_specs=pl.BlockSpec((rows, 1), row),
        out_shape=jax.ShapeDtypeStruct((m, 1), jnp.float32),
        scratch_shapes=[pltpu.VMEM((rows, 1), jnp.float32)] * 3,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary")
        ),
    )(
        logits,
        temperature.reshape(m, 1).astype(jnp.float32),
        target.reshape(m, 1).astype(jnp.int32),
    )
    return out[:, 0]


def _norm_kernel(*refs, eps, has_residual):
    if has_residual:
        x_ref, r_ref, w_ref, o_ref, ro_ref = refs
        s = x_ref[...].astype(jnp.float32) + r_ref[...].astype(jnp.float32)
        ro_ref[...] = s.astype(ro_ref.dtype)
    else:
        x_ref, w_ref, o_ref = refs
        s = x_ref[...].astype(jnp.float32)
    var = jnp.mean(s * s, axis=1, keepdims=True)
    o_ref[...] = (s * jax.lax.rsqrt(var + eps) * w_ref[...].astype(jnp.float32)).astype(
        o_ref.dtype
    )


def _rms_norm_call(args, *, eps, has_residual):
    """RMSNorm as RMSNorm.forward_native computes it: normalize the fp32 sum, round once to x's dtype."""
    x = args[0]
    m, h = x.shape
    # The largest power-of-two row count within the block budget.
    cap = 1 << ((_NORM_BLOCK_ELEMS // h).bit_length() - 1)
    rows = _row_block(m, cap, 16)
    blk = pl.BlockSpec((rows, h), lambda i: (i, 0))
    n_in = 2 if has_residual else 1
    out_shape = jax.ShapeDtypeStruct((m, h), x.dtype)
    return pl.pallas_call(
        functools.partial(_norm_kernel, eps=eps, has_residual=has_residual),
        grid=(pl.cdiv(m, rows),),
        in_specs=[blk] * n_in + [pl.BlockSpec((1, h), lambda i: (0, 0))],
        out_specs=[blk, blk] if has_residual else blk,
        out_shape=[out_shape, out_shape] if has_residual else out_shape,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(*args[:n_in], args[n_in].reshape(1, h))


def _register(name, fn):
    # jax_op derives the schema and the fake from the function's export; registered at import, outside any trace
    return _torch_pallas.jax_op(f"sglang_tpu_true_on_policy::{name}", fn)


matmul = _register("matmul", _jax_matmul)
logprob = _register("logprob", _jax_logprob)


# eps is a static (non-tensor) op argument: one program per eps value.
def _jax_rms_norm(x: jax.Array, w: jax.Array, eps: float) -> jax.Array:
    return _rms_norm_call((x, w), eps=eps, has_residual=False)


def _jax_rms_norm_residual(
    x: jax.Array, r: jax.Array, w: jax.Array, eps: float
) -> tuple[jax.Array, jax.Array]:
    return tuple(_rms_norm_call((x, r, w), eps=eps, has_residual=True))


_rms_norm = _register("rms_norm", _jax_rms_norm)
_rms_norm_residual = _register("rms_norm_residual", _jax_rms_norm_residual)


def rms_norm(x: torch.Tensor, residual, weight: torch.Tensor, eps: float):
    """[M, H] x (and residual) -> out, or (out, new residual) when a residual is given."""
    if residual is None:
        return _rms_norm(x, weight, eps)
    return _rms_norm_residual(x, residual, weight, eps)


# ===== Backward =====
# The trainer differentiates these ops. The forward is the batch-invariant program; the backward only has to be
# the gradient of the same math, so it is plain torch: fp32 for the norm and the log-prob, the gradient's own
# dtype (bf16, as for any bf16 linear) for the matmul.


def _save_inputs(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _matmul_grad(ctx, grad):
    x, w = ctx.saved_tensors
    return grad.mm(w.to(grad.dtype)).to(x.dtype), grad.t().mm(x.to(grad.dtype)).to(
        w.dtype
    )


matmul.register_autograd(_matmul_grad, setup_context=_save_inputs)


def _norm_input_grad(s, w, grad_out, eps):
    """d/ds of s * rsqrt(mean(s^2) + eps) * w, all fp32; returns (grad_s, grad_w)."""
    g = grad_out.float()
    rstd = torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps)
    s_hat = s * rstd
    gw = g * w.float()
    grad_s = rstd * (gw - s_hat * (gw * s_hat).mean(-1, keepdim=True))
    return grad_s, (g * s_hat).sum(0)


def _norm_setup(ctx, inputs, output):
    # eps, the last input, is a float: kept on ctx, and its gradient slot below is None.
    ctx.save_for_backward(*inputs[:-1])
    ctx.eps = inputs[-1]


def _norm_grad(ctx, grad):
    x, w = ctx.saved_tensors
    grad_s, grad_w = _norm_input_grad(x.float(), w, grad, ctx.eps)
    return grad_s.to(x.dtype), grad_w.to(w.dtype), None


def _norm_residual_grad(ctx, grad_out, grad_residual):
    x, r, w = ctx.saved_tensors
    grad_s, grad_w = _norm_input_grad(x.float() + r.float(), w, grad_out, ctx.eps)
    # The new residual is bf16(s); its gradient passes straight through the rounding.
    grad_s = grad_s + grad_residual.float()
    return grad_s.to(x.dtype), grad_s.to(r.dtype), grad_w.to(w.dtype), None


_rms_norm.register_autograd(_norm_grad, setup_context=_norm_setup)
_rms_norm_residual.register_autograd(_norm_residual_grad, setup_context=_norm_setup)


def _logprob_grad(ctx, grad):
    logits, temperature, target = ctx.saved_tensors
    t = temperature.float()[:, None]
    # (onehot(target) - softmax) * grad / T, built in the softmax's own [M, V] fp32 buffer.
    g = torch.softmax(logits.float() / t, dim=-1).neg_()
    g.scatter_add_(1, target.long()[:, None], torch.ones_like(t))
    return g.mul_(grad.float()[:, None] / t).to(logits.dtype), None, None


logprob.register_autograd(_logprob_grad, setup_context=_save_inputs)


# ===== Attention =====
# The engine's batched RPA kernel, with its KV block pinned to the page so decode, prefill, chunked prefill and a
# packed training batch walk a row's keys in the same blocks. The SEQ_ALONG_LANE layout and the page size are part
# of the contract: the two layouts round differently.
PAGE_SIZE = 256


def pin_kv_block(decode, prefill, page_size):
    """One KV block, the page, for the decode and the prefill launches."""
    if page_size != PAGE_SIZE:
        raise ValueError(
            f"True on-policy attention runs {PAGE_SIZE}-token pages, got {page_size}"
        )
    return dataclasses.replace(decode, bkv_sz=page_size), dataclasses.replace(
        prefill, bkv_sz=page_size
    )


def refuse_rpa_rounding_flags():
    """The RPA tile flags that change rounding must be unset in every process that traces this attention."""
    from sglang_tpu.envs import envs

    flags = (
        "SGLANG_TPU_RPA_EXP2_SOFTMAX",
        "SGLANG_TPU_RPA_ONES_V",
        "SGLANG_TPU_RPA_TRANSPOSED",
    )
    if on := [name for name in flags if getattr(envs, name).get()]:
        raise ValueError(
            f"True on-policy attention needs {', '.join(on)} unset: they change the rounding"
        )


@contextlib.contextmanager
def _pinned_kv_blocks():
    """Pin the KV block while the trainer's attention is traced (the engine pins it with a hook)."""
    from sglang_tpu.attention.pallas.batched_rpa import wrapper

    original = wrapper.calculate_block_sizes

    def pinned(model_cfgs, serve_cfgs, vmem_limit_bytes):
        return pin_kv_block(
            *original(model_cfgs, serve_cfgs, vmem_limit_bytes), serve_cfgs.page_size
        )

    wrapper.calculate_block_sizes = pinned
    try:
        yield
    finally:
        wrapper.calculate_block_sizes = original


def _jax_packed_causal_attention(
    q: jax.Array, k: jax.Array, v: jax.Array, cu_seqlens: jax.Array, sm_scale: float
) -> jax.Array:
    """Causal attention within each packed sequence, as the engine computes it.

    q [T, Hq, D], k and v [T, Hkv, D] bf16; cu_seqlens [n + 1] int32 sequence starts (rows past cu_seqlens[-1] are
    padding); sm_scale is the layer's softmax scale (a static argument). Each sequence gets its own pages; the
    kernel writes its K/V there and attends, as an engine extend with no cached prefix does. Plain causal attention
    over a bf16 cache only: no sliding window, logit cap or attention sink.
    """
    from sglang_tpu.attention.pallas.batched_rpa import configs, wrapper

    refuse_rpa_rounding_flags()
    tokens, _, head_dim = q.shape
    n = cu_seqlens.shape[0] - 1
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    pages_per_seq = pl.cdiv(tokens, PAGE_SIZE)
    num_pages = pages_per_seq + n
    seq_pages = (lens + PAGE_SIZE - 1) // PAGE_SIZE
    first_page = jnp.cumsum(seq_pages) - seq_pages
    page_indices = jnp.minimum(
        first_page[:, None] + jnp.arange(pages_per_seq)[None, :], num_pages - 1
    )
    # One idle sequence slot past the batch, as the serving backend pads it.
    page_indices = jnp.concatenate(
        [page_indices, jnp.zeros((1, pages_per_seq), page_indices.dtype)]
    ).reshape(-1)
    kv_lens = jnp.concatenate([lens, jnp.zeros((1,), lens.dtype)]).astype(jnp.int32)
    cu_q_lens = jnp.concatenate([cu_seqlens, cu_seqlens[-1:]]).astype(jnp.int32)
    layout = configs.KVLayout.SEQ_ALONG_LANE
    kv_cache = jnp.zeros(
        wrapper.get_kv_cache_shape(
            num_pages, PAGE_SIZE, k.shape[1], head_dim, k.dtype, kv_layout=layout
        ),
        k.dtype,
    )
    distribution = jnp.full((3,), n, jnp.int32)  # every sequence is an extend
    with _pinned_kv_blocks():
        out, _ = wrapper.ragged_paged_attention(
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices.astype(jnp.int32),
            cu_q_lens,
            distribution,
            sm_scale=sm_scale,
            kv_layout=layout,
        )
    return out


packed_causal_attention = _register(
    "packed_causal_attention", _jax_packed_causal_attention
)
