# SPDX-License-Identifier: Apache-2.0
"""True on-policy mode: every op whose bytes could depend on the batch runs a batch-invariant program.

Registered only with SGLANG_TPU_TRUE_ON_POLICY=1; unset, no hook exists.

Why: an RL trainer recomputes the rollout's log-probs in one packed forward. They equal the engine's bit for
     bit only if every op gives a row the same bytes whatever else shares its batch: XLA's dot, log-softmax and
     RMSNorm do not (they tile per shape), and the batched RPA kernel picks a different KV block for decode than
     for prefill. See sglang_tpu/rl/true_on_policy.py for the kernels.
How: AROUND UnquantizedLinearMethod.apply and LogitsProcessor._compute_lm_head_rows (fixed-tile matmul),
     AROUND RMSNorm.forward_native (fixed-tile norm), AROUND Sampler.forward and InputLogprobProcessor.forward
     (the returned log-probs come from the fixed-tile log-prob kernel), AFTER batched RPA calculate_block_sizes
     (one KV block, the page size, for decode and prefill), AFTER SamplingParams.verify (a request asking for a
     logit transform the log-prob does not model gets a 400). Anything else these programs do not cover raises.
Upstream PR or issue: not filed (prototype).
Remove when: core exposes batch-invariant op seams on TPU.
"""

from __future__ import annotations

from typing import Any

import torch

from sglang_tpu.envs import envs


def _kernels():
    from sglang_tpu.rl import true_on_policy

    return true_on_policy


def _refuse_tp():
    from sglang.srt.distributed.parallel_state import (
        get_tensor_model_parallel_world_size,
    )

    if get_tensor_model_parallel_world_size() != 1:
        raise ValueError(
            "TPU true on-policy mode serves TP=1 only: a TP split rounds partial sums"
        )


def true_on_policy_linear(original, self, layer, x: torch.Tensor, bias=None):
    """AROUND UnquantizedLinearMethod.apply: the fixed-tile matmul for every bf16 linear."""
    if x.device.type != "tpu":
        return original(self, layer, x, bias)
    _refuse_tp()
    if (
        bias is not None
        or x.dtype != torch.bfloat16
        or layer.weight.dtype != torch.bfloat16
    ):
        raise ValueError(
            f"TPU true on-policy linear needs bf16 x and weight and no bias, got {x.dtype=} "
            f"{layer.weight.dtype=} bias={bias is not None}"
        )
    out = _kernels().matmul(x.reshape(-1, x.shape[-1]), layer.weight)
    return out.reshape(*x.shape[:-1], out.shape[-1])


def true_on_policy_lm_head(
    original, self, hidden_states: torch.Tensor, lm_head, embedding_bias=None
):
    """AROUND LogitsProcessor._compute_lm_head_rows: bf16 logits from the fixed-tile matmul."""
    if hidden_states.device.type != "tpu":
        return original(self, hidden_states, lm_head, embedding_bias)
    if embedding_bias is not None or lm_head.weight.dtype != torch.bfloat16:
        raise ValueError(
            f"TPU true on-policy lm_head needs an unquantized bf16 head without bias, got "
            f"{lm_head.weight.dtype=} bias={embedding_bias is not None}"
        )
    return _kernels().matmul(hidden_states.to(torch.bfloat16), lm_head.weight)


def true_on_policy_rms_norm(
    original,
    self,
    x: torch.Tensor,
    residual=None,
    post_residual_addition=None,
    quant_linear=None,
):
    """AROUND RMSNorm.forward_native: the fixed-tile norm, rounding where forward_native rounds."""
    if x.device.type != "tpu":
        return original(self, x, residual, post_residual_addition, quant_linear)
    if (
        post_residual_addition is not None
        or self.variance_size_override is not None
        or self.cast_x_before_out_mul
        or self.fp32_residual
        or self.override_orig_dtype is not None
        or x.dtype != torch.bfloat16
        or self.weight.dtype != torch.bfloat16
    ):
        raise ValueError(
            "TPU true on-policy RMSNorm covers the plain bf16 norm (with or without residual) only"
        )
    h = x.shape[-1]
    flat = x.reshape(-1, h)
    if residual is None:
        return (
            _kernels()
            .rms_norm(flat, None, self.weight, self.variance_epsilon)
            .reshape(x.shape)
        )
    out, new_residual = _kernels().rms_norm(
        flat, residual.reshape(-1, h), self.weight, self.variance_epsilon
    )
    return out.reshape(x.shape), new_residual.reshape(x.shape)


def true_on_policy_sampler(
    original, self, logits_output, sampling_info, return_logprob, *args, **kwargs
):
    """AROUND Sampler.forward: sample as before; the returned token log-probs come from the log-prob kernel."""
    if not return_logprob:
        return original(
            self, logits_output, sampling_info, return_logprob, *args, **kwargs
        )
    # The sampler scales and softmaxes the logits in place.
    logits = logits_output.next_token_logits.clone()
    token_ids = original(
        self, logits_output, sampling_info, return_logprob, *args, **kwargs
    )
    temperature = sampling_info.temperatures.reshape(-1).to(torch.float32)
    logits_output.next_token_logprobs = _kernels().logprob(
        logits, temperature, token_ids.to(torch.int32)
    )
    return token_ids


def true_on_policy_input_logprobs(
    original,
    self,
    pruned_states,
    sample_indices,
    input_logprob_indices,
    token_to_seq_idx,
    lm_head,
    get_logits_fn,
    logits_metadata,
    *args,
    **kwargs,
):
    """AROUND InputLogprobProcessor.forward: the prompt token log-probs come from the log-prob kernel."""
    result, sampled_logits = original(
        self,
        pruned_states,
        sample_indices,
        input_logprob_indices,
        token_to_seq_idx,
        lm_head,
        get_logits_fn,
        logits_metadata,
        *args,
        **kwargs,
    )
    if (
        logits_metadata.extend_return_top_logprob
        or logits_metadata.extend_token_ids_logprob
    ):
        raise ValueError(
            "TPU true on-policy prompt log-probs cover the input tokens' own log-probs only"
        )
    logits = get_logits_fn(
        pruned_states, lm_head, logits_metadata, use_logits_buffer=False
    )
    rows = logits[input_logprob_indices]
    targets = logits_metadata.extend_input_logprob_token_ids_gpu
    # Prompt log-probs carry no temperature in sglang.
    ones = torch.ones(rows.shape[0], dtype=torch.float32, device=rows.device)
    result.token_logprobs = _kernels().logprob(rows, ones, targets.to(torch.int32))
    return result, sampled_logits


def refuse_transformed_sampling(result, self, vocab_size) -> Any:
    """AFTER SamplingParams.verify: the returned log-prob is log_softmax(logits / temperature); refuse the rest."""
    from sglang.srt.sampling.sampling_params import TOP_K_ALL

    transforms = {
        "top_p": self.top_p < 1.0,
        # greedy is top_k 1 at temperature 1: its log-prob is the plain log_softmax
        "top_k": 1 < self.top_k < TOP_K_ALL,
        "min_p": self.min_p > 0.0,
        "penalties": (
            self.frequency_penalty,
            self.presence_penalty,
            self.repetition_penalty,
        )
        != (0.0, 0.0, 1.0),
        "logit_bias": self.logit_bias is not None,
        "a grammar": any(
            g is not None
            for g in (self.json_schema, self.regex, self.ebnf, self.structural_tag)
        ),
    }
    if refused := [name for name, on in transforms.items() if on]:
        raise ValueError(
            f"SGLANG_TPU_TRUE_ON_POLICY serves plain temperature sampling; refused: {', '.join(refused)}"
        )
    return result


def pin_kv_block(result, model_cfgs, serve_cfgs, vmem_limit_bytes) -> Any:
    """AFTER batched RPA calculate_block_sizes: one KV block (the page) for decode and prefill alike."""
    return _kernels().pin_kv_block(*result, serve_cfgs.page_size)


def hook_specs():
    if not envs.SGLANG_TPU_TRUE_ON_POLICY.get():
        return []
    # The trainer attends in this layout; the two layouts round differently.
    if not envs.SGLANG_TPU_USE_SEQ_ALONG_LANE_KV.get():
        raise ValueError(
            "SGLANG_TPU_TRUE_ON_POLICY=1 needs SGLANG_TPU_USE_SEQ_ALONG_LANE_KV=1"
        )

    from sglang.srt.plugins.hook_registry import HookType

    return [
        (
            "sglang.srt.layers.quantization.unquant.UnquantizedLinearMethod.apply",
            true_on_policy_linear,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.logits_processor.LogitsProcessor._compute_lm_head_rows",
            true_on_policy_lm_head,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.layernorm.RMSNorm.forward_native",
            true_on_policy_rms_norm,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.sampler.Sampler.forward",
            true_on_policy_sampler,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.logprob_processor.InputLogprobProcessor.forward",
            true_on_policy_input_logprobs,
            HookType.AROUND,
        ),
        (
            "sglang_tpu.attention.pallas.batched_rpa.wrapper.calculate_block_sizes",
            pin_kv_block,
            HookType.AFTER,
        ),
        (
            "sglang.srt.sampling.sampling_params.SamplingParams.verify",
            refuse_transformed_sampling,
            HookType.AFTER,
        ),
    ]
