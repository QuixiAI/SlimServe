# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native step tape for DeepSeek-V4 decode on Apple Metal.

One C++ call per decoder layer (qc_tape_layer_forward) replaces the
Python/torch encode of the uniform layer body. Stage 1 is bit-exact by
construction: the C++ body chains the same quixicore host ops and the same
aten ops, in the same order, as the Python path (site references live in
qc_metal_serving.mm). Coverage is the non-indexer layers only:

  kind 0: compressor-only layers (compress_ratio 128). The c128 compress
          tail only runs on boundary steps; those steps fall back to the
          Python body for these layers.
  kind 1: SWA-only layers (compress_ratio 1).

Indexer layers (compress_ratio 4) always run the Python body.

Modes (VLLM_QC_STEP_TAPE): 0 = off (default), 1 = on, 2 = verify (run the
Python body, then the tape body, compare outputs bitwise, return the Python
result; cache writes are value-idempotent when the tape is correct, and the
first mismatching layer is what verification is for).

Fallback per call: no attention metadata dict, more than 8 tokens (the GGUF
GEMV batch ceiling for the 32k-row wq_b), or a c128 boundary step (kind 0).

Known open issue: the tape last verified bitwise against the Python routes
of 2026-08-11. The serving routes have since evolved (q2_K SoA repack,
sum-folded down projection, marshalling memos), and mode-2 verify currently
reports per-layer bitwise mismatches on the production layout. Serving is
unaffected (mode 2 returns the Python result; mode 0 is the default), but
do not trust mode 1 until the C++ body is re-validated route by route.
"""

import os

import torch

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.deepseek_v4.metal import (
    build_comp_none,
    build_comp_tables,
    build_swa_tables,
)
from vllm.quixicore.ops import quixicore_ops

logger = init_logger(__name__)


def _tape_mode() -> int:
    # metal_worker.py enables the tape for any value other than "0", then
    # imports this module; a non-numeric value like "on" must degrade to
    # off instead of raising during load_model.
    raw = os.environ.get("VLLM_QC_STEP_TAPE", "0") or "0"
    try:
        return int(raw)
    except ValueError:
        logger.warning("qc-tape: invalid VLLM_QC_STEP_TAPE=%r; tape off", raw)
        return 0


_MODE = _tape_mode()
_MAX_TAPE_TOKENS = 8  # gguf/linear.py mmvq ceiling for rows > 5120
_GGUF_Q8_0 = 8
_GGUF_Q2_K = 10
_GGUF_IQ2_XXS = 16


def _qt(module) -> int:
    return int(module.qweight_type.weight_type)


def _register_layer(layer, idx: int) -> bool:
    """Collect and register one layer's persistent tensors. Returns False
    (and disables the tape for this layer) if any structural assumption of
    the C++ body does not hold."""
    attn = layer.attn
    ffn = layer.ffn
    runner = ffn.experts
    gate = runner.gate
    router = runner.router
    rexp = runner.routed_experts
    shared = ffn.shared_experts
    kind = 0 if attn.compressor is not None else 1
    try:
        checks = [
            not layer.use_fused_mhc,
            attn.indexer is None,
            attn.padded_heads == attn.n_local_heads,
            attn.n_local_heads % attn.n_local_groups == 0,
            not hasattr(attn.wo_a, "qweight"),
            _qt(attn.fused_wqa_wkv) == _GGUF_Q8_0,
            _qt(attn.wq_b) == _GGUF_Q8_0,
            _qt(attn.wo_b) == _GGUF_Q8_0,
            _qt(shared.gate_up_proj) == _GGUF_Q8_0,
            _qt(shared.down_proj) == _GGUF_Q8_0,
            _qt_pair_ok(rexp),
            runner.is_internal_router,
            getattr(gate, "out_dtype", None) == torch.float32,
            type(router).__name__ == "FusedTopKBiasRouter",
            router.scoring_func == "sqrtsoftplus",
            router.num_fused_shared_experts == 0,
            rexp.global_to_local_expert_map is None,
            not rexp.apply_router_weight_on_input,
            not getattr(rexp, "_dsv4_defer_down", False),
            rexp.activation.value == "silu",
            float(shared.act_fn.alpha) == 1.0,
            float(shared.act_fn.beta) == 0.0,
            quixicore_ops.has("rms_norm"),
            quixicore_ops.has("dsv4_o_inv_rope"),
            quixicore_ops.has("qc_swiglu"),
            quixicore_ops.has("moe_weighted_sum"),
            quixicore_ops.has("dsv4_router_topk"),
        ]
    except AttributeError as exc:
        logger.warning("qc-tape: layer %d structure probe failed: %s", idx, exc)
        return False
    if not all(checks):
        logger.warning(
            "qc-tape: layer %d fails structural check #%d; running Python",
            idx,
            checks.index(False),
        )
        return False

    ctx = get_forward_context()
    md = ctx.attn_metadata
    assert isinstance(md, dict)

    tensors = {
        "hc_attn_fn": layer.hc_attn_fn.data,
        "hc_attn_scale": layer.hc_attn_scale.data,
        "hc_attn_base": layer.hc_attn_base.data,
        "hc_ffn_fn": layer.hc_ffn_fn.data,
        "hc_ffn_scale": layer.hc_ffn_scale.data,
        "hc_ffn_base": layer.hc_ffn_base.data,
        "attn_norm_w": layer.attn_norm.weight.data,
        "ffn_norm_w": layer.ffn_norm.weight.data,
        # The tape wants the fp32 copies fused_qk_rmsnorm memoizes; cache
        # them once here too.
        "q_norm_w": attn.q_norm.weight.data.float(),
        "kv_norm_w": attn.kv_norm.weight.data.float(),
        "wqa_wkv_qw": attn.fused_wqa_wkv.qweight,
        "wq_b_qw": attn.wq_b.qweight,
        "wo_b_qw": attn.wo_b.qweight,
        "swa_kv_cache": attn.swa_cache_layer.kv_cache,
        "attn_sink": attn.attn_sink,
        "cos_sin_cache": attn.rotary_emb.cos_sin_cache,
        # Same storage as metal.py's per-call reshape; cached view.
        "wo_a_w": attn.wo_a.weight.data.reshape(
            attn.n_local_groups,
            attn.o_lora_rank,
            (attn.n_local_heads // attn.n_local_groups) * attn.head_dim,
        ),
        "gate_w": gate.weight.data,
        "w13_qw": rexp.w13_qweight,
        "w2_qw": rexp.w2_qweight,
        "sh_gateup_qw": shared.gate_up_proj.qweight,
        "sh_down_qw": shared.down_proj.qweight,
    }
    if router.e_score_correction_bias is not None:
        tensors["router_bias"] = router.e_score_correction_bias.data
    if router._hash_indices_table is not None:
        tensors["hash_table"] = router._hash_indices_table

    state_block_size = 0
    state_width = 0
    compress_ratio = 0
    if kind == 0:
        compressor = attn.compressor
        state_md = md[compressor.state_cache.prefix]
        state_cache = compressor.state_cache.kv_cache
        tensors["comp_w"] = compressor.fused_wkv_wgate.weight.data
        # bf16 APE copy, cached once (mirrors the compressor's _ape_bf16
        # memo).
        tensors["ape_bf16"] = compressor.ape.to(torch.bfloat16).contiguous()
        tensors["state_cache"] = state_cache
        tensors["comp_kv_cache"] = attn.kv_cache
        state_block_size = int(state_md.block_size)  # type: ignore[attr-defined]
        state_width = state_cache.shape[-1] // 2
        compress_ratio = int(compressor.compress_ratio)

    scalars = {
        "kind": kind,
        "wqa_wkv_qt": _qt(attn.fused_wqa_wkv),
        "wq_b_qt": _qt(attn.wq_b),
        "wo_b_qt": _qt(attn.wo_b),
        "w13_qt": int(rexp.w13_qweight_type.weight_type),
        "w2_qt": int(rexp.w2_qweight_type.weight_type),
        "sh_gateup_qt": _qt(shared.gate_up_proj),
        "sh_down_qt": _qt(shared.down_proj),
        "w13_row": int(rexp.w13_qweight.shape[1]),
        "w2_row": int(rexp.w2_qweight.shape[1]),
        # The loader SoA-repacks q2_K w2 at load time; the tape body selects
        # the matching plane-layout kernels from these flags.
        "w13_soa": bool(getattr(rexp, "_dsv4_w1_repacked", False)),
        "w2_soa": bool(getattr(rexp, "_dsv4_w2_repacked", False)),
        "q_lora": int(attn.q_lora_rank),
        "kv_dim": int(attn.head_dim),
        "n_heads": int(attn.n_local_heads),
        "head_dim": int(attn.head_dim),
        "o_groups": int(attn.n_local_groups),
        "sinkhorn_iters": int(layer.hc_sinkhorn_iters),
        "top_k": int(router.top_k),
        "state_block_size": state_block_size,
        "state_width": state_width,
        "compress_ratio": compress_ratio,
        "rms_eps": float(layer.rms_norm_eps),
        "hc_eps": float(layer.hc_eps),
        "hc_post_alpha": float(layer.hc_post_alpha),
        "qk_eps": float(attn.eps),
        "sm_scale": float(attn.scale),
        "swiglu_limit": float(rexp.swiglu_limit),
        "routed_scaling": float(router.routed_scaling_factor),
        "renormalize": bool(router.renormalize),
    }
    if float(shared.act_fn.swiglu_limit) != float(rexp.swiglu_limit):
        logger.warning("qc-tape: layer %d swiglu limit mismatch", idx)
        return False
    quixicore_ops.qc_tape_register_layer(idx, tensors, scalars)
    return True


def _qt_pair_ok(rexp) -> bool:
    # The C++ body hard-codes the Metal decode vec route (fused SwiGLU vec
    # gate/up, sum-folded down), which gguf/fused_moe.py only takes for the
    # IQ2_XXS gate/up + q2_K down pair.
    return (
        int(rexp.w13_qweight_type.weight_type) == _GGUF_IQ2_XXS
        and int(rexp.w2_qweight_type.weight_type) == _GGUF_Q2_K
    )


def _layer_step_tensors(layer, ctx, positions, num_tokens):
    """Assemble THIS layer's per-step tensors from its own metadata objects,
    memoized in forward_mqa's pass cache with forward_mqa's exact keys so the
    Python path reuses the same tensors. Per-layer (not per-step) because
    vLLM's KV group unification can split identical cache specs into multiple
    groups with distinct slot mappings and block tables.

    Returns (step_dict, insert_block_size, boundary) or None if this layer's
    metadata is missing this step."""
    md = ctx.attn_metadata
    attn = layer.attn
    pass_cache = getattr(ctx, "_metal_mqa_cache", None)
    if pass_cache is None:
        pass_cache = {}
        ctx._metal_mqa_cache = pass_cache

    swa_md = md.get(attn.swa_cache_layer.prefix)
    if swa_md is None:
        return None
    device = attn.attn_sink.device
    swa_key = ("swa", id(swa_md), attn.window_size, num_tokens)
    swa_entry = pass_cache.get(swa_key)
    if swa_entry is None:
        swa_entry = build_swa_tables(
            swa_md, positions, num_tokens, attn.window_size, device
        )
        pass_cache[swa_key] = swa_entry
    req_ids, valid_tokens, swa_slots, swa_lens = swa_entry

    step = {
        "swa_slot_mapping": swa_md.slot_mapping,
        "swa_slots": swa_slots,
        "swa_lens": swa_lens,
    }

    boundary = False
    if attn.compressor is not None:
        layer_md = md.get(attn.prefix)
        if layer_md is None or attn.topk_indices_buffer is None:
            return None
        width = min(
            attn.topk_indices_buffer.shape[1],
            (attn.max_model_len + attn.compress_ratio - 1) // attn.compress_ratio,
        )
        comp_key = (
            "comp",
            id(layer_md),
            attn.compress_ratio,
            width,
            num_tokens,
        )
        comp_entry = pass_cache.get(comp_key)
        if comp_entry is None:
            comp_entry = build_comp_tables(
                layer_md,
                positions,
                req_ids,
                valid_tokens,
                attn.compress_ratio,
                width,
                num_tokens,
                device,
            )
            pass_cache[comp_key] = comp_entry
        step["comp_slots"], step["comp_lens"] = comp_entry
        state_md = md.get(attn.compressor.state_cache.prefix)
        if state_md is None:
            return None
        step["comp_state_slot_mapping"] = state_md.slot_mapping
        boundary = bool(state_md.c128_boundary)
    else:
        none_key = ("comp_none", num_tokens)
        none_entry = pass_cache.get(none_key)
        if none_entry is None:
            none_entry = build_comp_none(num_tokens, device)
            pass_cache[none_key] = none_entry
        step["comp_slots"], step["comp_lens"] = none_entry

    return step, int(swa_md.block_size), boundary


def _wrap_layer(layer, idx: int):
    orig_forward = layer.forward
    layer._qc_tape_registered = False
    layer._qc_tape_disabled = False
    kind = 0 if layer.attn.compressor is not None else 1

    def tape_forward(
        x, positions, input_ids, post_mix=None, res_mix=None, residual=None
    ):
        if layer._qc_tape_disabled:
            return orig_forward(x, positions, input_ids, post_mix, res_mix, residual)
        ctx = get_forward_context()
        md = ctx.attn_metadata
        if not isinstance(md, dict) or x.shape[0] > _MAX_TAPE_TOKENS or x.dim() != 3:
            return orig_forward(x, positions, input_ids, post_mix, res_mix, residual)
        if not layer._qc_tape_registered:
            if not _register_layer(layer, idx):
                layer._qc_tape_disabled = True
                return orig_forward(
                    x, positions, input_ids, post_mix, res_mix, residual
                )
            layer._qc_tape_registered = True
        step_res = _layer_step_tensors(layer, ctx, positions, x.shape[0])
        if step_res is None:
            return orig_forward(x, positions, input_ids, post_mix, res_mix, residual)
        step, insert_block, boundary = step_res
        if kind == 0 and boundary:
            # c128 compress tail runs this step; Python owns it.
            return orig_forward(x, positions, input_ids, post_mix, res_mix, residual)
        if _MODE == 2:
            out = orig_forward(x, positions, input_ids, post_mix, res_mix, residual)
            tape_x = quixicore_ops.qc_tape_layer_forward(
                idx, x, positions, input_ids, step, insert_block
            )
            ref = out[0]
            # int16 views: MPS has no eq kernel for uint16; the bit pattern
            # comparison is identical either way.
            same = torch.equal(ref.view(torch.int16), tape_x.view(torch.int16))
            if not same:
                diff = (ref.float() - tape_x.float()).abs().max().item()
                nz = (ref.view(torch.int16) != tape_x.view(torch.int16)).sum().item()
                logger.warning(
                    "qc-tape verify MISMATCH layer=%d kind=%d nz=%d/%d "
                    "max_abs_diff=%.3e",
                    idx,
                    kind,
                    nz,
                    ref.numel(),
                    diff,
                )
            return out
        tape_x = quixicore_ops.qc_tape_layer_forward(
            idx, x, positions, input_ids, step, insert_block
        )
        return tape_x, None, None, None

    layer.forward = tape_forward


def maybe_install_tape(model) -> None:
    """Wrap eligible decoder layers with the native tape. Called once after
    weight load; registration itself is lazy (KV caches bind later)."""
    if _MODE == 0:
        return
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        logger.warning("qc-tape: model has no .model.layers; tape disabled")
        return
    wrapped = []
    for idx, layer in enumerate(layers):
        attn = getattr(layer, "attn", None)
        if attn is None or getattr(attn, "indexer", None) is not None:
            continue  # indexer layers run the Python body
        if getattr(layer, "use_fused_mhc", True):
            continue
        _wrap_layer(layer, idx)
        wrapped.append(idx)
    logger.info(
        "qc-tape: mode %d, wrapped %d layers: %s",
        _MODE,
        len(wrapped),
        wrapped,
    )
