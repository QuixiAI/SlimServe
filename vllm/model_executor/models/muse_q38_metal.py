# SPDX-License-Identifier: Apache-2.0
"""Muse-qwen38: single-command-buffer decode step (N14).

Registration, eligibility, and shadow validation for the ``muse_q38_*``
extension entry points (csrc/quixicore/tm_metal/qc_metal_serving.mm).
Design + re-pin plan: csrc/quixicore/metal/muse_qwen38_design.md.

Env:
  VLLM_QC_MUSE=0|1|shadow  off (default) / serve the muse CB / run BOTH
                           per step, compare, serve the eager result
  VLLM_QC_MUSE_LAYERS=a,b  comma list of layer caps (0 = full stack);
                           shadow mode replays every cap per step against
                           cloned eager boundaries (divergence bisection)
  VLLM_QC_MUSE_DEBUG=L     layer-L stage-dump isolation (shadow + cap
                           sweep only): recomputes each MLP stage from
                           the muse dump with host ops

The muse step is NOT bit-identical to the eager layer loop by
construction (torch-MPS eager elementwise numerics are size-dependent);
shadow mode reports cos/max-abs stats per run, and the serve-mode flip is
gated by trajectory quality (needle, long decodes, acceptance delta) plus
NEW sha pins.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_MODE = os.environ.get("VLLM_QC_MUSE", "0")
if _MODE not in ("0", "1", "shadow"):
    logger.warning("VLLM_QC_MUSE=%r is not one of 0|1|shadow; muse stays OFF", _MODE)
    _MODE = "0"
# Comma list of layer caps for shadow bisection (0 = full stack); one boot
# sweeps every cap per step against cloned eager boundaries. This module is
# imported lazily from the model forward, so malformed values must degrade,
# not raise mid-decode.
try:
    _LAYERS_CAPS = [
        int(v) for v in os.environ.get("VLLM_QC_MUSE_LAYERS", "0").split(",")
    ]
except ValueError:
    logger.warning(
        "VLLM_QC_MUSE_LAYERS=%r is not a comma list of ints; using [0]",
        os.environ.get("VLLM_QC_MUSE_LAYERS"),
    )
    _LAYERS_CAPS = [0]
# Layer stage-dump instrument (shadow + cap sweep only): isolates which MLP
# stage diverges by re-computing each stage from the muse dump with the
# host ops.
_dbg_env = os.environ.get("VLLM_QC_MUSE_DEBUG", "")
try:
    _DEBUG_LAYER = int(_dbg_env) if _dbg_env != "" else None
except ValueError:
    logger.warning("VLLM_QC_MUSE_DEBUG=%r is not an int; ignored", _dbg_env)
    _DEBUG_LAYER = None
_MAX_ROWS = 8


def mode() -> str:
    return _MODE


def enabled() -> bool:
    return _MODE in ("1", "shadow")


def _qc():
    import vllm._quixicore_C as qc

    return qc


def _available() -> bool:
    try:
        return hasattr(_qc(), "muse_q38_run")
    except ImportError:
        return False


def _qproj(linear: torch.nn.Module, what: str) -> tuple[list[torch.Tensor], int]:
    """Extract the registered GEMV operands of a serving linear: the CT
    fp8ch/nvfp4 operands, or the raw bf16 weight for the few projections
    Unsloth Dynamic leaves unquantized (in_proj_ba)."""
    if getattr(linear, "metal_fp8ch", False):
        return [linear.fp8ch_weight, linear.fp8ch_scale], 0
    if getattr(linear, "metal_nvfp4", False):
        return [linear.nvfp4_weight, linear.nvfp4_scale, linear.nvfp4_global], 1
    w = getattr(linear, "weight", None)
    if (
        isinstance(w, torch.Tensor)
        and w.dim() == 2
        and w.dtype is torch.bfloat16
        and w.is_contiguous()
    ):
        return [w.data], 2
    raise RuntimeError(
        f"muse_q38: {what} has no usable operands (no CT GEMV attrs and "
        f"weight is {None if w is None else (w.dtype, tuple(w.shape))})"
    )


@dataclass
class _AttnInfo:
    prefix: str  # forward-context metadata key of the Attention op
    kc: torch.Tensor
    vc: torch.Tensor
    block_mult: int
    block_size: int


@dataclass
class _State:
    registered: bool = False
    failed: str | None = None
    num_layers: int = 0
    hidden: int = 0
    aux_layers: tuple[int, ...] = ()  # drafter tap boundaries (post-layer)
    # Metadata keys per GDN layer (layer order): the state pools are shared
    # with PER-LAYER slot windows, so every layer's own spec cache must
    # feed the muse run.
    gdn_prefixes: list[str] = field(default_factory=list)
    attn: list[_AttnInfo] = field(default_factory=list)
    gdn_states: list[tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=list
    )  # (conv_state, ssm_state) per GDN layer
    x_buf: torch.Tensor | None = None
    res_buf: torch.Tensor | None = None


_S = _State()


def _register(model: torch.nn.Module) -> None:
    """One-time weight/geometry registration from the live modules (called
    on the first eligible step, after load-time repack and cache bind)."""
    qc = _qc()
    cfg = model.config
    layers = list(model.layers)
    num_layers = len(layers)
    hidden = cfg.hidden_size
    eps = cfg.rms_norm_eps

    first_attn = next(m for m in layers if hasattr(m, "self_attn"))
    first_gdn = next(m for m in layers if hasattr(m, "linear_attn"))
    sa = first_attn.self_attn
    la = first_gdn.linear_attn
    if not la._metal_gdn_enabled():
        raise RuntimeError("muse_q38 needs the Metal GDN kernel path")
    st = la._metal_gdn
    if not (st.fused_prep and st.spec and st.fused_norm):
        raise RuntimeError(
            "muse_q38 needs gdn_fused_prepare + gdn_recur_spec + fused norm"
        )
    if la.gqa_interleaved_layout:
        raise RuntimeError("muse_q38 supports the non-interleaved layout only")

    gHk = la.num_k_heads // la.tp_size
    gHv = la.num_v_heads // la.tp_size
    gDk, gDv = la.head_k_dim, la.head_v_dim

    # Attention KV layout (mirrors metal_attn.forward's dense view).
    def kv_views(
        attn_mod: torch.nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
        kv = attn_mod.kv_cache  # bound as-is by AttentionLayerBase
        assert kv.dim() == 5 and kv.shape[0] == 2, kv.shape
        _, num_blocks, block_size, _, _ = kv.shape
        dense = kv.transpose(0, 1)
        if dense.is_contiguous():
            kc = dense.reshape(2 * num_blocks, block_size, sa.num_kv_heads, sa.head_dim)
            return kc, kc[1:], 2, block_size, num_blocks
        return kv[0], kv[1], 1, block_size, num_blocks

    kc0, _, _, block_size, num_blocks = kv_views(sa.attn)
    ref = torch.empty(1, dtype=torch.bfloat16, device=kc0.device)
    cos_sin = sa.rotary_emb._match_cos_sin_cache_dtype(ref)
    aux_layers = tuple(sorted(model.aux_hidden_state_layers))

    qc.muse_q38_init(
        num_layers=num_layers,
        hidden=hidden,
        heads=sa.num_heads,
        kv_heads=sa.num_kv_heads,
        head_dim=sa.head_dim,
        rot_dim=sa.rotary_emb.rotary_dim,
        attn_scale=sa.scaling,
        gdn_k_heads=gHk,
        gdn_v_heads=gHv,
        gdn_k_dim=gDk,
        gdn_v_dim=gDv,
        inter=cfg.intermediate_size,
        eps=eps,
        max_rows=_MAX_ROWS,
        # Upper bound on block-table entries per request: the pool's total
        # block count, NOT the per-step view width — registration runs on
        # the first eligible step (often the tiny ramp primer) and a
        # step-sized bound silently truncates long-context attention (the
        # muse-serve needle FAIL, P2 R12).
        max_blocks=num_blocks,
        block_size=block_size,
        final_norm_w=model.norm.weight,
        aux_layers=list(aux_layers),
        ref=ref,
    )

    attn_infos: list[_AttnInfo] = []
    gdn_states: list[tuple[torch.Tensor, torch.Tensor]] = []
    gdn_prefixes: list[str] = []
    for idx, layer in enumerate(layers):
        mlp = layer.mlp
        gu_t, gu_f = _qproj(mlp.gate_up_proj, f"layers.{idx}.gate_up")
        dn_t, dn_f = _qproj(mlp.down_proj, f"layers.{idx}.down")
        if hasattr(layer, "linear_attn"):
            la = layer.linear_attn
            if not la._metal_gdn_enabled():
                raise RuntimeError(f"layer {idx}: Metal GDN path unavailable")
            st = la._metal_gdn
            qkvz_t, qkvz_f = _qproj(la.in_proj_qkvz, f"layers.{idx}.qkvz")
            ba_t, ba_f = _qproj(la.in_proj_ba, f"layers.{idx}.ba")
            out_t, out_f = _qproj(la.out_proj, f"layers.{idx}.out")
            conv_state, ssm_state = la.kv_cache[0], la.kv_cache[1]
            qc.muse_q38_layer_gdn(
                idx,
                layer.input_layernorm.weight,
                layer.post_attention_layernorm.weight,
                qkvz_t,
                qkvz_f,
                ba_t,
                ba_f,
                st.conv_w,
                st.A_log,
                st.dt_bias,
                la.norm.weight,
                out_t,
                out_f,
                conv_state,
                ssm_state,
                st.eps,
                st.q_scale,
                st.k_scale,
                la.norm.eps,
                gu_t,
                gu_f,
                dn_t,
                dn_f,
            )
            gdn_states.append((conv_state, ssm_state))
            gdn_prefixes.append(la.prefix)
        else:
            sa_l = layer.self_attn
            qkv_t, qkv_f = _qproj(sa_l.qkv_proj, f"layers.{idx}.qkv")
            o_t, o_f = _qproj(sa_l.o_proj, f"layers.{idx}.o")
            kc, vc, mult, bs, _ = kv_views(sa_l.attn)
            qc.muse_q38_layer_attn(
                idx,
                layer.input_layernorm.weight,
                layer.post_attention_layernorm.weight,
                qkv_t,
                qkv_f,
                sa_l.q_norm.weight,
                sa_l.k_norm.weight,
                cos_sin,
                o_t,
                o_f,
                kc,
                vc,
                mult,
                gu_t,
                gu_f,
                dn_t,
                dn_f,
            )
            attn_infos.append(
                _AttnInfo(
                    prefix=sa_l.attn.layer_name,
                    kc=kc,
                    vc=vc,
                    block_mult=mult,
                    block_size=bs,
                )
            )

    dev = ref.device
    _S.num_layers = num_layers
    _S.hidden = hidden
    _S.aux_layers = aux_layers
    _S.gdn_prefixes = gdn_prefixes
    _S.attn = attn_infos
    _S.gdn_states = gdn_states
    _S.x_buf = torch.empty(_MAX_ROWS, hidden, dtype=torch.bfloat16, device=dev)
    _S.res_buf = torch.empty_like(_S.x_buf)
    _S.registered = True
    logger.info(
        "muse_q38 registered: %d layers (%d attn), hidden %d, mode=%s caps=%s",
        num_layers,
        len(attn_infos),
        hidden,
        _MODE,
        _LAYERS_CAPS,
    )


def _spec_cache(md: Any) -> Any:
    """Borrow or build the shared per-step GDN spec cache (mirrors
    _forward_core_metal_spec; pure-spec only).

    spec_indx/non_spec_indx are forced to None here: muse rejects any
    batch with prefills before this runs, and the eager reader only
    dereferences those fields when num_prefills > 0 — keep that eligibility
    check upstream of this call if the shape of either side changes."""
    cache = getattr(md, "_mps_spec_cache", None)
    if cache is None:
        spec_tab = md.spec_state_indices_tensor
        accepted = md.num_accepted_tokens
        if accepted.dtype != torch.int32:
            accepted = accepted.to(torch.int32)
        cache = SimpleNamespace(
            slot_table=spec_tab,
            conv_slots=spec_tab[:, 0].contiguous(),
            num_accepted=accepted.contiguous(),
            spec_cu=md.spec_query_start_loc[: md.num_spec_decodes + 1],
            spec_indx=None,
            non_spec_indx=None,
        )
        md._mps_spec_cache = cache
    return cache


_REJECT_LOGS = 12


def _reject(reason: str) -> None:
    """Bringup diagnostic: surface the first few eligibility rejections so a
    silently-inert muse path is visible in the engine log."""
    global _REJECT_LOGS
    if _REJECT_LOGS > 0:
        _REJECT_LOGS -= 1
        logger.warning("muse_q38 ineligible step: %s", reason)


def try_step(
    model: torch.nn.Module,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]] | None:
    """Run the muse single-CB step when the batch is eligible. Returns the
    final normed hidden states (plus the drafter aux taps when configured,
    matching the eager return convention), or None to fall back."""
    if not enabled() or _S.failed is not None or not _available():
        return None
    from vllm.forward_context import get_forward_context

    ctx = get_forward_context()
    amd = ctx.attn_metadata
    if not isinstance(amd, dict):
        return None

    if hidden_states.dtype is not torch.bfloat16:
        _reject(f"dtype {hidden_states.dtype}")
        return None

    # Eligibility: uniform pure-spec decode across both mixer families.
    # hidden_states may carry PADDED rows beyond num_actual_tokens; muse
    # runs the actual rows and zeroes the tail (padding is never sampled).
    if _S.registered:
        md_gdn = amd.get(_S.gdn_prefixes[0])
        md_attn = amd.get(_S.attn[0].prefix)
    else:
        md_gdn = md_attn = None
        for key, v in amd.items():
            cls = type(v).__name__
            if cls == "GDNAttentionMetadata" and md_gdn is None:
                md_gdn = v
            elif cls == "MetalAttentionMetadata" and md_attn is None:
                md_attn = v
    if md_gdn is None or md_attn is None:
        _reject(
            "metadata classes not found: "
            + ", ".join(sorted({type(v).__name__ for v in amd.values()}))
        )
        return None
    m = md_gdn.num_actual_tokens
    if md_gdn.spec_sequence_masks is None:
        _reject("no spec masks (non-spec batch)")
        return None
    if md_gdn.num_prefills != 0:
        _reject(f"num_prefills {md_gdn.num_prefills}")
        return None
    if m > _MAX_ROWS or m > hidden_states.shape[0]:
        _reject(f"rows {m} vs cap {_MAX_ROWS} / batch {hidden_states.shape[0]}")
        return None
    if md_attn.num_actual_tokens != m:
        _reject(f"attn tokens {md_attn.num_actual_tokens} != gdn {m}")
        return None
    q_len = md_attn.max_query_len
    reqs = md_attn.num_reqs
    if reqs * q_len != m or md_gdn.num_spec_decodes != reqs:
        _reject(
            f"non-uniform: reqs {reqs} q_len {q_len} m {m} "
            f"spec {md_gdn.num_spec_decodes}"
        )
        return None

    if not _S.registered:
        try:
            _register(model)
        except Exception as exc:  # noqa: BLE001 — quarantine, keep serving
            _S.failed = str(exc)
            logger.warning(
                "muse_q38 registration failed (eager keeps serving)",
                exc_info=True,
            )
            return None

    pos = positions[0] if positions.ndim == 2 else positions
    # One spec cache per GDN layer — per-layer slot windows into the
    # shared pools.
    caches = [_spec_cache(amd[pfx]) for pfx in _S.gdn_prefixes]
    # Attention KV GROUPS: the 16 attention layers span several KV groups
    # with separate block pools (group-local block ids). Build the
    # distinct-metadata list + per-attn-layer group index each step.
    attn_mds = [amd[i.prefix] for i in _S.attn]
    groups: list[Any] = []
    group_ids: dict[int, int] = {}
    attn_group: list[int] = []
    for md in attn_mds:
        key = id(md)
        if key not in group_ids:
            group_ids[key] = len(groups)
            groups.append(md)
        attn_group.append(group_ids[key])
    for gmd in groups:
        if gmd.num_actual_tokens != m or gmd.max_query_len != q_len:
            _reject("attn group metadata mismatch")
            return None
    g_bts = [gmd.block_table for gmd in groups]
    g_sls = [gmd.seq_lens_gpu for gmd in groups]
    g_slots = []
    for gmd in groups:
        sl = gmd.slot_mapping[:m]
        g_slots.append(sl if sl.dtype == torch.long else sl.to(torch.long))
    g_maxctx = [gmd.seq_lens_cpu_max for gmd in groups]

    if _MODE == "1":
        # Fresh outputs per step: the CB writes in place and downstream
        # (logits, drafter taps) may hold the tensors across the next
        # step's encode. Padded tail rows stay zero — never sampled.
        out = torch.zeros_like(hidden_states)
        x = out[:m]
        x.copy_(hidden_states[:m])
        res = _S.res_buf[:m]
        aux_out = (
            torch.zeros(
                len(_S.aux_layers),
                hidden_states.shape[0],
                _S.hidden,
                dtype=torch.bfloat16,
                device=hidden_states.device,
            )
            if _S.aux_layers
            else None
        )
        _qc().muse_q38_run(
            x,
            res,
            pos[:m].contiguous(),
            g_bts,
            g_sls,
            g_slots,
            g_maxctx,
            attn_group,
            [c.spec_cu for c in caches],
            [c.conv_slots for c in caches],
            [c.slot_table for c in caches],
            [c.num_accepted for c in caches],
            q_len,
            0,
            aux_out,
        )
        if aux_out is not None:
            return out, list(aux_out.unbind(0))
        return out

    # ---- shadow mode: eager serves, muse replays against the pre-state ----
    return _shadow_step(
        model,
        hidden_states,
        positions,
        caches,
        (g_bts, g_sls, g_slots, g_maxctx, attn_group),
        q_len,
        m,
        _LAYERS_CAPS,
    )


def _snapshot(g_slots: list, attn_group: list, caches: list) -> dict[str, Any]:
    """Capture every state row the step writes: per-GDN-layer conv/ssm slot
    rows (each layer has its OWN slot window into the shared pools),
    per-attn-layer KV rows."""
    snap: dict[str, Any] = {"gdn": [], "kv": [], "rows": []}
    for (conv_state, ssm_state), cache in zip(_S.gdn_states, caches):
        conv_rows = cache.conv_slots.long()
        # -1 sentinel slots alias onto row 0 here; harmless because the
        # paired restore(post) in _shadow_step puts row 0 back afterwards.
        ssm_rows = cache.slot_table.reshape(-1).long().clamp(min=0)
        snap["rows"].append((conv_rows, ssm_rows))
        snap["gdn"].append((conv_state[conv_rows].clone(), ssm_state[ssm_rows].clone()))
    for info, gi in zip(_S.attn, attn_group):
        sl = g_slots[gi]
        bs = info.block_size
        blocks = (sl // bs) * info.block_mult
        off = sl % bs
        if info.block_mult == 2:
            k_rows = info.kc[blocks, off].clone()
            v_rows = info.kc[blocks + 1, off].clone()
        else:
            k_rows = info.kc[blocks, off].clone()
            v_rows = info.vc[blocks, off].clone()
        snap["kv"].append((k_rows, v_rows))
    return snap


def _restore(snap: dict[str, Any], g_slots: list, attn_group: list) -> None:
    for (conv_state, ssm_state), (c_rows, s_rows), (conv_rows, ssm_rows) in zip(
        _S.gdn_states, snap["gdn"], snap["rows"]
    ):
        conv_state[conv_rows] = c_rows
        ssm_state[ssm_rows] = s_rows
    for info, gi, (k_rows, v_rows) in zip(_S.attn, attn_group, snap["kv"]):
        sl = g_slots[gi]
        bs = info.block_size
        blocks = (sl // bs) * info.block_mult
        off = sl % bs
        if info.block_mult == 2:
            info.kc[blocks, off] = k_rows
            info.kc[blocks + 1, off] = v_rows
        else:
            info.kc[blocks, off] = k_rows
            info.vc[blocks, off] = v_rows


def _eager_prefix(
    model: torch.nn.Module,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    caps: set[int],
) -> tuple[
    torch.Tensor,
    list[torch.Tensor],
    dict[int, tuple[torch.Tensor, torch.Tensor]],
]:
    """Run the eager layer loop (all layers) exactly as model.forward does —
    including the drafter aux taps — capturing CLONED (hidden, residual)
    boundaries after each requested cap. Clones are mandatory: the serving
    stack recycles/mutates intermediate buffers downstream, so a bare
    reference compared after the loop holds later-layer data (the P2
    round-6 false-garbage lesson)."""
    h, r = hidden_states, None
    bounds: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    aux = model._maybe_add_hidden_state([], 0, h, r)
    for idx, layer in enumerate(model.layers):
        h, r = layer(positions=positions, hidden_states=h, residual=r)
        model._maybe_add_hidden_state(aux, idx + 1, h, r)
        if idx + 1 in caps:
            bounds[idx + 1] = (h.clone(), r.clone())
    final, _ = model.norm(h, r)
    return final, [a.clone() for a in aux], bounds


def _gemma_ref(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    return (xf * torch.rsqrt(var + eps) * (1.0 + w.float())).to(torch.bfloat16)


def _host_gemv(linear: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The serving host op for this projection (encoded OUTSIDE the muse
    CB) — the reference the in-CB emit must match."""
    from vllm.quixicore import quixicore_ops

    if getattr(linear, "metal_nvfp4", False):
        return quixicore_ops.nvfp4_mul_mat_vec(
            linear.nvfp4_weight, x.contiguous(), linear.nvfp4_scale, linear.nvfp4_global
        )
    if getattr(linear, "metal_fp8ch", False):
        return quixicore_ops.fp8ch_mul_mat_vec(
            linear.fp8ch_weight, x.contiguous(), linear.fp8ch_scale
        )
    return torch.nn.functional.linear(x, linear.weight)


def _debug_report(
    model: torch.nn.Module,
    dbg: torch.Tensor,
    x_in: torch.Tensor,
    bounds: dict,
    cache: Any,
    restore_pre,
    m: int,
    layer_idx: int,
    x_muse: torch.Tensor,
    full_pre: tuple | None = None,
    full_mid: tuple | None = None,
) -> None:
    """Per-stage isolation for layer `layer_idx`: each muse dump is compared
    against a host-op recomputation from the PREVIOUS muse dump, so exactly
    one stage is under test at a time. GDN layers replay the three-kernel
    core host-side against the rewound state (restore_pre)."""
    from vllm.quixicore.ops import quixicore_ops as qc_ops

    layer = model.layers[layer_idx]
    inter = model.config.intermediate_size
    hidden = _S.hidden
    eps = layer.input_layernorm.variance_epsilon

    def flat(i: int, width: int) -> torch.Tensor:
        return dbg[i].reshape(-1)[: m * width].view(m, width)

    if layer_idx == 0:
        prev_h, prev_r = x_in, torch.zeros_like(x_in)
    else:
        bb = bounds.get(layer_idx)
        if bb is None:
            logger.warning(
                "muse_q38 DEBUG: no eager boundary for layer %d (add cap %d)",
                layer_idx,
                layer_idx,
            )
            return
        prev_h, prev_r = bb[0][:m], bb[1][:m]

    h0 = flat(0, hidden)
    t4 = flat(4, hidden)
    res5 = flat(5, hidden)
    gu = flat(6, 2 * inter)
    mid = flat(7, inter)
    entry_sum = (prev_h.float() + prev_r.float()).to(torch.bfloat16)
    parts = [_stats("h0", _gemma_ref(entry_sum, layer.input_layernorm.weight, eps), h0)]

    if hasattr(layer, "linear_attn"):
        la = layer.linear_attn
        st = la._metal_gdn
        gHk = la.num_k_heads // la.tp_size
        gHv = la.num_v_heads // la.tp_size
        gDk, gDv = la.head_k_dim, la.head_v_dim
        qkv_sz = 2 * gHk * gDk + gHv * gDv
        gdn_w = qkv_sz + gHv * gDv
        qkvz = flat(1, gdn_w)
        ba = flat(2, 2 * gHv)
        gn = flat(3, gHv * gDv)
        parts.append(_stats("qkvz", _host_gemv(la.in_proj_qkvz, h0), qkvz))
        parts.append(_stats("ba", _host_gemv(la.in_proj_ba, h0), ba))
        # Host replay of the GDN core against the rewound state — run
        # TWICE (restore before each) to separate kernel nondeterminism
        # from state-pollution (writes outside the snapshot row set).
        conv_state, ssm_state = la.kv_cache[0], la.kv_cache[1]
        mixed = qkvz[:, :qkv_sz]
        z = qkvz[:, qkv_sz:].reshape(m, gHv, gDv).contiguous()

        def _replay() -> torch.Tensor:
            restore_pre()
            q, k, v, decay, beta = qc_ops.gdn_fused_prepare(
                mixed,
                ba,
                st.conv_w,
                conv_state,
                cache.spec_cu,
                cache.conv_slots,
                st.A_log,
                st.dt_bias,
                gHk,
                gHv,
                gDk,
                gDv,
                st.eps,
                st.q_scale,
                st.k_scale,
                True,
                cache.num_accepted,
            )
            y = qc_ops.gdn_recur_spec(
                q,
                k,
                v,
                decay,
                beta,
                ssm_state,
                cache.spec_cu,
                cache.slot_table,
                cache.num_accepted,
                gHk,
                gHv,
                gDk,
                gDv,
            )
            return (
                qc_ops.gdn_gated_rmsnorm_f32(
                    y.view(-1, gDv), z, la.norm.weight, la.norm.eps
                )
                .view(m, -1)
                .clone()
            )

        gn_ref = _replay()
        gn_ref2 = _replay()
        parts.append(_stats("gdncore", gn_ref, gn))
        parts.append(_stats("replay2", gn_ref, gn_ref2))
        # Which side is actually wrong: eager's mixer output derived from
        # the CLONED boundaries (t1e = r_after - r_before, one-ulp sub).
        bb2 = bounds.get(layer_idx + 1)
        if bb2 is not None:
            t1e = (bb2[1][:m].float() - prev_r.float()).to(torch.bfloat16)
            parts.append(_stats("mixer_vs_eager", t1e, t4))
        # Escaping-write localization: restore, then diff the FULL pools
        # of this layer against their true pre-step clones.
        if full_pre is not None:

            def _rows(a, b):
                cd = (a[0] - b[0]).abs()
                sd = (a[1] - b[1]).abs()
                return (
                    torch.nonzero(cd.sum(dim=(1, 2))).flatten().tolist()[:12],
                    torch.nonzero(sd.sum(dim=(1, 2, 3))).flatten().tolist()[:12],
                )

            cur = (conv_state, ssm_state)
            eag_c, eag_s = (
                _rows(full_mid, full_pre) if full_mid is not None else ([], [])
            )
            rep_c, rep_s = _rows(cur, full_mid) if full_mid is not None else ([], [])
            restore_pre()
            res_c, res_s = _rows(cur, full_pre)
            parts.append(
                f"WRITES eager conv{eag_c} ssm{eag_s} | replays conv{rep_c} "
                f"ssm{rep_s} | after-restore-vs-pre conv{res_c} ssm{res_s} "
                f"| snapshot conv {cache.conv_slots.tolist()} ssm "
                f"{sorted(set(cache.slot_table.reshape(-1).tolist()))}"
            )
        parts.append(_stats("mixer", _host_gemv(la.out_proj, gn), t4))
    else:
        parts.append("mixer: n/a (no host reference at this cap)")

    parts.append(
        _stats(
            "res",
            (t4.float() + entry_sum.float()).to(torch.bfloat16),
            res5,
        )
    )
    h2_ref = _gemma_ref(
        (t4.float() + entry_sum.float()).to(torch.bfloat16),
        layer.post_attention_layernorm.weight,
        eps,
    )
    parts.append(_stats("gu", _host_gemv(layer.mlp.gate_up_proj, h2_ref), gu))
    parts.append(
        _stats(
            "mid",
            (
                torch.nn.functional.silu(gu[:, :inter].float()) * gu[:, inter:].float()
            ).to(torch.bfloat16),
            mid,
        )
    )
    parts.append(_stats("mlp", _host_gemv(layer.mlp.down_proj, mid), x_muse))
    logger.warning("muse_q38 DEBUG layer %d (m %d): %s", layer_idx, m, "; ".join(parts))


def _stats(tag: str, a: torch.Tensor, b: torch.Tensor) -> str:
    af, bf = a.float(), b.float()
    cos = torch.nn.functional.cosine_similarity(
        af.reshape(1, -1), bf.reshape(1, -1)
    ).item()
    mad = (af - bf).abs().max().item()
    denom = af.abs().max().item() or 1.0
    return f"{tag}: cos {cos:.8f} max|d| {mad:.3e} rel {mad / denom:.2e}"


def _shadow_step(
    model: torch.nn.Module,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    caches: list,
    attn_meta: tuple,
    q_len: int,
    m: int,
    caps: list[int],
) -> torch.Tensor:
    g_bts, g_sls, g_slots, g_maxctx, attn_group = attn_meta
    pos = positions[0] if positions.ndim == 2 else positions
    x = _S.x_buf[:m]
    res = _S.res_buf[:m]
    n = _S.num_layers
    cap_set = {c for c in caps if 0 < c < n}

    pre = _snapshot(g_slots, attn_group, caches)
    full_pre = full_mid = None
    dl_pools = None
    if _DEBUG_LAYER is not None:
        dl_layer = model.layers[_DEBUG_LAYER]
        if hasattr(dl_layer, "linear_attn"):
            dl_pools = (
                dl_layer.linear_attn.kv_cache[0],
                dl_layer.linear_attn.kv_cache[1],
            )
            full_pre = (dl_pools[0].clone(), dl_pools[1].clone())
    # Eager serves and runs the FULL (possibly padded) batch, exactly as
    # the normal path would; muse and the comparison see the actual rows.
    final_eager, aux_eager, bounds = _eager_prefix(
        model, hidden_states, positions, cap_set
    )
    if dl_pools is not None:
        full_mid = (dl_pools[0].clone(), dl_pools[1].clone())
    post = _snapshot(g_slots, attn_group, caches)

    # The muse replays are fully guarded: whatever happens, the post-eager
    # state is restored and the eager result serves — a bringup failure
    # must degrade to a log line, not corrupt serving state. Each cap in
    # the sweep replays against the restored pre-state.
    try:
        for cap in caps:
            eff = cap if 0 < cap < n else 0
            _restore(pre, g_slots, attn_group)
            x.copy_(hidden_states[:m])
            aux_out = (
                torch.zeros(
                    len(_S.aux_layers),
                    m,
                    _S.hidden,
                    dtype=torch.bfloat16,
                    device=hidden_states.device,
                )
                if _S.aux_layers and eff == 0
                else None
            )
            dbg = (
                torch.zeros(
                    8,
                    m,
                    2 * model.config.intermediate_size,
                    dtype=torch.bfloat16,
                    device=hidden_states.device,
                )
                if _DEBUG_LAYER is not None and cap == _DEBUG_LAYER + 1
                else None
            )
            _qc().muse_q38_run(
                x,
                res,
                pos[:m].contiguous(),
                g_bts,
                g_sls,
                g_slots,
                g_maxctx,
                attn_group,
                [c.spec_cu for c in caches],
                [c.conv_slots for c in caches],
                [c.slot_table for c in caches],
                [c.num_accepted for c in caches],
                q_len,
                eff,
                aux_out,
                dbg,
                _DEBUG_LAYER or 0,
            )
            parts = []
            if eff and cap in bounds:
                bh, br = bounds[cap]
                parts.append(_stats("hidden", bh[:m], x))
                parts.append(_stats("residual", br[:m], res))
            else:
                parts.append(_stats("final", final_eager[:m], x))
                if aux_out is not None:
                    for i, tap in enumerate(_S.aux_layers):
                        if i < len(aux_eager):
                            parts.append(
                                _stats(f"aux{tap}", aux_eager[i][:m], aux_out[i])
                            )
            logger.warning(
                "muse_q38 shadow (cap %d, m %d): %s", cap, m, "; ".join(parts)
            )
            if dbg is not None:
                gdn_ord = sum(
                    1
                    for i in range(_DEBUG_LAYER)
                    if hasattr(model.layers[i], "linear_attn")
                )
                dbg_cache = (
                    caches[gdn_ord]
                    if hasattr(model.layers[_DEBUG_LAYER], "linear_attn")
                    else caches[0]
                )
                _debug_report(
                    model,
                    dbg,
                    hidden_states[:m],
                    bounds,
                    dbg_cache,
                    lambda: _restore(pre, g_slots, attn_group),
                    m,
                    _DEBUG_LAYER,
                    x,
                    full_pre,
                    full_mid,
                )
    except Exception as exc:  # noqa: BLE001 — quarantine, keep serving
        _S.failed = str(exc)
        logger.warning(
            "muse_q38 shadow replay failed (eager keeps serving)",
            exc_info=True,
        )
    finally:
        _restore(post, g_slots, attn_group)
    if aux_eager:
        return final_eager, aux_eager
    return final_eager
