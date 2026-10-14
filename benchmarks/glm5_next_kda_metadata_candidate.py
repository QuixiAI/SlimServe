# SPDX-License-Identifier: Apache-2.0
"""Quarantined pure-prefill metadata reuse; not imported by serving.

Keep all KDA arithmetic and kernels unchanged. The production GDN builder
already prepares an exact chunk table on the A100/triton path. Its table
describes *only* prefill rows in a mixed ordinary batch, so do not reuse it
for KDA's full non-spec chunk call in that case. Speculation is also excluded
from this initial candidate. A future serving integration must retain those
fallbacks and explicitly carry the table through both fused-gate wrappers.
"""

from vllm.models.kimi_k3.amd.ops.third_party.kda.chunk import (
    FLA_CHUNK_SIZE,
    _chunk_kda_fwd_with_cumulative_g,
    fused_kda_gate_chunk_cumsum,
    l2norm_fwd,
)


def pure_prefill_indices(metadata):
    if (
        metadata.num_prefills <= 0
        or metadata.num_decodes != 0
        or metadata.num_spec_decodes != 0
        or metadata.spec_sequence_masks is not None
    ):
        return None
    return metadata.chunk_indices


def chunk_kda_precomputed(
    *,
    q,
    k,
    v,
    raw_g,
    raw_beta,
    A_log,
    g_bias,
    cu_seqlens,
    chunk_indices,
    scale=None,
    initial_state=None,
    output_final_state=False,
    lower_bound=None,
    use_qk_l2norm_in_kernel=False,
):
    """Same fused-gate pipeline, supplied exact metadata instead of D2H rebuild.

    Like the baseline, the kernel may use contiguous V as its output buffer.
    Callers comparing two arms must supply independent V buffers.
    """
    assert chunk_indices is not None
    assert chunk_indices.ndim == 2 and chunk_indices.shape[1] == 2
    assert chunk_indices.device == cu_seqlens.device
    assert chunk_indices.dtype == cu_seqlens.dtype
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q.contiguous())
        k = l2norm_fwd(k.contiguous())
    g, beta = fused_kda_gate_chunk_cumsum(
        raw_g.contiguous(),
        raw_beta=raw_beta,
        A_log=A_log,
        g_bias=g_bias,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=FLA_CHUNK_SIZE,
        lower_bound=lower_bound,
    )
    return _chunk_kda_fwd_with_cumulative_g(
        q=q,
        k=k,
        v=v.contiguous(),
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state.contiguous() if initial_state is not None else None,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=FLA_CHUNK_SIZE,
        safe_gate=lower_bound is not None,
    )
