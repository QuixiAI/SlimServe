# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash (glm5_next) routing smoke on Apple Metal.

Builds the text model with a TINY random config (2 KDA layers + 1 DSA
layer, hidden 64, dense FFNs) on MPS, then runs a 13-token prefill and a
1-token decode through ``Glm5NextTextModel.forward`` under a hand-built
forward context (GDN metadata for the KDA layers, METAL_MLA_SPARSE metadata
for the DSA layer, DSV3.2-style indexer metadata for its pooled indexer)
with real paged caches. Proves, without Triton or CUDA:

* backend selection lands on METAL_MLA_SPARSE and the KDA layers on the
  torch-native core, mHC on the QuixiCore-Metal dsv4_mhc_* route (the Metal
  norm-fused epilogue when its kernel is built, no SM80 fusion, no
  projection stream),
* the latent / indexer inserts, the pooled top-k, the sparse MLA and the
  KDA state updates execute on MPS and stay finite,
* prefill(14 tokens) and prefill(13) + decode(1) agree on the last token's
  logits (cache and recurrent-state continuity across all three layer
  kinds).

Why dense FFNs: the unquantized ``FusedMoE`` has no Metal backend in
``fused_moe/oracle/unquantized.py`` (serving uses the GGUF MoE kernels);
the GLM router itself is pinned in tests/glm5_next/test_metal_router_swiglu.py.

Skips without MPS.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires Apple Metal (MPS)"
)

DEV = "mps"
HIDDEN = 64
VOCAB = 128
BS = 16  # KV block size
KDA_HEADS, KDA_HEAD_DIM = 2, 16
MLA_HEADS, KV_LORA, Q_LORA, NOPE, V_DIM = 4, 64, 32, 32, 32
INDEX_HEADS, INDEX_TOPK, KPOOL = 4, 16, 4


def _tiny_config():
    from vllm.transformers_utils.configs.glm5_next import Glm5NextTextConfig

    return Glm5NextTextConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=MLA_HEADS,
        num_key_value_heads=MLA_HEADS,
        n_shared_experts=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        kv_lora_rank=KV_LORA,
        q_lora_rank=Q_LORA,
        qk_rope_head_dim=0,
        qk_nope_head_dim=NOPE,
        v_head_dim=V_DIM,
        index_topk=INDEX_TOPK,
        index_head_dim=128,
        index_n_heads=INDEX_HEADS,
        index_kpool=KPOOL,
        index_kpool_compress=True,
        index_kpool_always_select_tail=True,
        max_position_embeddings=1024,
        layer_types=["linear_attention", "linear_attention",
                     "deepseek_sparse_attention"],
        mlp_layer_types=["dense", "dense", "dense"],
        linear_attn_config={
            "num_heads": KDA_HEADS, "head_dim": KDA_HEAD_DIM,
            "short_conv_kernel_size": 4, "gate_lower_bound": -5.0,
            "use_full_rank_gate": False,
        },
        hc_mult=4, hc_eps=1e-6, hc_sinkhorn_iters=20, rms_norm_eps=1e-5,
        swiglu_limit=10.0, hidden_act="silu", first_k_dense_replace=3,
        tie_word_embeddings=False,
    )


@pytest.fixture(scope="module")
def dist():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.utils.network_utils import get_open_port

    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1, rank=0, local_rank=0, backend="gloo",
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
        )
        ensure_model_parallel_initialized(1, 1)
    yield
    destroy_model_parallel()
    destroy_distributed_environment()


def _vllm_config(cfg):
    from vllm.config import VllmConfig

    vc = VllmConfig()
    vc.cache_config.cache_dtype = "auto"  # bf16 latent pages (no fp8 on MPS)
    vc.cache_config.block_size = BS
    vc.model_config = SimpleNamespace(
        hf_config=cfg, hf_text_config=cfg, dtype=torch.bfloat16, max_model_len=256,
        get_num_attention_heads=lambda pc: MLA_HEADS, is_attention_free=False,
        head_dtype=torch.bfloat16, logits_processor_pattern=None,
        use_mla=True,
    )
    return vc


def _build_model(cfg, vc):
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.models.glm5_next import Glm5NextForCausalLM

    torch.manual_seed(0)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with set_current_vllm_config(vc):
            model = Glm5NextForCausalLM(vllm_config=vc)
    finally:
        torch.set_default_dtype(prev)
    g = torch.Generator().manual_seed(1)
    for name, p in model.named_parameters():
        if p.dim() == 0:
            continue
        if name.endswith("norm.weight") or ".k_norm.weight" in name:
            p.data.fill_(1.0)
        elif name.endswith("k_norm.bias"):
            p.data.zero_()
        elif "hc_" in name and name.endswith("_scale"):
            p.data.fill_(1.0)
        elif name.endswith("A_log"):
            p.data.copy_(torch.randn(p.shape, generator=g) * 0.3)
        else:
            p.data.copy_((torch.randn(p.shape, generator=g) * 0.05).to(p.dtype))
    model = model.to(DEV)
    for layer in model.model.layers:
        if not layer.is_linear:
            layer.self_attn.mla_attn.mla_attn.process_weights_after_loading(
                torch.bfloat16
            )
    return model


def _attach_caches(model, num_blocks=4, slots=3):
    mla_cache = torch.zeros((num_blocks, BS, KV_LORA), dtype=torch.bfloat16,
                            device=DEV)
    idx_cache = torch.zeros((num_blocks, BS, 256), dtype=torch.bfloat16, device=DEV)
    for layer in model.model.layers:
        attn = layer.self_attn
        if layer.is_linear:
            (conv_shape, ssm_shape) = attn.get_state_shape()
            conv_dtype, ssm_dtype = attn.get_state_dtype()
            attn.kv_cache = (
                torch.zeros((slots, *conv_shape), dtype=conv_dtype, device=DEV),
                torch.zeros((slots, *ssm_shape), dtype=ssm_dtype, device=DEV),
            )
        else:
            attn.mla_attn.mla_attn.kv_cache = mla_cache
            attn.indexer.k_cache.kv_cache = idx_cache
    return mla_cache, idx_cache


def _names(model):
    layers = list(model.model.layers)
    kda = [f"model.layers.{i}.self_attn" for i, lyr in enumerate(layers)
           if lyr.is_linear]
    dsa = [i for i, lyr in enumerate(layers) if not lyr.is_linear]
    assert len(dsa) == 1
    mla_name = model.model.layers[dsa[0]].self_attn.mla_attn.mla_attn.layer_name
    idx_name = model.model.layers[dsa[0]].self_attn.indexer.k_cache.prefix
    return kda, mla_name, idx_name


def _metadata(vc, kda_names, mla_name, idx_name, block_row, ctx, new, slot):
    """One request: context `ctx` tokens after this step, `new` of them new
    (prefill when new == ctx, decode when new == 1 < ctx)."""
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
    from vllm.v1.attention.backends.mla.indexer import (
        DeepSeekV32IndexerDecodeMetadata,
        DeepseekV32IndexerMetadata,
        DeepseekV32IndexerPrefillChunkMetadata,
        DeepseekV32IndexerPrefillMetadata,
    )
    from vllm.v1.attention.backends.mla.metal_mla_sparse import (
        MetalMLASparseMetadataBuilder,
    )
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    i32 = torch.int32
    bt = torch.tensor([block_row], dtype=i32, device=DEV)
    positions = list(range(ctx - new, ctx))
    slots = [int(block_row[p // BS]) * BS + p % BS for p in positions]
    slot_mapping = torch.tensor(slots, dtype=torch.int64, device=DEV)
    is_prefill = new == ctx
    gdn = GDNAttentionMetadata(
        num_prefills=1 if is_prefill else 0,
        num_prefill_tokens=new if is_prefill else 0,
        num_decodes=0 if is_prefill else 1,
        num_decode_tokens=0 if is_prefill else 1,
        num_spec_decodes=0, num_spec_decode_tokens=0,
        num_actual_tokens=new,
        has_initial_state=torch.tensor([not is_prefill], device=DEV),
        non_spec_query_start_loc=torch.tensor([0, new], dtype=i32, device=DEV),
        non_spec_state_indices_tensor=torch.tensor([slot], dtype=i32, device=DEV),
    )
    spec = MLAAttentionSpec(block_size=BS, num_kv_heads=1, head_size=KV_LORA,
                            dtype=torch.bfloat16)
    builder = MetalMLASparseMetadataBuilder(spec, [mla_name], vc, torch.device(DEV))
    cam = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, new], dtype=i32, device=DEV),
        query_start_loc_cpu=torch.tensor([0, new], dtype=i32),
        seq_lens=torch.tensor([ctx], dtype=i32, device=DEV),
        num_reqs=1, num_actual_tokens=new, max_query_len=new, max_seq_len=ctx,
        block_table_tensor=bt, slot_mapping=slot_mapping,
        seq_lens_cpu_upper_bound=torch.tensor([ctx], dtype=i32),
    )
    mla_md = builder.build(0, cam)
    if is_prefill:
        chunk = DeepseekV32IndexerPrefillChunkMetadata(
            block_table=bt,
            cu_seqlen_ks=torch.zeros(new, dtype=i32, device=DEV),
            cu_seqlen_ke=torch.arange(1, new + 1, dtype=i32, device=DEV),
            cu_seq_lens=torch.tensor([0, ctx], dtype=i32, device=DEV),
            token_to_seq=torch.zeros(ctx, dtype=i32, device=DEV),
            total_seq_lens=ctx, max_seq_len=ctx, token_start=0, token_end=new,
            num_reqs=1,
        )
        idx_md = DeepseekV32IndexerMetadata(
            seq_lens=cam.seq_lens, max_seq_len=ctx, slot_mapping=slot_mapping,
            num_decodes=0, num_decode_tokens=0, num_prefills=1,
            num_prefill_tokens=new,
            prefill=DeepseekV32IndexerPrefillMetadata([chunk]),
        )
    else:
        idx_md = DeepseekV32IndexerMetadata(
            seq_lens=cam.seq_lens, max_seq_len=ctx, slot_mapping=slot_mapping,
            num_decodes=1, num_decode_tokens=1, num_prefills=0,
            num_prefill_tokens=0,
            decode=DeepSeekV32IndexerDecodeMetadata(
                block_table=bt, seq_lens=cam.seq_lens,
                decode_lens=torch.ones(1, dtype=i32, device=DEV),
                requires_padding=False, schedule_metadata=torch.empty(0),
            ),
        )
    attn = {n: gdn for n in kda_names}
    attn[mla_name] = mla_md
    attn[idx_name] = idx_md
    return attn, {mla_name: slot_mapping}, positions


def _run(model, vc, attn, slot_mapping, positions, input_ids):
    from vllm.forward_context import set_forward_context

    pos = torch.tensor(positions, dtype=torch.int64, device=DEV)
    ids = torch.tensor(input_ids, dtype=torch.int64, device=DEV)
    with set_forward_context(attn, vc, num_tokens=len(positions),
                             slot_mapping=slot_mapping):
        hidden = model(ids, pos)
        logits = model.compute_logits(hidden)
    return hidden, logits


@torch.no_grad()
def test_glm5_next_routes_on_metal(dist):
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.models.glm5_next import (
        _metal_mhc_norm_fusion_available,
    )
    from vllm.v1.attention.backends.mla.metal_mla_sparse import (
        MetalMLASparseImpl,
    )

    cfg = _tiny_config()
    vc = _vllm_config(cfg)
    with set_current_vllm_config(vc):
        model = _build_model(cfg, vc)
        kda_names, mla_name, idx_name = _names(model)
        dsa = model.model.layers[2]
        kda0 = model.model.layers[0]

        # ---- routing facts
        mla = dsa.self_attn.mla_attn.mla_attn
        assert mla.attn_backend.get_name() == "METAL_MLA_SPARSE"
        assert isinstance(mla.impl, MetalMLASparseImpl)
        assert mla.prefill_backend is None  # every token on the MQA path
        assert kda0.self_attn.use_native_kda is True
        # mHC norm fusion on Metal is the dsv4_mhc_pre_finalize_norm epilogue
        # (one dispatch per site), never the SM80 fusion.
        fused = _metal_mhc_norm_fusion_available()
        assert dsa._fuse_mhc_norm is fused and kda0._fuse_mhc_norm is fused
        assert not kda0._overlap_kda and not dsa._overlap_router
        assert model.model.mhc_projection_stream is None
        assert model.model.topk_indices_buffer.device.type == DEV
        assert dsa.self_attn.indexer.decode_logits.device.type == DEV
        assert dsa.self_attn.indexer.topk_tokens == 32  # roundup32(16 + 3)

        mla_cache, idx_cache = _attach_caches(model)
        block_row = [1, 2]  # block 0 / state slot 0 stay the null entries
        torch.manual_seed(5)
        tokens = torch.randint(0, VOCAB, (14,)).tolist()

        # ---- A: prefill 13, then decode token 14
        attn, sm, pos = _metadata(vc, kda_names, mla_name, idx_name, block_row,
                                  ctx=13, new=13, slot=1)
        hidden, logits = _run(model, vc, attn, sm, pos, tokens[:13])
        assert hidden.shape == (13, HIDDEN) and logits.shape == (13, VOCAB)
        assert torch.isfinite(hidden.float()).all()
        assert torch.isfinite(logits.float()).all()
        # the pooled indexer selected every token (13 <= selected_limit 19),
        # tail right after the complete pools, -1 padding
        buf = model.model.topk_indices_buffer[:13].cpu().tolist()
        for t in range(13):
            got = [v for v in buf[t] if v >= 0]
            assert sorted(got) == list(range(t + 1)), (t, buf[t])
            assert buf[t][len(got):] == [-1] * (32 - len(got))
        # latent + indexer rows landed in the pages of block 1
        assert (mla_cache[1, :13].float().abs().sum(-1) > 0).all()
        assert (idx_cache[1, :13].float().abs().sum(-1) > 0).all()
        assert (mla_cache[1, 13:].float() == 0).all()
        # KDA state was written for slot 1 only (slot 0 is the null slot)
        conv, ssm = kda0.self_attn.kv_cache
        assert ssm[1].float().abs().sum() > 0 and conv[1].float().abs().sum() > 0
        assert ssm[0].float().abs().sum() == 0 and ssm[2].float().abs().sum() == 0

        attn, sm, pos = _metadata(vc, kda_names, mla_name, idx_name, block_row,
                                  ctx=14, new=1, slot=1)
        _, dec_logits = _run(model, vc, attn, sm, pos, tokens[13:14])
        assert dec_logits.shape == (1, VOCAB)
        assert torch.isfinite(dec_logits.float()).all()
        buf = model.model.topk_indices_buffer[0].cpu().tolist()
        assert sorted(v for v in buf if v >= 0) == list(range(14))
        assert (mla_cache[1, 13].float().abs().sum() > 0)

        # ---- B: fresh prefill of all 14 tokens (new caches, slot 2)
        mla_cache_b, idx_cache_b = _attach_caches(model)
        attn, sm, pos = _metadata(vc, kda_names, mla_name, idx_name, block_row,
                                  ctx=14, new=14, slot=2)
        _, logits_b = _run(model, vc, attn, sm, pos, tokens)
        a = dec_logits[0].float()
        b = logits_b[13].float()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        assert math.isfinite(cos) and cos > 0.98, cos
        torch.testing.assert_close(a, b, atol=0.15, rtol=0.05)
        assert torch.equal(a.argmax(), b.argmax())


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-q"])
