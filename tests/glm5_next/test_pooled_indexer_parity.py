# Parity: our paged pooled selection vs transformers' Glm5NextTextIndexer
# on one sequence with random weights. Compares the selected token SETS per
# query row (order-free), which is what sparse attention consumes.
import sys, torch
sys.path.insert(0, "/home/ubuntu/SlimServe")
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextIndexer
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
from vllm.model_executor.layers.glm5_next_indexer import _pooled_select, _ROW_DIM

torch.manual_seed(0)
dev = "cuda:0"
cfg = Glm5NextTextConfig(
    hidden_size=256, q_lora_rank=64, index_n_heads=32, index_head_dim=128,
    index_topk=64, index_kpool=4, index_kpool_compress=True,
    index_kpool_always_select_tail=True, qk_rope_head_dim=0,
)
ref = Glm5NextTextIndexer(cfg, layer_idx=0).to(dev).to(torch.bfloat16)
with torch.no_grad():
    for p in ref.parameters():
        p.normal_(0, 0.2)
    ref.k_norm.weight.fill_(1.0); ref.k_norm.bias.zero_()

L = 61  # tokens: 15 full pools + 1-token tail
x = torch.randn(1, L, cfg.hidden_size, device=dev, dtype=torch.bfloat16)
qr = torch.randn(1, L, cfg.q_lora_rank, device=dev, dtype=torch.bfloat16)
mask = torch.ones(1, L, dtype=torch.bool, device=dev)
with torch.no_grad():
    ref_idx = ref(x, qr, mask, None)  # [1, L, topk + kp - 1]

# ---- ours: build the cached rows exactly as the module does
with torch.no_grad():
    q = ref.wq_b(qr[0]).view(L, cfg.index_n_heads, cfg.index_head_dim).to(torch.bfloat16)
    k = torch.nn.functional.layer_norm(ref.wk(x[0]).float(), (cfg.index_head_dim,), ref.k_norm.weight.float(), ref.k_norm.bias.float(), 1e-6).to(torch.bfloat16)
    gate = torch.nn.functional.linear(x[0], ref.index_kpool_compress_gate).to(torch.bfloat16)
    w = (ref.weights_proj(x[0]).float() * cfg.index_n_heads ** -0.5).contiguous()
    ape = ref.index_kpool_compress_ape.float().contiguous()
    BS = 16
    nblk = (L + BS - 1) // BS
    cache = torch.zeros(nblk * BS, _ROW_DIM, device=dev, dtype=torch.bfloat16)
    cache[:L] = torch.cat([k, gate], -1)
    bt = torch.arange(nblk, device=dev, dtype=torch.int32).view(1, -1)
    row_req = torch.zeros(L, dtype=torch.int32, device=dev)
    visible = torch.arange(1, L + 1, dtype=torch.int32, device=dev)
    ksel = cfg.index_topk // cfg.index_kpool
    width = (cfg.index_topk + cfg.index_kpool - 1 + 31) // 32 * 32
    out = torch.full((L, width), -1, dtype=torch.int32, device=dev)
    logits = torch.empty((L, (L + 3) // 4), dtype=torch.float32, device=dev)
    _pooled_select(q.contiguous(), w, ape, cache, bt, row_req, visible, logits,
                   logits.shape[1], BS, cfg.index_head_dim ** -0.5, ksel, out, cfg.index_kpool)
torch.cuda.synchronize()

bad = 0
for r in range(L):
    a = set(int(v) for v in ref_idx[0, r].tolist() if v >= 0)
    b = set(int(v) for v in out[r].tolist() if v >= 0)
    if a != b:
        bad += 1
        if bad <= 3:
            print(f"row {r}: ref-only {sorted(a-b)[:8]} ours-only {sorted(b-a)[:8]} | sizes {len(a)} vs {len(b)}")
print("POOLED INDEXER PARITY:", "PASS" if bad == 0 else f"FAIL ({bad}/{L} rows differ)")
