"""Serve ZAI's own FP8 block tensors in place of a conversion's BF16 twins.

The native GLM-5.3-Flash checkpoint stores the dense MLP, the shared experts
and the DSA ``q_b_proj`` / ``o_proj`` (and ``q_a_proj`` / ``kv_a_proj_with_mqa``)
as e4m3 with 128x128 F32 block scales; NVFP4 conversions such as
RedHatAI/GLM-5.3-Flash-NVFP4 (2026-08) keep those modules as the BF16
dequant of the same bytes, at twice the size (0.72 GB per GPU per token at
TP=4). This tool reads only the affected byte ranges from the native shards
and writes two files next to the served checkpoint:

* ``<model>/fp8-swapset.safetensors``: the FP8 weights under their checkpoint
  names and the scales renamed from ``weight_scale_inv`` to ``weight_scale``
  (the compressed-tensors parameter name);
* ``<model>/fp8-swapset.json``: the manifest - tensor list, the vLLM modules
  they load into, and the compressed-tensors config group that quantizes
  exactly those modules (copied from the checkpoint's own FP8 group so the
  schema matches what the loader parses).

When both files are present the loader substitutes the weights, injects the
scales, and the quantization config gains the group; set
``SLIMSERVE_FP8_SWAPSET=0`` to serve the BF16 twins for an A/B, or to the stem
of another manifest next to the checkpoint to serve that sidecar instead.

``self_attn.q_a_proj`` / ``kv_a_proj_with_mqa`` are left out: they load into
``fused_qkv_a_proj`` together with three indexer shards the native
checkpoint keeps in BF16, and one module takes one scheme.

``--self-quant-kda`` adds the KDA (linear-attention) projections, which every
GLM-5.3-Flash checkpoint keeps in BF16: ``q/k/v/b/f_a/g_a_proj`` (the merged
``in_proj_qkvgfab``) and the KDA ``o_proj`` are quantized here to the same
128x128 block format (per-block absmax / 448), each tensor in its checkpoint
shape, so the sidecar is tensor-parallel agnostic. The merged projection's
beta shard (``b_proj``, one row per head) is smaller than a scale block, so
the model gives it a whole replicated block of rows when the manifest lists
the module (``beta_block_rows``) and each rank reads its heads from it.
Quality is the experiment's gate, not a given.

    python -m slimserve.fp8_swapset --native /path/to/GLM-5.3-Flash \\
        --model /path/to/GLM-5.3-Flash-NVFP4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import torch

from slimserve.f32_overrides import _headers, _read

SWAPSET_FILE = "fp8-swapset.safetensors"
MANIFEST_FILE = "fp8-swapset.json"
SWAPSET_ENV = "SLIMSERVE_FP8_SWAPSET"
GROUP_NAME = "slimserve_fp8_swapset"
BLOCK = 128

# Checkpoint module suffix (after ``layers.N.``) -> vLLM module suffix it loads into.
SWAP_MODULES = {
    "mlp.gate_proj": "mlp.gate_up_proj",
    "mlp.up_proj": "mlp.gate_up_proj",
    "mlp.down_proj": "mlp.down_proj",
    "mlp.shared_experts.gate_proj": "mlp.shared_experts.gate_up_proj",
    "mlp.shared_experts.up_proj": "mlp.shared_experts.gate_up_proj",
    "mlp.shared_experts.down_proj": "mlp.shared_experts.down_proj",
    "self_attn.q_b_proj": "self_attn.q_b_proj",
    "self_attn.o_proj": "self_attn.o_proj",
    # KDA layers (self-quantized only; BF16 in the native checkpoint).
    "self_attn.q_proj": "self_attn.in_proj_qkvgfab",
    "self_attn.k_proj": "self_attn.in_proj_qkvgfab",
    "self_attn.v_proj": "self_attn.in_proj_qkvgfab",
    "self_attn.b_proj": "self_attn.in_proj_qkvgfab",
    "self_attn.f_a_proj": "self_attn.in_proj_qkvgfab",
    "self_attn.g_a_proj": "self_attn.in_proj_qkvgfab",
}
KDA_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.b_proj",
    "self_attn.f_a_proj",
    "self_attn.g_a_proj",
    "self_attn.o_proj",
)
KDA_MARKER = "self_attn.b_proj"  # a tensor only the KDA layers carry
BETA_ROWS = BLOCK  # rows the model reserves for the replicated beta shard
FP8_MAX = 448.0  # float8_e4m3fn
_LAYER_RE = re.compile(
    r"^(?P<prefix>.*\.layers\.)(?P<layer>\d+)\.(?P<suffix>.+)\.weight$"
)


def enabled() -> bool:
    return os.environ.get(SWAPSET_ENV, "1") != "0"


def manifest_path(model_path: str | None) -> str | None:
    """``<model>/fp8-swapset.json`` when present and enabled. The environment
    variable also selects another manifest by stem (``SLIMSERVE_FP8_SWAPSET=
    fp8-swapset-dense`` reads ``<model>/fp8-swapset-dense.json``), so two
    sidecars can be compared without renaming files."""
    if not model_path or not enabled():
        return None
    choice = os.environ.get(SWAPSET_ENV, "1")
    name = MANIFEST_FILE if choice == "1" else f"{choice}.json"
    path = os.path.join(model_path, name)
    return path if os.path.isfile(path) else None


def _split(name: str) -> tuple[str, int, str] | None:
    m = _LAYER_RE.match(name)
    if m is None:
        return None
    return m.group("prefix"), int(m.group("layer")), m.group("suffix")


def select(native: dict, converted: dict, skip_layer: int | None) -> list[str]:
    """Weight names that are FP8 in the native checkpoint, belong to a swap
    module, have a block-scale companion, and are BF16 in the conversion."""
    names = []
    for name, (entry, _, _) in native.items():
        parts = _split(name)
        if parts is None or entry["dtype"] != "F8_E4M3":
            continue
        _, layer, suffix = parts
        if suffix not in SWAP_MODULES or layer == skip_layer:
            continue
        scale = native.get(name[: -len(".weight")] + ".weight_scale_inv")
        other = converted.get(name)
        if scale is None or scale[0]["dtype"] != "F32":
            continue
        if other is None or other[0]["dtype"] != "BF16":
            continue
        names.append(name)
    return sorted(names)


def select_kda(converted: dict, skip_layer: int | None) -> list[str]:
    """BF16 KDA projection weights of the conversion (layers that carry the
    beta projection), to be self-quantized."""
    names = []
    for name, (entry, _, _) in converted.items():
        parts = _split(name)
        if parts is None or entry["dtype"] != "BF16":
            continue
        prefix, layer, suffix = parts
        if suffix not in KDA_SUFFIXES or layer == skip_layer:
            continue
        if f"{prefix}{layer}.{KDA_MARKER}.weight" not in converted:
            continue  # DSA layer: its o_proj is a native FP8 twin, not KDA
        names.append(name)
    return sorted(names)


def quantize_block(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """128x128 block e4m3 (per-block absmax / 448) of a [N, K] weight; N may
    end in a partial block, K must be a multiple of 128. Returns (fp8 [N, K],
    f32 scale [ceil(N/128), K/128])."""
    n, k = w.shape
    if k % BLOCK:
        raise ValueError(f"K={k} is not a multiple of {BLOCK}")
    rows = (n + BLOCK - 1) // BLOCK
    wp = torch.zeros(rows * BLOCK, k, dtype=torch.float32, device=w.device)
    wp[:n] = w.float()
    blocks = wp.view(rows, BLOCK, k // BLOCK, BLOCK)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)
    scale = amax / FP8_MAX
    q = (blocks / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.view(rows * BLOCK, k)[:n].contiguous(), scale.view(rows, k // BLOCK)


def targets_for(names: list[str]) -> list[str]:
    """compressed-tensors ``re:`` targets on the vLLM module names, one per
    (module suffix, layer set); explicit layer lists so a KDA ``o_proj``
    never matches a DSA target."""
    layers: dict[str, set[int]] = defaultdict(set)
    for name in names:
        _, layer, suffix = _split(name)
        layers[SWAP_MODULES[suffix]].add(layer)
    out = []
    for suffix in sorted(layers):
        ids = "|".join(str(i) for i in sorted(layers[suffix]))
        out.append(rf"re:.*\.layers\.({ids})\.{re.escape(suffix)}$")
    return out


def _fp8_group_template(model_dir: str) -> dict:
    """The checkpoint's own float-quantized block group (schema source)."""
    with open(os.path.join(model_dir, "config.json")) as fh:
        cfg = json.load(fh)
    qc = cfg.get("quantization_config") or cfg.get("text_config", {}).get(
        "quantization_config"
    )
    if not qc or qc.get("quant_method") != "compressed-tensors":
        raise SystemExit("the served checkpoint is not a compressed-tensors model")
    for group in qc.get("config_groups", {}).values():
        w = group.get("weights") or {}
        if (
            group.get("format") == "float-quantized"
            and w.get("strategy") == "block"
            and list(w.get("block_structure") or []) == [BLOCK, BLOCK]
        ):
            return json.loads(json.dumps(group))
    raise SystemExit(
        "no float-quantized 128x128 block group in the checkpoint's "
        "quantization_config to copy the schema from"
    )


def dequant_bf16(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    n, k = weight.shape
    rows, cols = scale.shape
    w = torch.zeros(rows * BLOCK, cols * BLOCK, dtype=torch.float32)
    w[:n, :k] = weight.float()
    w = w.view(rows, BLOCK, cols, BLOCK) * scale.view(rows, 1, cols, 1)
    return w.view(rows * BLOCK, cols * BLOCK)[:n, :k].to(torch.bfloat16)


def build(
    native_dir: str,
    model_dir: str,
    out: str | None = None,
    skip_layer: int | None = 45,
    self_quant_kda: bool = False,
) -> Path:
    native = _headers(native_dir)
    converted = _headers(model_dir)
    names = select(native, converted, skip_layer)
    if not names:
        raise SystemExit("nothing to swap: no FP8 twins of BF16 conversion tensors")
    kda = select_kda(converted, skip_layer) if self_quant_kda else []
    if self_quant_kda and not kda:
        raise SystemExit("--self-quant-kda found no BF16 KDA projections")
    group = _fp8_group_template(model_dir)
    group["targets"] = targets_for(names + kda)
    tensors: dict[str, torch.Tensor] = {}
    mismatch: dict[str, list[tuple[int, int]]] = defaultdict(list)
    qerr: dict[str, list[tuple[float, float]]] = defaultdict(list)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name in kda:
        base = name[: -len(".weight")]
        w = _read(*converted[name]).to(device)
        q, sc = quantize_block(w)
        d = dequant_bf16(q.cpu(), sc.cpu()).float()
        ref = w.float().cpu()
        err = d - ref
        key = base.split(".layers.", 1)[1].split(".", 1)[1]
        qerr[key].append(
            (
                (err.norm() / ref.norm().clamp(min=1e-12)).item(),
                (err.abs().max() / ref.abs().max().clamp(min=1e-12)).item(),
            )
        )
        tensors[name] = q.cpu()
        tensors[base + ".weight_scale"] = sc.cpu()
        del w, q, sc, d, ref, err
    for name in names:
        base = name[: -len(".weight")]
        w = _read(*native[name]).contiguous()
        s = _read(*native[base + ".weight_scale_inv"]).contiguous()
        c = _read(*converted[name])
        d = dequant_bf16(w, s)
        diff = (d.view(torch.int16).int() - c.view(torch.int16).int()).abs()
        key = base.split(".layers.", 1)[1].split(".", 1)[1]
        mismatch[key].append((int((diff > 0).sum()), int(diff.max())))
        tensors[name] = w
        tensors[base + ".weight_scale"] = s
    from safetensors.torch import save_file

    out_path = Path(out or os.path.join(model_dir, SWAPSET_FILE))
    save_file(
        tensors,
        str(out_path),
        metadata={
            "source": os.path.abspath(native_dir),
            "purpose": "FP8 block tensors of the native checkpoint (weights + "
            "weight_scale) replacing the conversion's BF16 dequant",
        },
    )
    manifest = {
        "source": os.path.abspath(native_dir),
        "file": out_path.name,
        "tensors": sorted(tensors),
        "modules": sorted(
            {
                _split(n)[0].replace("model.language_model.", "language_model.model.")
                + f"{_split(n)[1]}."
                + SWAP_MODULES[_split(n)[2]]
                for n in names + kda
            }
        ),
        "config_group": group,
    }
    if kda:
        manifest["self_quantized"] = kda
        manifest["beta_block_rows"] = BETA_ROWS
    manifest_file = out_path.with_name(MANIFEST_FILE)
    with open(manifest_file, "w") as fh:
        json.dump(manifest, fh, indent=1)
    nbytes = sum(t.numel() * t.element_size() for t in tensors.values())
    print(
        f"{len(tensors)} tensors, {nbytes} bytes -> {out_path}; "
        f"manifest {manifest_file}"
    )
    for key, rows in sorted(mismatch.items()):
        elems = sum(n for n, _ in rows)
        print(
            f"  {key:36s} n={len(rows):3d} dequant vs conversion BF16: "
            f"{elems} elements differ, max {max(m for _, m in rows)} bf16 ulp"
        )
    for key, rows in sorted(qerr.items()):
        print(
            f"  {key:36s} n={len(rows):3d} self-quantized: rel Frobenius error "
            f"mean {sum(f for f, _ in rows) / len(rows):.4f} max "
            f"{max(f for f, _ in rows):.4f}; worst element "
            f"{max(m for _, m in rows):.4f} of the tensor's absmax"
        )
    return out_path


def _manifest(model_path: str | None) -> dict | None:
    path = manifest_path(model_path)
    if path is None:
        return None
    with open(path) as fh:
        return json.load(fh)


def beta_block_rows(model_path: str | None, module: str) -> int | None:
    """Rows the KDA layer reserves for its replicated beta shard when the
    swap-set self-quantized ``module`` (a vLLM ``in_proj_qkvgfab`` name);
    None keeps the model's own layout."""
    manifest = _manifest(model_path)
    if not manifest or not manifest.get("self_quantized"):
        return None
    # Compare from ``layers.N`` on: the model's prefix above it depends on the
    # wrapping (multimodal vs text-only) while the manifest stores one form.
    tail = module.split(".layers.", 1)[-1]
    listed = {m.split(".layers.", 1)[-1] for m in manifest["modules"]}
    return manifest["beta_block_rows"] if tail in listed else None


def apply_config_group(model_path: str | None, hf_quant_config: dict | None) -> bool:
    """Add the manifest's config group to a compressed-tensors quantization
    config when the swap-set is present and enabled. Returns True if added."""
    path = manifest_path(model_path)
    if path is None or not hf_quant_config:
        return False
    if hf_quant_config.get("quant_method") != "compressed-tensors":
        raise ValueError(f"{path}: the swap-set needs a compressed-tensors model")
    with open(path) as fh:
        manifest = json.load(fh)
    groups = hf_quant_config.setdefault("config_groups", {})
    groups[GROUP_NAME] = manifest["config_group"]
    # The conversion lists the swapped modules under ``ignore`` (they were
    # unquantized BF16); the ignore check runs before target matching in
    # vLLM's compressed-tensors config, so they must leave it.
    swapped = {name.rsplit(".", 1)[0] for name in manifest["tensors"]}
    ignore = hf_quant_config.get("ignore")
    if ignore:
        hf_quant_config["ignore"] = [n for n in ignore if n not in swapped]
    return True


def hash_factor(model_path: str | None) -> str:
    """Compile-cache factor: the manifest's digest, or 'off'."""
    path = manifest_path(model_path)
    if path is None:
        return "fp8_swapset=off"
    with open(path, "rb") as fh:
        return "fp8_swapset=" + hashlib.sha256(fh.read()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--native", required=True, help="native (FP8) checkpoint dir")
    ap.add_argument("--model", required=True, help="served (converted) checkpoint dir")
    ap.add_argument("--out", help=f"output file (default <model>/{SWAPSET_FILE})")
    ap.add_argument(
        "--skip-layer", type=int, default=45, help="layer to leave out (MTP head)"
    )
    ap.add_argument(
        "--self-quant-kda",
        action="store_true",
        help="also quantize the BF16 KDA projections",
    )
    args = ap.parse_args()
    build(
        args.native,
        args.model,
        args.out,
        args.skip_layer,
        self_quant_kda=args.self_quant_kda,
    )


if __name__ == "__main__":
    main()
