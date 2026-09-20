# SPDX-License-Identifier: Apache-2.0
"""NVFP4 (W4A16) sidecar for the dense projections of GLM-5.3-Flash.

The served conversion keeps the dense projections in BF16 and the FP8 swap-set
(``slimserve.fp8_swapset``) serves most of them as block-FP8, which at decode
is the largest byte stream after the experts (1.9 GB per rank per step on the
rtx6000 record, 2026-09-17 physics-floor entry). This sidecar quantizes the
same modules to NVFP4 - e2m1 values, e4m3 scales per 16 inputs, one fp32
global scale per merged module, the experts' own recipe - and serves them as a
compressed-tensors ``NVFP4A16`` config group, which vLLM runs through the
Marlin W4A16 NVFP4 dense kernel. Bytes halve; the per-family logprob canary is
the gate, and ``SLIMSERVE_NVFP4_FAMILIES`` limits the sidecar to a subset of
its families for that canary or for a record that fails one family.

Families (vLLM module suffix <- checkpoint shards, all BF16 in the
conversion): ``self_attn.in_proj_qkvgfab`` (q, k, v, b, f_a, g_a of the KDA
layers), ``self_attn.o_proj`` (KDA and DSA layers), ``self_attn.q_b_proj``,
``mlp.gate_up_proj`` / ``mlp.down_proj`` (the dense layers),
``mlp.shared_experts.gate_up_proj`` / ``mlp.shared_experts.down_proj``. The
MTP layer, the lm_head (FP8 channel), ``fused_qkv_a_proj``, ``kv_b_proj``
(absorbed) and the K=128 gate projections stay as they are.

    python -m slimserve.nvfp4_swapset --model /path/to/GLM-5.3-Flash-NVFP4

Serving: ``SLIMSERVE_NVFP4_SWAPSET=nvfp4-swapset`` (a manifest stem next to
the checkpoint; unset or 0 = off). Modules the sidecar takes leave the FP8
swap-set's config group and overrides.
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

SWAPSET_ENV = "SLIMSERVE_NVFP4_SWAPSET"
FAMILIES_ENV = "SLIMSERVE_NVFP4_FAMILIES"
DEFAULT_STEM = "nvfp4-swapset"
GROUP_NAME = "slimserve_nvfp4_swapset"
GROUP_SIZE = 16
E4M3_MAX = 448.0
E2M1_MAX = 6.0
# e2m1 magnitudes by code 0..7
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

# checkpoint shard suffix -> vLLM module suffix
SWAP_MODULES = {
    "self_attn.q_proj": "self_attn.in_proj_qkvgfab",
    "self_attn.k_proj": "self_attn.in_proj_qkvgfab",
    "self_attn.v_proj": "self_attn.in_proj_qkvgfab",
    "self_attn.b_proj": "self_attn.in_proj_qkvgfab",
    "self_attn.f_a_proj": "self_attn.in_proj_qkvgfab",
    "self_attn.g_a_proj": "self_attn.in_proj_qkvgfab",
    "self_attn.o_proj": "self_attn.o_proj",
    "self_attn.q_b_proj": "self_attn.q_b_proj",
    "mlp.gate_proj": "mlp.gate_up_proj",
    "mlp.up_proj": "mlp.gate_up_proj",
    "mlp.down_proj": "mlp.down_proj",
    "mlp.shared_experts.gate_proj": "mlp.shared_experts.gate_up_proj",
    "mlp.shared_experts.up_proj": "mlp.shared_experts.gate_up_proj",
    "mlp.shared_experts.down_proj": "mlp.shared_experts.down_proj",
}
FAMILIES = sorted(set(SWAP_MODULES.values()))
KDA_MARKER = "self_attn.b_proj"
_LAYER_RE = re.compile(r"^(?P<prefix>.*\.layers\.)(?P<layer>\d+)\.(?P<suffix>.+)\.weight$")


def _split(name: str) -> tuple[str, int, str] | None:
    m = _LAYER_RE.match(name)
    return None if m is None else (m.group("prefix"), int(m.group("layer")), m.group("suffix"))


# ----------------------------------------------------------------- quantize
def quantize_nvfp4(
    w: torch.Tensor, global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """NVFP4 of a [N, K] weight with a given fp32 global scale (E4M3_MAX *
    E2M1_MAX / absmax of the merged module): returns (packed uint8 [N, K/2],
    low nibble = even column; e4m3 scales [N, K/16]). Same rounding as
    vLLM's ``ref_nvfp4_quant``."""
    n, k = w.shape
    if k % GROUP_SIZE:
        raise ValueError(f"K={k} is not a multiple of {GROUP_SIZE}")
    x = w.float().reshape(n, k // GROUP_SIZE, GROUP_SIZE)
    vec_max = x.abs().amax(dim=-1, keepdim=True)
    scale = (global_scale * (vec_max / E2M1_MAX)).clamp(max=E4M3_MAX)
    scale = scale.to(torch.float8_e4m3fn)
    scale_f = scale.float()
    inv = torch.where(scale_f == 0, torch.zeros_like(scale_f), global_scale / scale_f)
    scaled = (x * inv).clamp(-E2M1_MAX, E2M1_MAX).reshape(n, k)
    sign = scaled < 0
    mag = scaled.abs()
    # nearest e2m1 magnitude (ties as ref_nvfp4_quant's cast_to_fp4)
    code = torch.zeros_like(mag, dtype=torch.uint8)
    code[mag > 0.25] = 1
    code[mag >= 0.75] = 2
    code[mag > 1.25] = 3
    code[mag >= 1.75] = 4
    code[mag > 2.5] = 5
    code[mag >= 3.5] = 6
    code[mag > 5.0] = 7
    code = code | (sign.to(torch.uint8) << 3)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    return packed, scale.reshape(n, k // GROUP_SIZE).contiguous()


def dequant_nvfp4(
    packed: torch.Tensor, scale: torch.Tensor, global_scale: torch.Tensor
) -> torch.Tensor:
    n = packed.shape[0]
    low = packed & 0x0F
    high = packed >> 4
    code = torch.stack((low, high), dim=-1).reshape(n, -1)
    mag = _E2M1.to(packed.device)[(code & 0x07).long()]
    val = torch.where((code & 0x08).bool(), -mag, mag)
    k = val.shape[1]
    val = val.reshape(n, k // GROUP_SIZE, GROUP_SIZE) * scale.float().unsqueeze(-1)
    return (val / global_scale).reshape(n, k)


def global_scale_for(tensors: list[torch.Tensor]) -> torch.Tensor:
    amax = max(t.float().abs().amax().item() for t in tensors)
    return torch.tensor(E4M3_MAX * E2M1_MAX / max(amax, 1e-12), dtype=torch.float32)


# -------------------------------------------------------------------- build
def select(converted: dict, skip_layer: int | None, families: list[str]) -> dict:
    """{(prefix, layer, vLLM module suffix): [checkpoint weight names]} of the
    BF16 conversion tensors of the requested families."""
    out: dict[tuple[str, int, str], list[str]] = defaultdict(list)
    for name, (entry, _, _) in converted.items():
        parts = _split(name)
        if parts is None or entry["dtype"] != "BF16":
            continue
        prefix, layer, suffix = parts
        module = SWAP_MODULES.get(suffix)
        if module is None or module not in families or layer == skip_layer:
            continue
        if module == "self_attn.in_proj_qkvgfab" and f"{prefix}{layer}.{KDA_MARKER}.weight" not in converted:
            continue
        out[(prefix, layer, module)].append(name)
    return {k: sorted(v) for k, v in sorted(out.items())}


def config_group(modules: list[tuple[int, str]]) -> dict:
    layers: dict[str, set[int]] = defaultdict(set)
    for layer, module in modules:
        layers[module].add(layer)
    targets = []
    for module in sorted(layers):
        ids = "|".join(str(i) for i in sorted(layers[module]))
        targets.append(rf"re:.*\.layers\.({ids})\.{re.escape(module)}$")
    return {
        "targets": targets,
        "format": "nvfp4-pack-quantized",
        "input_activations": None,
        "output_activations": None,
        "weights": {
            "num_bits": 4,
            "type": "float",
            "strategy": "tensor_group",
            "group_size": GROUP_SIZE,
            "symmetric": True,
            "dynamic": False,
            "observer": "static_minmax",
            "actorder": None,
            "block_structure": None,
            "scale_dtype": "torch.float8_e4m3fn",
        },
    }


def build(model_dir: str, out: str | None, skip_layer: int | None, families: list[str]) -> Path:
    converted = _headers(model_dir)
    groups = select(converted, skip_layer, families)
    if not groups:
        raise SystemExit("nothing to quantize: no BF16 tensors of the requested families")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensors: dict[str, torch.Tensor] = {}
    qerr: dict[str, list[float]] = defaultdict(list)
    for (prefix, layer, module), names in groups.items():
        ws = {n: _read(*converted[n]).to(device) for n in names}
        gs = global_scale_for(list(ws.values())).to(device)
        for name, w in ws.items():
            base = name[: -len(".weight")]
            packed, scale = quantize_nvfp4(w, gs)
            err = (dequant_nvfp4(packed, scale, gs) - w.float()).norm() / w.float().norm().clamp(min=1e-12)
            qerr[module].append(err.item())
            tensors[base + ".weight_packed"] = packed.cpu()
            tensors[base + ".weight_scale"] = scale.cpu()
            tensors[base + ".weight_global_scale"] = gs.reshape(1).cpu()
        del ws
    from safetensors.torch import save_file

    out_path = Path(out or os.path.join(model_dir, DEFAULT_STEM + ".safetensors"))
    save_file(tensors, str(out_path), metadata={"purpose": "NVFP4 (W4A16, group 16) sidecar of the dense projections"})
    modules = sorted({(layer, module) for (_, layer, module) in groups})
    manifest = {
        "file": out_path.name,
        "families": sorted({m for _, m in modules}),
        "tensors": sorted(tensors),
        "modules": [f"layers.{layer}.{module}" for layer, module in modules],
        "shards": {n[: -len(".weight")]: SWAP_MODULES[_split(n)[2]] for names in groups.values() for n in names},
        "config_group": config_group(modules),
    }
    manifest_file = out_path.with_suffix(".json")
    with open(manifest_file, "w") as fh:
        json.dump(manifest, fh, indent=1)
    nbytes = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"{len(tensors)} tensors, {nbytes} bytes -> {out_path}; manifest {manifest_file}")
    for module, rows in sorted(qerr.items()):
        print(f"  {module:40s} n={len(rows):3d} rel Frobenius error mean {sum(rows) / len(rows):.4f} max {max(rows):.4f}")
    return out_path


# ------------------------------------------------------------------ serving
def enabled() -> bool:
    return os.environ.get(SWAPSET_ENV, "0") not in ("", "0")


def manifest_path(model_path: str | None) -> str | None:
    if not model_path or not enabled():
        return None
    choice = os.environ.get(SWAPSET_ENV, "0")
    stem = DEFAULT_STEM if choice == "1" else choice
    path = os.path.join(model_path, f"{stem}.json")
    return path if os.path.isfile(path) else None


def _families_filter() -> set[str] | None:
    raw = os.environ.get(FAMILIES_ENV, "").strip()
    if not raw:
        return None
    chosen = {f.strip() for f in raw.split(",") if f.strip()}
    unknown = chosen - set(FAMILIES)
    if unknown:
        raise ValueError(f"{FAMILIES_ENV}: unknown families {sorted(unknown)}; known {FAMILIES}")
    return chosen


def load_manifest(model_path: str | None) -> dict | None:
    """The manifest restricted to the families in ``SLIMSERVE_NVFP4_FAMILIES``
    (all of its families when unset); None when the sidecar is off."""
    path = manifest_path(model_path)
    if path is None:
        return None
    with open(path) as fh:
        manifest = json.load(fh)
    chosen = _families_filter()
    if chosen is not None:
        keep = [m for m in manifest["modules"] if m.split(".", 2)[2] in chosen]
        manifest["modules"] = keep
        manifest["families"] = [f for f in manifest["families"] if f in chosen]
        manifest["shards"] = {s: m for s, m in manifest["shards"].items() if m in chosen}
        manifest["tensors"] = [t for t in manifest["tensors"] if t.rsplit(".", 1)[0] in manifest["shards"]]
        manifest["config_group"] = config_group(
            [(int(m.split(".")[1]), m.split(".", 2)[2]) for m in keep]
        )
    manifest["path"] = path
    return manifest


def claimed_modules(model_path: str | None) -> set[str]:
    """``layers.N.<module>`` names the sidecars serve (empty when off): the
    dense families of the NVFP4 sidecar and the shared experts the
    shared-expert sidecar turns into routed-format experts."""
    manifest = load_manifest(model_path)
    claimed = set(manifest["modules"]) if manifest else set()
    shared = load_shared_manifest(model_path)
    if shared:
        claimed |= set(shared["modules"])
    return claimed


def apply_config_group(model_path: str | None, hf_quant_config: dict | None, fp8_group_name: str | None) -> bool:
    """Add the NVFP4A16 config group; take its modules out of ``ignore`` and
    out of the FP8 swap-set's group targets (the FP8 group is keyed by
    ``fp8_group_name`` when present)."""
    manifest = load_manifest(model_path)
    dense = manifest is not None and bool(manifest["modules"])
    # The shared-expert sidecar claims its layers' shared-expert modules too
    # (served as expert E through the checkpoint's own NVFP4 experts group).
    shared = load_shared_manifest(model_path)
    claimed: set[str] = set(shared["modules"]) if shared else set()
    if dense:
        claimed |= set(manifest["modules"])
    if not claimed or not hf_quant_config:
        return False
    if hf_quant_config.get("quant_method") != "compressed-tensors":
        raise ValueError(f"{(manifest or shared)['path']}: the sidecar needs a compressed-tensors model")
    groups = hf_quant_config.setdefault("config_groups", {})
    if dense:
        groups[GROUP_NAME] = manifest["config_group"]
    ignore = hf_quant_config.get("ignore")
    if ignore:
        hf_quant_config["ignore"] = [
            n for n in ignore if not any(n.endswith(m) for m in claimed)
        ]
    fp8 = groups.get(fp8_group_name) if fp8_group_name else None
    if fp8:
        fp8["targets"] = _trim_targets(fp8["targets"], claimed)
    return True


def _trim_targets(targets: list[str], claimed: set[str]) -> list[str]:
    """Drop the claimed (layer, module) pairs from ``re:.*\\.layers\\.(ids)\\.module$`` targets."""
    out = []
    pat = re.compile(r"^re:\.\*\\\.layers\\\.\((?P<ids>[0-9|]+)\)\\\.(?P<module>.+)\$$")
    for t in targets:
        m = pat.match(t)
        if m is None:
            out.append(t)
            continue
        module = m.group("module").replace("\\.", ".")
        ids = [i for i in m.group("ids").split("|") if f"layers.{i}.{module}" not in claimed]
        if ids:
            out.append(rf"re:.*\.layers\.({'|'.join(ids)})\.{re.escape(module)}$")
    return out


def hash_factor(model_path: str | None) -> str:
    manifest = load_manifest(model_path)
    if manifest is None:
        return "nvfp4_swapset=off"
    with open(manifest["path"], "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()[:16]
    return f"nvfp4_swapset={digest}:{','.join(sorted({m.split('.', 2)[2] for m in manifest['modules']}))}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="served (converted) checkpoint dir")
    ap.add_argument("--out", help=f"output file (default <model>/{DEFAULT_STEM}.safetensors)")
    ap.add_argument("--skip-layer", type=int, default=45, help="layer to leave alone (the MTP layer)")
    ap.add_argument("--families", default=",".join(FAMILIES), help="comma-separated vLLM module suffixes")
    ap.add_argument("--mtp-experts", action="store_true", help=f"build the MTP layer's experts sidecar ({MTP_STEM}) instead")
    ap.add_argument("--shared-experts", action="store_true",
                    help=f"build the shared-expert sidecar ({SHARED_STEM}: each sparse layer's shared expert as one more NVFP4 expert) instead")
    args = ap.parse_args()
    if args.mtp_experts:
        build_mtp_experts(args.model, args.out, args.skip_layer)
        return
    if args.shared_experts:
        build_shared_experts(args.model, args.out, args.skip_layer)
        return
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    unknown = set(families) - set(FAMILIES)
    if unknown:
        raise SystemExit(f"unknown families {sorted(unknown)}; known {FAMILIES}")
    build(args.model, args.out, args.skip_layer, families)



# ------------------------------------------------- the MTP layer's experts
# The conversion left the MTP (draft) layer's 288 routed experts in block-FP8
# (25 MB per expert, 1.8 GB per rank), so every draft step streams twice the
# bytes of a target MoE layer and Marlin's W8A16 path serves it instead of the
# NVFP4 decode pair. This sidecar re-quantizes them with the experts' own
# NVFP4 recipe (one global scale per expert for gate+up, one for down) and
# serves them through the checkpoint's NVFP4 experts config group.
#     python -m slimserve.nvfp4_swapset --model <dir> --mtp-experts
# Serving: ``SLIMSERVE_NVFP4_MTP_SWAPSET=nvfp4-swapset-mtp`` (a manifest stem
# next to the checkpoint; unset or 0 = off).
MTP_ENV = "SLIMSERVE_NVFP4_MTP_SWAPSET"
MTP_STEM = "nvfp4-swapset-mtp"
_MTP_EXPERT_RE = re.compile(r"^(?P<base>.*\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<e>\d+)\.(?P<proj>gate_proj|up_proj|down_proj))\.weight$")


def build_mtp_experts(model_dir: str, out: str | None, layer: int) -> Path:
    from slimserve.fp8_swapset import dequant_bf16

    converted = _headers(model_dir)
    experts: dict[int, dict[str, str]] = defaultdict(dict)
    for name, (entry, _, _) in converted.items():
        m = _MTP_EXPERT_RE.match(name)
        if m is None or int(m.group("layer")) != layer or entry["dtype"] != "F8_E4M3":
            continue
        experts[int(m.group("e"))][m.group("proj")] = m.group("base")
    if not experts:
        raise SystemExit(f"no FP8 experts found in layer {layer}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensors: dict[str, torch.Tensor] = {}
    consume: list[str] = []
    qerr: dict[str, list[float]] = defaultdict(list)
    for e in sorted(experts):
        projs = experts[e]
        if set(projs) != {"gate_proj", "up_proj", "down_proj"}:
            raise SystemExit(f"expert {e}: incomplete projections {sorted(projs)}")
        ws = {}
        for proj, base in projs.items():
            w = _read(*converted[base + ".weight"])
            sc = _read(*converted[base + ".weight_scale"])
            ws[proj] = dequant_bf16(w, sc.float()).to(device)
            consume += [base + ".weight", base + ".weight_scale"]
        gs13 = global_scale_for([ws["gate_proj"], ws["up_proj"]]).to(device)
        gs2 = global_scale_for([ws["down_proj"]]).to(device)
        for proj, base in projs.items():
            gs = gs2 if proj == "down_proj" else gs13
            packed, scale = quantize_nvfp4(ws[proj], gs)
            err = (dequant_nvfp4(packed, scale, gs) - ws[proj].float()).norm() / ws[proj].float().norm().clamp(min=1e-12)
            qerr[proj].append(err.item())
            tensors[base + ".weight_packed"] = packed.cpu()
            tensors[base + ".weight_scale"] = scale.cpu()
            tensors[base + ".weight_global_scale"] = gs.reshape(1).cpu()
            # W4A16 through Marlin never quantizes the activations; the loader still
            # wants the parameter, as the target's experts carry it.
            tensors[base + ".input_global_scale"] = torch.ones(1, dtype=torch.float32)
        del ws
    from safetensors.torch import save_file

    out_path = Path(out or os.path.join(model_dir, MTP_STEM + ".safetensors"))
    save_file(tensors, str(out_path), metadata={"purpose": f"NVFP4 (W4A16, group 16) sidecar of layer {layer}'s routed experts"})
    manifest = {
        "file": out_path.name,
        "kind": "mtp_experts",
        "layer": layer,
        "experts": len(experts),
        "tensors": sorted(tensors),
        "consume": sorted(consume),
        "target": rf"re:.*\.layers\.{layer}\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$",
    }
    manifest_file = out_path.with_suffix(".json")
    with open(manifest_file, "w") as fh:
        json.dump(manifest, fh, indent=1)
    nbytes = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"{len(tensors)} tensors, {nbytes} bytes -> {out_path}; manifest {manifest_file}")
    for proj, rows in sorted(qerr.items()):
        print(f"  {proj:12s} n={len(rows):3d} rel Frobenius error mean {sum(rows) / len(rows):.4f} max {max(rows):.4f}")
    return out_path


def mtp_manifest_path(model_path: str | None) -> str | None:
    choice = os.environ.get(MTP_ENV, "0")
    if not model_path or choice in ("", "0"):
        return None
    stem = MTP_STEM if choice == "1" else choice
    path = os.path.join(model_path, f"{stem}.json")
    return path if os.path.isfile(path) else None


def load_mtp_manifest(model_path: str | None) -> dict | None:
    path = mtp_manifest_path(model_path)
    if path is None:
        return None
    with open(path) as fh:
        manifest = json.load(fh)
    manifest["path"] = path
    return manifest


def apply_mtp_config_group(model_path: str | None, hf_quant_config: dict | None) -> bool:
    """Move the MTP layer's experts from their FP8 group into the checkpoint's
    NVFP4 experts group (the one whose weights are 4-bit tensor_group)."""
    manifest = load_mtp_manifest(model_path)
    if manifest is None or not hf_quant_config:
        return False
    if hf_quant_config.get("quant_method") != "compressed-tensors":
        raise ValueError(f"{manifest['path']}: the sidecar needs a compressed-tensors model")
    groups = hf_quant_config.get("config_groups") or {}
    sample = f"model.language_model.layers.{manifest['layer']}.mlp.experts.0.gate_proj"
    nvfp4 = None
    for name, g in list(groups.items()):
        targets = g.get("targets") or []
        w = g.get("weights") or {}
        if w.get("num_bits") == 4 and w.get("strategy") == "tensor_group" and nvfp4 is None and any("experts" in t for t in targets):
            nvfp4 = g
            continue
        kept = [t for t in targets if not (t.startswith("re:") and re.match(t[3:], sample))]
        if len(kept) != len(targets):
            if kept:
                g["targets"] = kept
            else:
                del groups[name]
    if nvfp4 is None:
        raise ValueError(f"{manifest['path']}: no NVFP4 experts config group to join")
    if manifest["target"] not in nvfp4["targets"]:
        nvfp4["targets"] = list(nvfp4["targets"]) + [manifest["target"]]
    return True


def mtp_hash_factor(model_path: str | None) -> str:
    manifest = load_mtp_manifest(model_path)
    if manifest is None:
        return "nvfp4_mtp_swapset=off"
    with open(manifest["path"], "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()[:16]
    return f"nvfp4_mtp_swapset={digest}"


# ------------------------------------------------- the shared experts as experts
# GLM-5.3-Flash's shared expert has exactly a routed expert's per-rank shape
# (intermediate 2048), and at decode it ran as three fp8 launches on a side
# stream that contends with the NVFP4 decode pair for DRAM. This sidecar
# re-quantizes every sparse layer's shared expert with the experts' NVFP4
# recipe and emits it as expert index E (= n_routed_experts) of that layer, so
# the layer serves it as one more routed-format expert: the router appends it
# to every token's slots at weight 1.0 (the routed scaling stays on the routed
# slots) and the MoE kernels see E + 1 experts. The fp8 swap-set's
# shared-expert modules are dropped for those layers.
#     python -m slimserve.nvfp4_swapset --model <dir> --shared-experts
# Serving: ``SLIMSERVE_NVFP4_SHARED_SWAPSET=nvfp4-swapset-shared`` (a manifest
# stem next to the checkpoint; unset or 0 = off).
SHARED_ENV = "SLIMSERVE_NVFP4_SHARED_SWAPSET"
SHARED_STEM = "nvfp4-swapset-shared"
_SHARED_RE = re.compile(r"^(?P<prefix>.*\.layers\.)(?P<layer>\d+)\.mlp\.shared_experts\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$")


def build_shared_experts(model_dir: str, out: str | None, skip_layer: int | None) -> Path:
    with open(os.path.join(model_dir, "config.json")) as fh:
        cfg = json.load(fh)
    cfg = cfg.get("text_config", cfg)
    expert = int(cfg["n_routed_experts"])
    converted = _headers(model_dir)
    layers: dict[int, dict[str, tuple[str, str]]] = defaultdict(dict)
    for name, (entry, *_rest) in converted.items():
        m = _SHARED_RE.match(name)
        if m is None or entry["dtype"] != "BF16" or int(m.group("layer")) == skip_layer:
            continue
        layers[int(m.group("layer"))][m.group("proj")] = (m.group("prefix"), name)
    if not layers:
        raise SystemExit("no BF16 shared experts found")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensors: dict[str, torch.Tensor] = {}
    consume: list[str] = []
    qerr: dict[str, list[float]] = defaultdict(list)
    for layer in sorted(layers):
        projs = layers[layer]
        if set(projs) != {"gate_proj", "up_proj", "down_proj"}:
            raise SystemExit(f"layer {layer}: incomplete shared expert {sorted(projs)}")
        ws = {p: _read(*converted[n]).to(device) for p, (_, n) in projs.items()}
        gs13 = global_scale_for([ws["gate_proj"], ws["up_proj"]]).to(device)
        gs2 = global_scale_for([ws["down_proj"]]).to(device)
        for proj, (prefix, name) in projs.items():
            gs = gs2 if proj == "down_proj" else gs13
            packed, scale = quantize_nvfp4(ws[proj], gs)
            err = (dequant_nvfp4(packed, scale, gs) - ws[proj].float()).norm() / ws[proj].float().norm().clamp(min=1e-12)
            qerr[proj].append(err.item())
            base = f"{prefix}{layer}.mlp.experts.{expert}.{proj}"
            tensors[base + ".weight_packed"] = packed.cpu()
            tensors[base + ".weight_scale"] = scale.cpu()
            tensors[base + ".weight_global_scale"] = gs.reshape(1).cpu().clone()
            tensors[base + ".input_global_scale"] = torch.ones(1, dtype=torch.float32)
            consume.append(name)
        del ws
    from safetensors.torch import save_file

    out_path = Path(out or os.path.join(model_dir, SHARED_STEM + ".safetensors"))
    save_file(tensors, str(out_path), metadata={"purpose": f"NVFP4 (W4A16, group 16) sidecar: each sparse layer's shared expert as expert {expert}"})
    manifest = {
        "file": out_path.name,
        "kind": "shared_experts",
        "expert": expert,
        "layers": sorted(layers),
        "tensors": sorted(tensors),
        "consume": sorted(consume),
        "modules": [f"layers.{layer}.mlp.shared_experts.{m}" for layer in sorted(layers) for m in ("gate_up_proj", "down_proj")],
    }
    manifest_file = out_path.with_suffix(".json")
    with open(manifest_file, "w") as fh:
        json.dump(manifest, fh, indent=1)
    nbytes = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"{len(tensors)} tensors, {nbytes} bytes -> {out_path}; manifest {manifest_file}")
    for proj, rows in sorted(qerr.items()):
        print(f"  {proj:12s} n={len(rows):3d} rel Frobenius error mean {sum(rows) / len(rows):.4f} max {max(rows):.4f}")
    return out_path


def shared_manifest_path(model_path: str | None) -> str | None:
    choice = os.environ.get(SHARED_ENV, "0")
    if not model_path or choice in ("", "0"):
        return None
    stem = SHARED_STEM if choice == "1" else choice
    path = os.path.join(model_path, f"{stem}.json")
    return path if os.path.isfile(path) else None


def load_shared_manifest(model_path: str | None) -> dict | None:
    path = shared_manifest_path(model_path)
    if path is None:
        return None
    with open(path) as fh:
        manifest = json.load(fh)
    manifest["path"] = path
    return manifest


def shared_expert_layers(model_path: str | None) -> set[int]:
    """Layers whose shared expert the sidecar serves as expert E (empty when off)."""
    manifest = load_shared_manifest(model_path)
    return set(manifest["layers"]) if manifest else set()


def shared_hash_factor(model_path: str | None) -> str:
    manifest = load_shared_manifest(model_path)
    if manifest is None:
        return "nvfp4_shared_swapset=off"
    with open(manifest["path"], "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()[:16]
    return f"nvfp4_shared_swapset={digest}"


if __name__ == "__main__":
    main()
