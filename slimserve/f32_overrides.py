"""Rebuild the F32 tensors that an NVFP4 conversion of GLM-5.3-Flash downcast.

RedHatAI/GLM-5.3-Flash-NVFP4 (2026-08) stores the MoE router bias
(``mlp.gate.e_score_correction_bias``), the KDA decay tensors (``A_log``,
``dt_bias``) and the mHC base/scale vectors as BF16, while the native
checkpoint keeps them in F32 and the serving model holds F32 parameters for
them. The BF16 copies carry up to a 0.39% relative error (one BF16 ulp at
values of 5-15), which is the same order as the per-layer spread of the
router bias itself.

This tool reads only the affected byte ranges from the native shards (no
full load) and writes ``<model>/f32-overrides.safetensors``; the glm5_next
loader substitutes those tensors when the file is present.

The partial reads are offline preparation, not a claimed serving-speed gain.
The F32 repair's model comparison and artifact record are linked in
docs/glm53-flash-sm120-review.md, "Qualification record map".

    python -m slimserve.f32_overrides --native /path/to/GLM-5.3-Flash \\
        --model /path/to/GLM-5.3-Flash-NVFP4
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import struct
from collections import defaultdict
from pathlib import Path

import torch

OVERRIDES_FILE = "f32-overrides.safetensors"

# Checkpoint-name suffixes of the tensors the conversion downcast.
SUFFIXES = (
    "hc_attn_base",
    "hc_attn_scale",
    "hc_ffn_base",
    "hc_ffn_scale",
    "mlp.gate.e_score_correction_bias",
    "self_attn.A_log",
    "self_attn.dt_bias",
)

_DTYPES = {
    "F32": torch.float32,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}


def _headers(model_dir: str) -> dict[str, tuple[dict, str, int]]:
    """name -> (header entry, shard path, data section offset)."""
    out: dict[str, tuple[dict, str, int]] = {}
    index = Path(model_dir) / "model.safetensors.index.json"
    if index.is_file():
        names = set(json.loads(index.read_text())["weight_map"].values())
        # The MTP shard ships outside the target's index.
        if (Path(model_dir) / "model_mtp.safetensors").is_file():
            names.add("model_mtp.safetensors")
        shards = [str(Path(model_dir) / name) for name in sorted(names)]
    else:
        shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
        shards = [
            path
            for path in shards
            if Path(path).name not in {OVERRIDES_FILE, "fp8-swapset.safetensors"}
        ]
    for shard in shards:
        with open(shard, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        for name, entry in header.items():
            if name != "__metadata__":
                out[name] = (entry, shard, 8 + n)
    return out


def _read(entry: dict, shard: str, base: int) -> torch.Tensor:
    start, end = entry["data_offsets"]
    with open(shard, "rb") as fh:
        fh.seek(base + start)
        buf = bytearray(fh.read(end - start))
    return torch.frombuffer(buf, dtype=_DTYPES[entry["dtype"]]).reshape(entry["shape"])


def select(native: dict, converted: dict, skip_layer: int | None) -> list[str]:
    """Names that are F32 in the native checkpoint but not in the conversion."""
    names = []
    for name, (entry, _, _) in native.items():
        if not name.endswith(SUFFIXES) or entry["dtype"] != "F32":
            continue
        if skip_layer is not None and f".layers.{skip_layer}." in name:
            continue
        other = converted.get(name)
        if other is None or other[0]["dtype"] == "F32":
            continue
        names.append(name)
    return sorted(names)


def build(
    native_dir: str, model_dir: str, out: str | None = None, skip_layer: int | None = 45
) -> Path:
    native = _headers(native_dir)
    converted = _headers(model_dir)
    names = select(native, converted, skip_layer)
    if not names:
        raise SystemExit(
            "nothing to override: the conversion already keeps these tensors in F32"
        )
    tensors: dict[str, torch.Tensor] = {}
    err: dict[str, list[float]] = defaultdict(list)
    for name in names:
        t = _read(*native[name]).contiguous()
        tensors[name] = t
        c = _read(*converted[name]).float()
        key = name.split(".", 3)[-1].split(".", 1)[-1] if ".layers." in name else name
        err[key].append(((t - c).abs() / (t.abs() + 1e-9)).max().item())
    from safetensors.torch import save_file

    out_path = Path(out or os.path.join(model_dir, OVERRIDES_FILE))
    save_file(
        tensors,
        str(out_path),
        metadata={
            "source": os.path.abspath(native_dir),
            "purpose": "F32 tensors the NVFP4 conversion downcast to BF16",
        },
    )
    print(
        f"{len(tensors)} tensors, "
        f"{sum(t.numel() * 4 for t in tensors.values())} bytes -> {out_path}"
    )
    for key, rel in sorted(err.items()):
        print(
            f"  {key:36s} n={len(rel):3d} "
            f"max rel error of the conversion's copy {max(rel):.3%}"
        )
    return out_path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--native",
        required=True,
        help="directory of the native (F32-carrying) checkpoint",
    )
    ap.add_argument(
        "--model", required=True, help="directory of the converted checkpoint to serve"
    )
    ap.add_argument("--out", help=f"output file (default <model>/{OVERRIDES_FILE})")
    ap.add_argument(
        "--keep-mtp-layer",
        action="store_true",
        help="also override layer 45 (the MTP layer)",
    )
    args = ap.parse_args(argv)
    build(
        args.native,
        args.model,
        args.out,
        skip_layer=None if args.keep_mtp_layer else 45,
    )


if __name__ == "__main__":
    main()
