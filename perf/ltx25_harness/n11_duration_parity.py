"""N11: the duration head against upstream's DurationHead (torch, CPU).

  ref  OUT.npz          upstream ltx_core DurationHead on fixed random connector
                        tokens (1, 1024, 4096) + (1, 1024, 2048): seconds for
                        video-only, audio-only and both
  ours OUT.npz REF.npz  our DurationHead on the same tokens; prints the diffs
  prompt "..."          (GPU, through gpu_run.py --need-gb 40) encode a prompt
                        with the real text encoder and print the predicted
                        seconds / frames at 24 fps

Run `ref` in the torch env (conda vllm-mlx), `ours` in venv-slimserve."""

import sys

import numpy as np
from paths import MODELS, UP

HEAD = f"{MODELS}/model_patches/ltx-2.5-duration-head-bf16.safetensors"
CASES = ("video", "audio", "both")


def inputs():
    rng = np.random.default_rng(5)
    v = (rng.standard_normal((1, 1024, 4096)) * 0.5).astype(np.float32)
    a = (rng.standard_normal((1, 1024, 2048)) * 0.5).astype(np.float32)
    return v, a


def ref(out):
    import torch

    sys.path[:0] = [f"{UP}/ltx-core/src"]
    from ltx_core.duration_head.model_configurator import (
        DURATION_HEAD_KEY_OPS,
        DurationHeadConfigurator,
    )
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder

    head = (
        SingleGPUModelBuilder(
            model_class_configurator=DurationHeadConfigurator,
            model_path=HEAD,
            model_sd_ops=DURATION_HEAD_KEY_OPS,
        )
        .build(device=torch.device("cpu"), dtype=torch.float32)
        .eval()
    )
    v, a = (torch.from_numpy(x) for x in inputs())
    res = {}
    with torch.inference_mode():
        res["video"] = float(head(v, None).item())
        res["audio"] = float(head(None, a).item())
        res["both"] = float(head(v, a).item())
    print(res)
    np.savez(out, **{k: np.array(x) for k, x in res.items()})


def ours(out, ref_path):
    import mlx.core as mx

    from slimserve.video.ltx25.duration import DurationHead

    head = DurationHead().load()
    v, a = (mx.array(x) for x in inputs())
    res = {
        "video": head.seconds(v, None),
        "audio": head.seconds(None, a),
        "both": head.seconds(v, a),
    }
    r = np.load(ref_path)
    for k in CASES:
        rel = abs(res[k] - float(r[k])) / float(r[k])
        print(f"{k:5s} ref {float(r[k]):.4f} s  ours {res[k]:.4f} s  rel {rel:.2e}")
    np.savez(out, **{k: np.array(x) for k, x in res.items()})


def prompt(text):
    import mlx.core as mx

    from slimserve.video.ltx25.duration import DurationHead
    from slimserve.video.ltx25.text import TextEncoder

    enc = TextEncoder()
    enc.load()
    v, a = enc.encode(text)
    mx.eval(v, a)
    frames, seconds = DurationHead().num_frames(v, a, 24.0)
    print(f"{seconds:.2f} s -> {frames} frames at 24 fps: {text[:80]}")


if __name__ == "__main__":
    {"ref": ref, "ours": ours, "prompt": prompt}[sys.argv[1]](*sys.argv[2:])
