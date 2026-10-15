"""N10: the fast tier, one lever at a time, against the exact pipeline.

Usage: PYTHONPATH=<worktree> python n10_fast_ab.py MODE OUTDIR NAME:JSON [NAME:JSON ...]
  MODE   distilled | dev | dfr
  JSON   pipeline.Fast fields plus optional "decoder" ("diffusion" | "conv"),
         e.g. 'cache10:{"step_cache": 0.10}'  'exact:{}'
Each variant renders OUTDIR/NAME.mp4 (1536x1024x121, seed 7, the beat1d prompt),
saves its uint8 frames (NAME.npy), a 5-frame contact sheet (NAME_frames.png) and
appends a row to OUTDIR/results.tsv: wall, spans, step-cache stats and the PSNR
against OUTDIR/exact.npy when it exists. One engine for all variants (the text
is encoded once). Run through gpu_run.py --need-gb 80."""

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from slimserve.video.ltx25 import mux, pipeline
from slimserve.video.ltx25.dit import StepCache
from paths import SCRATCH

MODE, OUT = sys.argv[1], Path(sys.argv[2]).expanduser()
OUT.mkdir(parents=True, exist_ok=True)
PROMPT = (
    Path(SCRATCH, "demo/v4/prompt_beat1d.txt")
    .read_text()
    .strip()
)
W, H, FRAMES, SEED = 1536, 1024, 121, 7

captured = {}
_write = mux.write_mp4


def write_mp4(path, frames, fps, waveform, sample_rate):
    captured["frames"] = np.asarray(frames)
    return _write(path, frames, fps, waveform, sample_rate)


mux.write_mp4 = write_mp4
caches: list[StepCache] = []
_cache = pipeline.Fast.cache


def cache(self):
    c = _cache(self)
    if c is not None:
        caches.append(c)
    return c


pipeline.Fast.cache = cache


def psnr(a: np.ndarray, b: np.ndarray) -> tuple[float, list[float]]:
    per = []
    for x, y in zip(a, b):
        mse = float(np.mean((x.astype(np.float32) - y.astype(np.float32)) ** 2))
        per.append(10 * np.log10(255.0**2 / max(mse, 1e-6)))
    return float(np.mean(per)), per


def sheet(frames: np.ndarray, path: Path) -> None:
    from PIL import Image

    idx = [0, len(frames) // 4, len(frames) // 2, 3 * len(frames) // 4, len(frames) - 1]
    tiles = [Image.fromarray(frames[i]).resize((W // 4, H // 4)) for i in idx]
    out = Image.new("RGB", (W // 4 * len(tiles), H // 4))
    for n, t in enumerate(tiles):
        out.paste(t, (n * W // 4, 0))
    out.save(path)


engine = pipeline.LTX25Engine(variant="dev" if MODE == "dev" else "distilled")
text_embeds = None
if MODE != "dev":
    text_embeds = engine.load_text().encode(PROMPT)[:2]
    mx.eval(*text_embeds)
ref = OUT / "exact.npy"
for spec in sys.argv[3:]:
    name, _, js = spec.partition(":")
    cfg = json.loads(js or "{}")
    decoder = cfg.pop("decoder", "diffusion")
    fast = pipeline.Fast.from_config(cfg)
    caches.clear()
    kw = dict(height=H, width=W, num_frames=FRAMES, seed=SEED, fast=fast)
    if text_embeds is not None:
        kw["text_embeds"] = text_embeds
    t0 = time.perf_counter()
    result = getattr(engine, MODE)(PROMPT, **kw)
    engine.render(result, OUT / f"{name}.mp4", seed=SEED, decoder=decoder)
    wall = time.perf_counter() - t0
    frames = captured["frames"]
    np.save(OUT / f"{name}.npy", frames)
    sheet(frames, OUT / f"{name}_frames.png")
    spans = " ".join(f"{k}={v:.1f}" for k, v in result.timings.spans.items())
    stats = ";".join(
        f"{c.computed}c/{c.skipped}s["
        + ",".join(f"{r:.3f}{'*' if s else ''}" for r, s in c.history)
        + "]"
        for c in caches
    )
    line = f"{MODE}\t{name}\t{json.dumps(cfg)}\t{decoder}\t{wall:.1f}\t{spans}\t{stats}"
    if ref.exists() and name != "exact":
        mean, per = psnr(np.load(ref, mmap_mode="r"), frames)
        line += f"\tpsnr={mean:.2f}\tmin={min(per):.2f}"
    print(line, flush=True)
    with open(OUT / "results.tsv", "a") as fh:
        fh.write(line + "\n")
    del result, frames
    captured.clear()
    mx.clear_cache()
