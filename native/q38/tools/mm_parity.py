#!/usr/bin/env python3
"""Image and video prompts: the q38 engine against the Hugging Face reference.

  mm_parity.py ref    --model ~/models/Qwen3.8-27B --image a.png --out ref.npz [--device mps]
  mm_parity.py engine --model ~/models/Qwen3.8-27B --ref ref.npz --pack bf16.q38w [--build DIR]

`ref` runs the reference model (transformers) on two prompts, one with the
image and one with a synthetic 8-frame video, and saves the prompt ids, the
processor's pixel inputs, the projected vision rows and the logits of the last
positions. `engine` feeds the same ids through slimserve/native/vision.py and
the engine (q38_step_mm) and reports, per prompt: cosine of the vision rows,
and KL(reference || engine), top-1 agreement and the largest logit difference
over the compared positions. Run the two steps one after the other: each loads
the whole model.
"""
import argparse
import ctypes as C
import os
import sys

import numpy as np

TAIL = 48  # positions compared, counted from the end of the prompt


def synthetic_video(frames=8, size=224):
    """A white square crossing a dark frame left to right, on a colour ramp."""
    out = np.zeros((frames, size, size, 3), dtype=np.uint8)
    ramp = np.linspace(0, 120, size, dtype=np.uint8)
    for f in range(frames):
        out[f, :, :, 2] = ramp[None, :]
        out[f, :, :, 1] = ramp[:, None] // 2
        x = 10 + f * (size - 70) // (frames - 1)
        out[f, 80:130, x:x + 50] = 255
    return out


def cmd_ref(a):
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    proc = AutoProcessor.from_pretrained(a.model)
    dev = torch.device(a.device)
    dtype = torch.bfloat16 if a.device == "cpu" else torch.float16
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=dtype).to(dev).eval()
    image = Image.open(a.image).convert("RGB")
    video = synthetic_video()
    cases = {
        "image": [{"role": "user", "content": [{"type": "image", "image": image},
                                               {"type": "text", "text": "Describe this image in two sentences."}]}],
        "video": [{"role": "user", "content": [{"type": "video", "video": list(video)},
                                               {"type": "text", "text": "What moves in this video, and in which direction?"}]}],
    }
    out = {}
    for name, messages in cases.items():
        kw = {"fps": 2.0} if name == "video" else {}
        inp = proc.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                                       return_tensors="pt", **kw)
        video_case = "pixel_values_videos" in inp
        pix = inp["pixel_values_videos" if video_case else "pixel_values"]
        grid = inp["video_grid_thw" if video_case else "image_grid_thw"]
        with torch.inference_mode():
            feats = model.model.get_image_features(pix.to(dev, dtype), grid.to(dev), return_dict=True).pooler_output
            feats = torch.cat(list(feats), 0) if isinstance(feats, (list, tuple)) else feats
            res = model(**{k: (v.to(dev, dtype) if v.is_floating_point() else v.to(dev)) for k, v in inp.items()})
        logits = res.logits[0, -TAIL:].float().cpu().numpy()
        ids = inp["input_ids"][0].cpu().numpy().astype(np.int32)
        out.update({f"{name}_ids": ids, f"{name}_pix": pix.float().cpu().numpy(), f"{name}_grid": grid.cpu().numpy(),
                    f"{name}_feats": feats.float().cpu().numpy(), f"{name}_logits": logits,
                    f"{name}_types": inp["mm_token_type_ids"][0].cpu().numpy().astype(np.int8)})
        print(f"{name}: {len(ids)} tokens, grid {grid.tolist()}, vision rows {tuple(feats.shape)}, "
              f"reference next token {int(logits[-1].argmax())}", flush=True)
    np.savez(os.path.expanduser(a.out), **out)


def cmd_engine(a):
    import torch

    repo = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    sys.path.insert(0, repo)
    from slimserve.native import q38, vision

    build = os.path.expanduser(a.build or os.path.join(repo, "native", "q38", "build"))
    eng = q38.Engine(os.path.join(build, "libq38.dylib"), os.path.expanduser(a.pack),
                     ":".join(os.path.join(build, n) for n in ("q38_kernels.metallib", "q38_native.metallib")))
    lib, h = eng._lib, eng._h
    lib.q38_step_mm.restype, lib.q38_step_mm.argtypes = C.c_int, [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p,
                                                                C.c_int, C.c_void_p]
    lib.q38_logits_bf16.restype, lib.q38_logits_bf16.argtypes = C.c_void_p, [C.c_void_p]
    lib.q38_reset.restype, lib.q38_reset.argtypes = None, [C.c_void_p]
    V, H = eng.vocab, eng.hidden
    tower = vision.VisionTower(os.path.expanduser(a.model))
    ref = np.load(os.path.expanduser(a.ref))
    worst = 0.0
    for name in ("image", "video"):
        ids, grid, types = ref[f"{name}_ids"], ref[f"{name}_grid"].reshape(-1), ref[f"{name}_types"]
        rows = tower.encode(name, torch.from_numpy(ref[f"{name}_pix"]), torch.from_numpy(grid))
        mine = torch.from_numpy(rows.view(np.int16).copy()).view(torch.bfloat16).float().numpy()
        want = ref[f"{name}_feats"]
        cos = (mine * want).sum(1) / (np.linalg.norm(mine, axis=1) * np.linalg.norm(want, axis=1) + 1e-9)
        t, hh, ww = int(grid[0]), int(grid[1]) // tower.merge, int(grid[2]) // tower.merge
        emb = np.flatnonzero(types != 0)
        offset, length = int(emb[0]), int(emb[-1] - emb[0] + 1)
        mask = types[offset:offset + length] != 0
        g = (hh, ww) if name == "image" else (t, hh, ww)
        tokens, embeds, pos3 = vision.plan(ids.tolist(), [(offset, length, mask, name, rows, g)])
        lib.q38_reset(h)
        n = len(tokens)
        tail = np.zeros((min(n, TAIL), V), dtype=np.float32)
        erow = np.cumsum(tokens < 0) - 1
        for s in range(0, n, 8):
            m = min(8, n - s)
            tk = np.ascontiguousarray(tokens[s:s + m])
            eb = np.zeros((m, H), dtype=np.uint16)
            for i in range(m):
                if tk[i] < 0:
                    eb[i] = embeds[erow[s + i]]
            ps = np.ascontiguousarray(pos3[s:s + m])
            rc = lib.q38_step_mm(h, tk.ctypes.data, eb.ctypes.data, ps.ctypes.data, m, None)
            if rc:
                raise RuntimeError(f"q38_step_mm failed: {rc}")
            raw = np.ctypeslib.as_array(C.cast(lib.q38_logits_bf16(h), C.POINTER(C.c_uint16)), shape=(m, V))
            lg = torch.from_numpy(raw.view(np.int16).copy()).view(torch.bfloat16).float().numpy()
            for i in range(m):
                k = s + i - (n - len(tail))
                if k >= 0:
                    tail[k] = lg[i]
        want_l = ref[f"{name}_logits"][-len(tail):]
        lp = want_l - np.logaddexp.reduce(want_l, axis=1, keepdims=True)
        lq = tail - np.logaddexp.reduce(tail, axis=1, keepdims=True)
        kl = (np.exp(lp) * (lp - lq)).sum(1)
        top1 = (want_l.argmax(1) == tail.argmax(1)).mean()
        ty = types[-len(tail):]
        text = ty == 0
        worst = max(worst, float(kl[text].mean()))
        print(f"{name}: per position (v = vision row): " + " ".join(f"{'v' if ty[i] else 't'}{kl[i]:.3f}" for i in range(len(kl))))
        print(f"{name}: text positions only ({int(text.sum())}): KL mean {kl[text].mean():.5f} max {kl[text].max():.5f}, "
              f"top-1 {(want_l.argmax(1) == tail.argmax(1))[text].mean():.3f}")
        print(f"{name}: {n} tokens ({int((tokens < 0).sum())} vision rows, grid {g}); vision rows cosine min "
              f"{cos.min():.5f} mean {cos.mean():.5f}; last {len(tail)} positions: KL mean {kl.mean():.5f} max "
              f"{kl.max():.5f}, top-1 {top1:.3f}, max |logit diff| {np.abs(want_l - tail).max():.3f}; next token "
              f"reference {int(want_l[-1].argmax())} engine {int(tail[-1].argmax())}", flush=True)
    eng.close()
    return 0 if worst < a.max_kl else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("ref")
    r.add_argument("--model", required=True)
    r.add_argument("--image", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--device", default="mps")
    e = sub.add_parser("engine")
    e.add_argument("--model", required=True)
    e.add_argument("--ref", required=True)
    e.add_argument("--pack", required=True)
    e.add_argument("--build")
    e.add_argument("--max-kl", type=float, default=0.02)
    a = ap.parse_args()
    sys.exit(cmd_ref(a) if a.cmd == "ref" else cmd_engine(a))


if __name__ == "__main__":
    main()
