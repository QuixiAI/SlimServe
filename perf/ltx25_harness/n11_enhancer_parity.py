# ruff: noqa: E501
"""N11: the prompt enhancer against transformers' Gemma-4 on the CPU.

ref  OUT.npz "prompt"       (conda vllm-mlx, through gpu_run.py --need-gb 30: the
                            fp32 model is 20 GiB of RAM) upstream's exact recipe:
                            Gemma4ForConditionalGeneration, apply_chat_template with
                            the T2V system prompt, generate(do_sample=False,
                            no_repeat_ngram_size=5, max_new_tokens=600). Saves the
                            prompt ids, the generated ids, the text, and the first
                            step's logits.
ours OUT.npz REF.npz        (venv-slimserve, --need-gb 24) our PromptEnhancer on the
                            same prompt: compares prompt ids, first-step logits,
                            and the generated ids token for token.
ref-i2v OUT.npz IMAGE "prompt"   upstream enhance_i2v: the still decoded and scaled
                            to a 896 long side (bilinear, torch uint8), the Gemma-4
                            processor with the PIL image processor (as ltx_core
                            gemma_assets picks it), the I2V system prompt. Saves
                            the processor's pixel values and positions, the
                            vision tower's soft tokens and the same fields as ref.
ours-i2v OUT.npz REF.npz    our image path against it stage by stage: the 896
                            still, the patches, the soft tokens, the prompt ids,
                            the first logits, the generation.
teacher-ref / teacher-ours  as before; work for both kinds of reference.
"""

import math
import pathlib
import sys
import time

import numpy as np
from paths import MODELS, REPO, SCRATCH

ROOT = f"{MODELS}/prompt_enhancer/gemma-4-E2B-it"
# processor_config.json of google/gemma-4-E2B-it at the pinned revision (not
# part of the registry manifest: our port reads what it needs from config.json)
PROCESSOR_CONFIG = f"{SCRATCH}/gemma_aux/processor_config.json"


def _processor(tk):
    """Gemma4Processor as ltx_core.gemma_assets builds it: the PIL image
    processor (the torchvision one 'shifts I2V enhance goldens')."""
    import json

    from transformers import (
        Gemma4AudioFeatureExtractor,
        Gemma4ImageProcessorPil,
        Gemma4Processor,
        Gemma4VideoProcessor,
    )

    with open(PROCESSOR_CONFIG) as fh:
        cfg = json.load(fh)
    sub = lambda d: {k: v for k, v in d.items() if not k.endswith("_type")}  # noqa: E731
    return Gemma4Processor(
        feature_extractor=Gemma4AudioFeatureExtractor(**sub(cfg["feature_extractor"])),
        image_processor=Gemma4ImageProcessorPil(**sub(cfg["image_processor"])),
        tokenizer=tk,
        video_processor=Gemma4VideoProcessor(**sub(cfg["video_processor"])),
        chat_template=pathlib.Path(f"{ROOT}/chat_template.jinja").read_text(),
        image_seq_length=cfg["image_seq_length"],
        audio_seq_length=cfg["audio_seq_length"],
        audio_ms_per_token=cfg["audio_ms_per_token"],
    )


def ref_i2v(out, image_path, prompt):
    import torch
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    sys.path.append(REPO)
    from slimserve.video.ltx25 import image as image_mod
    from slimserve.video.ltx25.enhancer import system_prompt

    # ltx_pipelines helpers.generate_enhanced_prompt: decode_image, tensor,
    # resize_aspect_ratio_preserving(896) -> uint8
    img = torch.tensor(image_mod.decode_image(image_path))
    h, w = img.shape[0], img.shape[1]
    scale = 896 / float(max(h, w))
    th, tw = int(h * scale), int(w * scale)
    x = img.permute(2, 0, 1)[None]
    s = max(th / h, tw / w)
    nh, nw = math.ceil(h * s), math.ceil(w * s)
    x = torch.nn.functional.interpolate(
        x, size=(nh, nw), mode="bilinear", align_corners=False
    )
    top, left = (nh - th) // 2, (nw - tw) // 2
    still = x[0, :, top : top + th, left : left + tw].permute(1, 2, 0).to(torch.uint8)
    print("still", tuple(still.shape), still.dtype)

    tk = AutoTokenizer.from_pretrained(ROOT)
    proc = _processor(tk)
    msgs = [
        {"role": "system", "content": system_prompt("i2v")},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"User Raw Input Prompt: {prompt}."},
            ],
        },
    ]
    text = proc.tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    inputs = proc(text=text, images=still, return_tensors="pt")
    n_soft = (
        int(inputs["num_soft_tokens_per_image"][0])
        if "num_soft_tokens_per_image" in inputs
        else None
    )
    model_inputs = {
        k: v
        for k, v in inputs.items()
        if k
        in (
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_position_ids",
            "mm_token_type_ids",
        )
    }
    print(
        "input ids",
        model_inputs["input_ids"].shape,
        "pixel_values",
        model_inputs["pixel_values"].shape,
        "soft tokens",
        n_soft,
    )
    model = Gemma4ForConditionalGeneration.from_pretrained(
        ROOT, dtype=torch.float32
    ).eval()
    with torch.inference_mode():
        feats = (
            model.model.get_image_features(
                model_inputs["pixel_values"],
                model_inputs["image_position_ids"],
                return_dict=True,
            )
            .pooler_output.float()
            .numpy()
        )
        first = model(**model_inputs, logits_to_keep=1).logits[0, -1].float().numpy()
        t0 = time.time()
        gen = model.generate(
            **model_inputs, do_sample=False, no_repeat_ngram_size=5, max_new_tokens=600
        )
    new = gen[0][model_inputs["input_ids"].shape[1] :].numpy()
    print(f"generated {len(new)} tokens in {time.time() - t0:.0f}s")
    print(tk.decode(new, skip_special_tokens=True))
    np.savez(
        out,
        image_path=np.array(image_path),
        still=still.numpy(),
        pixel_values=model_inputs["pixel_values"][0].numpy(),
        image_position_ids=model_inputs["image_position_ids"][0].numpy(),
        features=feats,
        prompt_ids=model_inputs["input_ids"][0].numpy(),
        first_logits=first,
        new_ids=new,
        text=np.array(tk.decode(new, skip_special_tokens=True)),
    )


def ours_i2v(out, ref_path):
    import mlx.core as mx

    sys.path.append(REPO)
    from slimserve.video.ltx25 import image as image_mod
    from slimserve.video.ltx25.enhancer import (
        BOI_TOKEN,
        EOI_TOKEN,
        IMAGE_TOKEN,
        PromptEnhancer,
        chat_text,
        fit_long_side,
        image_patches,
        system_prompt,
    )

    r = np.load(ref_path)
    e = PromptEnhancer().load()
    decoded = image_mod.decode_image(str(r["image_path"]))
    still = fit_long_side(decoded)
    ref_still = r["still"]
    print("still shape", still.shape, "ref", ref_still.shape)
    if still.shape == ref_still.shape:
        d = np.abs(still.astype(int) - ref_still.astype(int))
        print(f"still pixels differing: {(d > 0).mean() * 100:.3f}% (max {d.max()})")
    vc = e.vcfg
    patches, ph, pw = image_patches(
        still,
        vc["patch_size"],
        e.top["vision_soft_tokens_per_image"],
        vc["pooling_kernel_size"],
    )
    pos = r["image_position_ids"]
    valid = ~(pos == -1).all(axis=1)
    ref_patches = r["pixel_values"][valid].astype(np.float64)
    print(
        f"patches {patches.shape} ref valid {ref_patches.shape}; grid {ph}x{pw} ref max x,y"
        f" {pos[valid].max(axis=0).tolist()}; max-abs diff"
        f" {np.abs(patches.astype(np.float64) - ref_patches).max():.2e}"
    )
    feats = e.image_features(decoded)
    mx.eval(feats)
    ours_f = np.array(feats).astype(np.float64)
    ref_f = r["features"].astype(np.float64)
    rel = np.linalg.norm(ours_f - ref_f) / np.linalg.norm(ref_f)
    per = np.linalg.norm(ours_f - ref_f, axis=1) / np.linalg.norm(ref_f, axis=1)
    print(
        f"soft tokens {ours_f.shape} rel-L2 {rel:.2e} per-token median {np.median(per):.2e} max {per.max():.2e}"
    )

    prompt_ids = r["prompt_ids"].tolist()
    text = e.tokenizer.decode(prompt_ids[1:], skip_special_tokens=False)
    user = text.split("<|turn>user\n", 1)[1].split("<turn|>", 1)[0]
    raw = user.split(EOI_TOKEN, 1)[1]
    placeholders = BOI_TOKEN + IMAGE_TOKEN * feats.shape[0] + EOI_TOKEN
    ids = e.tokenize(chat_text(system_prompt("i2v"), placeholders + raw))
    print("prompt ids equal:", ids == prompt_ids, len(ids), len(prompt_ids))
    img, pad = e.top["image_token_id"], e.cfg["pad_token_id"]
    slots = [i for i, t in enumerate(ids) if t == img]
    cache = [None] * len(e.cfg["layer_types"])
    logits = e._logits(
        e._forward([pad if t == img else t for t in ids], 0, cache, (slots, feats))
    )
    mx.eval(logits)
    ours_first = np.array(logits).astype(np.float64)
    ref_first = r["first_logits"].astype(np.float64)
    rel = np.linalg.norm(ours_first - ref_first) / np.linalg.norm(ref_first)
    print(
        f"first logits rel-L2 {rel:.2e} max-abs {np.abs(ours_first - ref_first).max():.2e}"
        f" argmax {int(ours_first.argmax())} vs {int(ref_first.argmax())}"
    )
    t0 = time.time()
    new = e.generate(ids, features=feats)
    print(f"generated {len(new)} tokens in {time.time() - t0:.1f}s")
    ref_new = r["new_ids"].tolist()
    if ref_new and ref_new[-1] in e.gen["eos_token_id"]:
        ref_new = ref_new[:-1]
    common = 0
    for a, b in zip(new, ref_new):
        if a != b:
            break
        common += 1
    print(
        f"ref {len(ref_new)} tokens; identical prefix {common}; equal: {new == ref_new}"
    )
    print("ours:", e.tokenizer.decode(new, skip_special_tokens=True))
    if new != ref_new:
        print("ref :", str(r["text"]))
    np.savez(
        out,
        new_ids=np.array(new),
        prompt_ids=np.array(ids),
        first_logits=ours_first,
        features=ours_f,
    )


def ref(out, prompt):
    import torch
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    sys.path.append(REPO)
    from slimserve.video.ltx25.enhancer import system_prompt

    tk = AutoTokenizer.from_pretrained(ROOT)
    msgs = [
        {"role": "system", "content": system_prompt("t2v")},
        {"role": "user", "content": f"user prompt: {prompt}"},
    ]
    text = tk.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tk(text, return_tensors="pt")
    model = Gemma4ForConditionalGeneration.from_pretrained(
        ROOT, dtype=torch.float32
    ).eval()
    with torch.inference_mode():
        first = model(**ids, logits_to_keep=1).logits[0, -1].float().numpy()
        t0 = time.time()
        gen = model.generate(
            **ids, do_sample=False, no_repeat_ngram_size=5, max_new_tokens=600
        )
    new = gen[0][ids["input_ids"].shape[1] :].numpy()
    print(f"generated {len(new)} tokens in {time.time() - t0:.0f}s")
    print(tk.decode(new, skip_special_tokens=True))
    np.savez(
        out,
        prompt_ids=ids["input_ids"][0].numpy(),
        first_logits=first,
        new_ids=new,
        text=np.array(tk.decode(new, skip_special_tokens=True)),
    )


def ours(out, ref_path):
    import mlx.core as mx

    sys.path.append(REPO)
    from slimserve.video.ltx25.enhancer import PromptEnhancer, chat_text, system_prompt

    r = np.load(ref_path)
    prompt_ids = r["prompt_ids"].tolist()
    e = PromptEnhancer().load()
    text = e.tokenizer.decode(prompt_ids[1:], skip_special_tokens=False)
    user = text.split("<|turn>user\n", 1)[1].split("<turn|>", 1)[0]
    ids = e.tokenize(chat_text(system_prompt("t2v"), user))
    print("prompt ids equal:", ids == prompt_ids, len(ids), len(prompt_ids))
    cache = [None] * len(e.cfg["layer_types"])
    logits = e._logits(e._forward(ids, 0, cache))
    mx.eval(logits)
    ours_first = np.array(logits).astype(np.float64)
    ref_first = r["first_logits"].astype(np.float64)
    rel = np.linalg.norm(ours_first - ref_first) / np.linalg.norm(ref_first)
    print(
        f"first logits rel-L2 {rel:.2e} max-abs {np.abs(ours_first - ref_first).max():.2e}"
        f" argmax {int(ours_first.argmax())} vs {int(ref_first.argmax())}"
    )
    t0 = time.time()
    new = e.generate(ids)
    print(f"generated {len(new)} tokens in {time.time() - t0:.1f}s")
    ref_new = r["new_ids"].tolist()
    eos = e.gen["eos_token_id"]
    if ref_new and ref_new[-1] in eos:  # generate() keeps the stop token; ours does not
        ref_new = ref_new[:-1]
    common = 0
    for a, b in zip(new, ref_new):
        if a != b:
            break
        common += 1
    print(
        f"ref {len(ref_new)} tokens; identical prefix {common}; equal: {new == ref_new}"
    )
    if new != ref_new:
        print("ours:", e.tokenizer.decode(new, skip_special_tokens=True))
        print("ref :", str(r["text"]))
    np.savez(
        out, new_ids=np.array(new), prompt_ids=np.array(ids), first_logits=ours_first
    )


def teacher_ref(out, ref_path):
    """Logits at every generated position of the reference sequence (one fp32
    forward over prompt + generation)."""
    import torch
    from transformers import Gemma4ForConditionalGeneration

    r = np.load(ref_path)
    seq = np.concatenate([r["prompt_ids"], r["new_ids"]])
    n_prompt = len(r["prompt_ids"])
    extra = {}
    if "pixel_values" in r:  # an I2V reference: the still goes along
        extra = {
            "pixel_values": torch.tensor(r["pixel_values"])[None],
            "image_position_ids": torch.tensor(r["image_position_ids"])[None],
        }
    model = Gemma4ForConditionalGeneration.from_pretrained(
        ROOT, dtype=torch.float32
    ).eval()
    with torch.inference_mode():
        logits = (
            model(input_ids=torch.tensor(seq)[None], **extra).logits[0].float().numpy()
        )
    # logits[i] predicts token i + 1: positions n_prompt - 1 .. end - 1 predict the generation
    extra = {"image_path": r["image_path"]} if "image_path" in r else {}
    np.savez(out, logits=logits[n_prompt - 1 : -1], seq=seq, n_prompt=n_prompt, **extra)


def teacher_ours(ref_logits_path):
    """Our logits at the same positions (one prefill over the reference
    sequence); per-position error, and the reference's top-2 margin wherever
    our argmax differs."""
    import mlx.core as mx

    sys.path.append(REPO)
    from slimserve.video.ltx25.enhancer import PromptEnhancer

    t = np.load(ref_logits_path)
    seq, n_prompt, ref = (
        t["seq"].tolist(),
        int(t["n_prompt"]),
        t["logits"].astype(np.float64),
    )
    e = PromptEnhancer().load()
    cache = [None] * len(e.cfg["layer_types"])
    image = None
    if "image_path" in t:
        from slimserve.video.ltx25 import image as image_mod

        feats = e.image_features(image_mod.decode_image(str(t["image_path"])))
        img, pad = e.top["image_token_id"], e.cfg["pad_token_id"]
        image = ([i for i, tok in enumerate(seq) if tok == img], feats)
        seq = [pad if tok == img else tok for tok in seq]
    h = e._forward(seq, 0, cache, image)
    x = h[0, n_prompt - 1 : -1].astype(e.O)
    logits = (x @ e.w["embed_tokens.weight"].astype(e.O).T).astype(mx.float32)
    cap = e.cfg["final_logit_softcapping"]
    logits = cap * mx.tanh(logits / cap)
    mx.eval(logits)
    ours = np.array(logits).astype(np.float64)
    rel = np.linalg.norm(ours - ref, axis=1) / np.linalg.norm(ref, axis=1)
    print(
        f"{len(rel)} positions: rel-L2 median {np.median(rel):.2e} max {rel.max():.2e}"
    )
    flips = np.nonzero(ours.argmax(1) != ref.argmax(1))[0]
    print(f"argmax differs at {len(flips)} positions")
    for i in flips[:10]:
        top2 = np.sort(ref[i])[-2:]
        print(
            f"  gen step {i}: ref top-2 margin {top2[1] - top2[0]:.3f} logits;"
            f" ours picks {int(ours[i].argmax())} (ref {int(ref[i].argmax())})"
        )


if __name__ == "__main__":
    {
        "ref": ref,
        "ours": ours,
        "ref-i2v": ref_i2v,
        "ours-i2v": ours_i2v,
        "teacher-ref": teacher_ref,
        "teacher-ours": teacher_ours,
    }[sys.argv[1]](*sys.argv[2:])
