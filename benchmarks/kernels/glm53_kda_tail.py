# SPDX-License-Identifier: Apache-2.0
"""Fixtures and explicit launches for the remaining KDA arithmetic diagnostics."""

from itertools import product

from benchmarks.analyze_glm53_quality_pair import require
from benchmarks.kernels import check_glm53_kda_gate as gate
from benchmarks.kernels import check_glm53_kda_recompute as recompute

NAMES = {
    "state": "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
    "output": "chunk_gla_fwd_kernel_o",
}
CONFIGS = {
    "state": ({"num_warps": 4, "num_stages": 2}, {"num_warps": 4, "num_stages": 3}),
    "output": ({"num_warps": 8, "num_stages": 2}, {"num_warps": 4, "num_stages": 2}),
}
OUTPUT_NAMES = {
    "state": ("snapshots", "new_values", "final_state"),
    "output": ("output",),
}


def matrix():
    cases = []
    for rank, lengths in product(range(4), gate.LAYOUTS):
        base = dict(rank=rank, lengths=list(lengths))
        cases.extend(
            dict(**base, regime=regime, seed=530901, magnitude=1.0)
            for regime in ("identity", "basis")
        )
        cases.extend(
            dict(**base, regime="conditioned", seed=seed, magnitude=mag)
            for seed, mag in product(gate.SEEDS, gate.MAGNITUDES)
        )
    return cases


def jit_for(stage):
    from triton.runtime.jit import JITFunction

    if stage == "state":
        from vllm.third_party.flash_linear_attention.ops.chunk_delta_h import (
            chunk_gated_delta_rule_fwd_kernel_h_blockdim64 as wrapped,
        )
    else:
        from vllm.models.kimi_k3.amd.ops.third_party.kda.chunk import (
            chunk_gla_fwd_kernel_o as wrapped,
        )
    jit = wrapped.fn.fn
    require(isinstance(jit, JITFunction) and jit.__name__ == NAMES[stage], "wrong JIT")
    return jit


def make_inputs(stage, case, device):
    import torch

    regime = case["regime"]
    base_case = {
        **case,
        "regime": "conditioned" if regime == "conditioned" else "identity",
    }
    raw = recompute.make_inputs(base_case, device=device)
    starts, chunks = gate.sequence_metadata(case["lengths"])
    tokens, count, sequences = sum(case["lengths"]), len(chunks), len(case["lengths"])
    gen = torch.Generator().manual_seed(case["seed"] + 53)
    if stage == "state":
        if regime == "conditioned":
            w, u, kg = recompute.reference(raw, case["lengths"])
            w, u, kg = w.bfloat16(), u.bfloat16(), kg.bfloat16()
        else:
            kg, w, u = torch.zeros_like(raw["k"]), torch.zeros_like(raw["k"]), raw["v"]
            if regime == "basis":
                u = torch.zeros_like(u)
                for seq, length in enumerate(case["lengths"]):
                    row = torch.arange(length, device=device)
                    kg[0, starts[seq] + row, :, row % 128] = 1
                    u[0, starts[seq] + row, :, (row + 1) % 128] = 1
        h0 = torch.zeros((sequences, 16, 128, 128), device=device)
        if regime == "identity" or (regime == "conditioned" and case["seed"] == 530902):
            h0.copy_(
                (torch.randn(h0.shape, generator=gen) * 0.03125).bfloat16().float()
            )
        offsets = [0]
        for length in case["lengths"]:
            offsets.append(offsets[-1] + (length + 63) // 64)
        return dict(
            k=kg,
            w=w,
            v=u,
            gk=raw["gk"],
            h0=h0,
            starts=raw["starts"],
            offsets=torch.tensor(offsets, dtype=torch.int32, device=device),
        )
    q, v, A, g = raw["k"], raw["v"], raw["A"], raw["gk"]
    h = torch.zeros((1, count, 16, 128, 128), dtype=torch.bfloat16, device=device)
    if regime == "identity":
        q.zero_()
    elif regime == "basis":
        q.zero_()
        A.zero_()
        row = torch.arange(tokens, device=device)
        q[0, row, :, row % 128] = 1
        h.copy_(torch.eye(128, device=device))
    else:
        h.copy_((torch.randn(h.shape, generator=gen) * 0.03125).bfloat16())
    return dict(q=q, v=v, A=A, g=g, h=h, starts=raw["starts"], chunks=raw["chunks"])


def output_spec(stage, lengths):
    tokens = sum(lengths)
    chunks = sum((n + 63) // 64 for n in lengths)
    if stage == "state":
        return (
            ((1, chunks, 16, 128, 128), "bfloat16"),
            ((1, tokens, 16, 128), "bfloat16"),
            ((len(lengths), 16, 128, 128), "float32"),
        )
    return (((1, tokens, 16, 128), "bfloat16"),)


def mutate(stage, inputs):
    inputs["v"].mul_(0.5)
    inputs["h0" if stage == "state" else "h"].mul_(0.5)
    # Both state and output are linear in these two operands for the exact
    # algebraic fixtures. Keep metadata, gates, routing and other inputs fixed.


def launch(stage, jit, inputs, outputs, config):
    require(config in CONFIGS[stage], "unprescribed config")
    if stage == "state":
        h, v_new, ht = outputs
        return jit[(4, (len(inputs["starts"]) - 1) * 16)](
            k=inputs["k"],
            v=inputs["v"],
            w=inputs["w"],
            v_new=v_new,
            g=None,
            gk=inputs["gk"],
            h=h,
            h0=inputs["h0"],
            ht=ht,
            cu_seqlens=inputs["starts"],
            chunk_offsets=inputs["offsets"],
            T=inputs["k"].shape[1],
            H=16,
            Hg=16,
            K=128,
            V=128,
            BT=64,
            BV=32,
            USE_G=False,
            USE_GK=True,
            USE_INITIAL_STATE=True,
            STORE_FINAL_STATE=True,
            SAVE_NEW_VALUE=True,
            IS_VARLEN=True,
            USE_EXP2=True,
            **config,
        )
    return jit[(2, len(inputs["chunks"]), 16)](
        q=inputs["q"],
        v=inputs["v"],
        g=inputs["g"],
        h=inputs["h"],
        o=outputs[0],
        A=inputs["A"],
        cu_seqlens=inputs["starts"],
        chunk_indices=inputs["chunks"],
        scale=128**-0.5,
        T=inputs["q"].shape[1],
        H=16,
        K=128,
        V=128,
        BT=64,
        BK=64,
        BV=64,
        IS_VARLEN=True,
        **config,
    )


def state_reference(inputs, lengths):
    """Float64 block math with BF16 state/value operand round points.

    Conditioned errors are observations, not accuracy qualification. Exact fixtures
    also have a separate analytic reference below, independent of this recurrence.
    """
    import torch

    snapshots, new_values, final = [], [], []
    start = 0
    for seq, length in enumerate(lengths):
        current = inputs["h0"][seq].double()
        for offset in range(0, length, 64):
            sl = slice(start + offset, start + min(offset + 64, length))
            snapshots.append(current.bfloat16().double())
            w, u, k = [
                inputs[name][0, sl].double().permute(1, 0, 2)
                for name in ("w", "v", "k")
            ]
            narrowed = current.bfloat16().double()
            correction = sum(
                w[:, :, part : part + 64]
                @ narrowed[:, :, part : part + 64].transpose(1, 2)
                for part in (0, 64)
            )
            new_v = u - correction
            new_values.append(new_v.bfloat16().double().permute(1, 0, 2))
            decay = inputs["gk"][0, sl][-1].double().exp2()
            current = (
                current * decay[:, None, :]
                + new_v.bfloat16().double().transpose(1, 2) @ k
            )
            current = current.float().double()
        final.append(current)
        start += length
    return (
        torch.stack(snapshots)[None],
        torch.cat(new_values)[None],
        torch.stack(final),
    )


def output_reference(inputs, lengths):
    import torch

    results = []
    starts, chunks = gate.sequence_metadata(lengths)
    for chunk_index, (seq, chunk) in enumerate(chunks):
        lo = starts[seq] + chunk * 64
        hi = min(lo + 64, starts[seq + 1])
        q = (inputs["q"][0, lo:hi].double() * (128**-0.5)).bfloat16().double()
        qg = (q * inputs["g"][0, lo:hi].double().exp2()).bfloat16().double()
        h = inputs["h"][0, chunk_index].double()
        carry = sum(
            qg[:, :, part : part + 64].permute(1, 0, 2)
            @ h[:, :, part : part + 64].transpose(1, 2)
            for part in (0, 64)
        )
        A = inputs["A"][0, lo:hi, :, : hi - lo].double().permute(1, 0, 2).tril()
        v = inputs["v"][0, lo:hi].double().permute(1, 0, 2)
        results.append((carry + A @ v).permute(1, 0, 2))
    return (torch.cat(results)[None],)


def exact_reference(stage, case, inputs):
    """Independent analytic expected outputs for identity/basis fixtures."""
    import torch

    require(case["regime"] in ("identity", "basis"), "not an exact fixture")
    lengths = case["lengths"]
    if stage == "output":
        if case["regime"] == "identity":
            return (inputs["v"].clone(),)
        # q is one-hot, g=0, h a scaled identity: no reductions with cancellation.
        factor = inputs["h"][0, 0, 0, 0, 0]
        return (
            (
                (inputs["q"].float() * (128**-0.5)).bfloat16().float() * factor
            ).bfloat16(),
        )
    h, final = [], []
    start = 0
    for seq, length in enumerate(lengths):
        current = inputs["h0"][seq].clone()
        for offset in range(0, length, 64):
            h.append(current.bfloat16())
            if case["regime"] == "basis":
                row = torch.arange(
                    offset, min(offset + 64, length), device=current.device
                )
                # One visit per (value,key) pair per128 tokens, max60 visits in
                # this matrix: the integer/half-integer sums remain exact.
                values = inputs["v"][0, start + row, 0, (row + 1) % 128].float()
                current[:, (row + 1) % 128, row % 128] += values
        final.append(current)
        start += length
    return (torch.stack(h)[None], inputs["v"].clone(), torch.stack(final))


def compare(actual, expected):
    import torch

    require(
        actual.shape == expected.shape and actual.dtype == expected.dtype,
        "mismatched output shape/dtype",
    )
    require(
        torch.isfinite(actual).all().item() and torch.isfinite(expected).all().item(),
        "nonfinite output",
    )
    if actual.dtype == torch.bfloat16:
        return recompute.compare(actual.reshape(-1, 128), expected.reshape(-1, 128))
    require(actual.dtype == torch.float32, "unsupported output type")
    return gate.pair_delta(actual, expected)


def reference_metrics(actual, reference):
    result = compare(actual, reference.to(actual.dtype))
    result["fp64_max_abs"] = (actual.double() - reference.double()).abs().max().item()
    return result
