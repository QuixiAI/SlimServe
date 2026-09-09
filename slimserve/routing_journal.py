# SPDX-License-Identifier: Apache-2.0
"""Bounded CPU-side routing evidence, disconnected unless explicitly enabled.

Record actual scheduler steps, not an inferred alignment of independent request
histories. Enabling worker routing capture changes scheduling and adds copies;
these records must never be used as authoritative serving-throughput timings.
"""

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np


def diagnostic_plan(plan, directory):
    """Opt into a deliberately non-baseline capture on the target profile only."""
    if plan.profile_id != "glm53-nvfp4-4" or plan.platform != "rtx6000":
        raise ValueError("routing journal currently supports GLM-5.3 on RTX6000 TP4")
    if plan.speculative:
        raise ValueError("routing journal requires the no-spec profile")
    additional = dict(plan.engine.get("additional_config") or {})
    additional["slimserve_routing_journal"] = str(Path(directory).resolve())
    return replace(
        plan,
        engine={
            **plan.engine,
            "enable_return_routed_experts": True,
            "async_scheduling": False,
            "additional_config": additional,
        },
    )


class RoutingJournal:
    def __init__(
        self,
        directory,
        *,
        model,
        num_layers,
        first_moe_layer,
        num_experts,
        top_k,
        max_tokens=16,
        max_records=4096,
    ):
        if not 0 <= first_moe_layer < num_layers:
            raise ValueError("invalid contiguous MoE layer range")
        if min(num_experts, top_k, max_tokens, max_records) < 1:
            raise ValueError("journal dimensions and bounds must be positive")
        if top_k > num_experts:
            raise ValueError("top_k exceeds the expert count")
        self.num_layers = num_layers
        self.first_moe_layer = first_moe_layer
        self.num_experts = num_experts
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.max_records = max_records
        self.max_steps = 4 * max_records
        self.steps = self.recorded = 0
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"routing-{os.getpid()}.jsonl"
        self.stream = self.path.open("x", buffering=1)
        self._write(
            {
                "kind": "header",
                "schema": 1,
                "model": model,
                "diagnostic_only": True,
                "source": "scheduler step; model-runner request order",
                "num_layers": num_layers,
                "first_moe_layer": first_moe_layer,
                "num_experts": num_experts,
                "top_k": top_k,
                "max_tokens": max_tokens,
                "max_records": max_records,
                "max_steps": self.max_steps,
            }
        )

    def _write(self, record):
        self.stream.write(json.dumps(record, separators=(",", ":")) + "\n")

    def close(self):
        self.stream.close()

    def record(self, data, slots, req_ids, scheduled, computed, prompt_lengths):
        """Called with fully D2H'd routing data and this step's request metadata.

        Only all-decode steps with one token per request and <= max_tokens are
        retained. Prefill/mixed steps are counted as skips; the capture limit is
        explicit in the journal. No request ordering is reconstructed from dicts.
        """
        self.steps += 1
        if self.recorded >= self.max_records:
            return
        if self.steps > self.max_steps:
            if self.steps == self.max_steps + 1:
                self._write({"kind": "step_limit", "steps": self.max_steps})
            return
        counts = [scheduled[rid] for rid in req_ids]
        is_decode = all(
            count == 1 and computed[rid] > prompt_lengths[rid]
            for rid, count in zip(req_ids, counts)
        )
        if not req_ids or not is_decode or len(req_ids) > self.max_tokens:
            self._write({"kind": "skip", "step": self.steps, "counts": counts})
            return
        row = {
            "kind": "decode",
            "step": self.steps,
            "request_ids": list(req_ids),
            "computed_tokens": [computed[rid] for rid in req_ids],
            "slots": np.asarray(slots).tolist(),
            "routes": np.asarray(data)[:, self.first_moe_layer :, :].tolist()
            if np.ndim(data) == 3
            else np.asarray(data).tolist(),
        }
        try:
            if np.shape(data) != (len(req_ids), self.num_layers, self.top_k):
                raise ValueError("routing data does not match this scheduler step")
            if np.shape(slots) != (len(req_ids),):
                raise ValueError("slot mapping does not match this scheduler step")
            active = np.asarray(data)[:, self.first_moe_layer :, :]
            if not np.issubdtype(active.dtype, np.integer):
                raise ValueError("expert IDs must have integer dtype")
            if np.any(active < 0) or np.any(active >= self.num_experts):
                raise ValueError("expert ID outside the configured range")
            ordered = np.sort(active, axis=-1)
            if np.any(ordered[..., 1:] == ordered[..., :-1]):
                raise ValueError("duplicate expert IDs: capture may be unpopulated")
        except ValueError as error:
            row["kind"] = "invalid"
            row["error"] = str(error)
            self._write(row)
            raise
        self._write(row)
        self.recorded += 1
        if self.recorded == self.max_records:
            self._write({"kind": "limit", "recorded": self.recorded})
