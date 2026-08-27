# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

DynamicSDSchedule = list[tuple[int, int, int]]


def validate_and_normalize_dynamic_sd_schedule(
    num_speculative_tokens_per_batch_size: object,
) -> DynamicSDSchedule:
    """Validate and normalize a Dynamic SD batch-size schedule.

    The schedule is expressed as a list of inclusive ranges:

    ``[(range_start, range_end, num_speculative_tokens), ...]``
    """
    if num_speculative_tokens_per_batch_size is None:
        raise ValueError(
            "num_speculative_tokens_per_batch_size is required for "
            "dynamic speculative decoding."
        )
    if not isinstance(num_speculative_tokens_per_batch_size, list):
        raise ValueError(
            "num_speculative_tokens_per_batch_size must be a non-empty list of "
            "(range_start, range_end, num_speculative_tokens) entries."
        )
    if not num_speculative_tokens_per_batch_size:
        raise ValueError("num_speculative_tokens_per_batch_size must not be empty.")

    parsed_schedule: DynamicSDSchedule = []
    for entry in num_speculative_tokens_per_batch_size:
        if not isinstance(entry, list | tuple) or len(entry) != 3:
            raise ValueError(
                "Each num_speculative_tokens_per_batch_size entry must be a "
                "3-item sequence: (range_start, range_end, num_speculative_tokens)."
            )

        range_start, range_end, num_speculative_tokens = (
            int(entry[0]),
            int(entry[1]),
            int(entry[2]),
        )

        if range_start <= 0 or range_end <= 0:
            raise ValueError(
                f"Batch-size range ({range_start}, {range_end}) must be positive."
            )
        if range_start > range_end:
            raise ValueError(
                "Batch-size range start must be <= end for "
                f"({range_start}, {range_end}, {num_speculative_tokens})."
            )
        if num_speculative_tokens < 0:
            raise ValueError(
                "num_speculative_tokens_per_batch_size values must be >= 0."
            )

        parsed_schedule.append((range_start, range_end, num_speculative_tokens))

    parsed_schedule.sort(key=lambda entry: entry[0])

    previous_end = 0
    for range_start, range_end, _ in parsed_schedule:
        if range_start <= previous_end:
            raise ValueError("Batch-size ranges must be non-overlapping and sorted.")
        previous_end = range_end

    first_range_start = parsed_schedule[0][0]
    if first_range_start != 1:
        raise ValueError(
            "The first batch-size range must start at 1 so every runtime "
            "batch size has a defined schedule."
        )

    return parsed_schedule


def build_dynamic_sd_schedule_lookup(
    num_speculative_tokens_per_batch_size: object,
    vllm_max_batch_size: int,
    vllm_num_speculative_tokens: int,
) -> list[int]:
    """Expand the configured schedule into a dense batch_size -> K lookup.

    "dense_schedule" means a 1-indexed lookup table where index ``batch_size``
    stores the exact K to use for that runtime batch size. This lets the
    scheduler do a simple array lookup instead of searching the configured
    ranges on every scheduling step.
    """
    if vllm_max_batch_size <= 0:
        raise ValueError("vllm_max_batch_size must be > 0.")
    if vllm_num_speculative_tokens <= 0:
        raise ValueError("vllm_num_speculative_tokens must be > 0.")

    parsed_schedule = validate_and_normalize_dynamic_sd_schedule(
        num_speculative_tokens_per_batch_size
    )

    # Index 0 is intentionally unused so that valid runtime batch sizes can be
    # looked up directly as dense_schedule[batch_size].
    dense_schedule = [0] * (vllm_max_batch_size + 1)
    next_batch_size = 1
    last_num_speculative_tokens: int | None = None

    for range_start, range_end, num_speculative_tokens in parsed_schedule:
        if range_start > next_batch_size and last_num_speculative_tokens is not None:
            # Fill any gap before the next configured range by carrying forward
            # the previous K. For example, [(1, 16, 3), (32, 128, 2)] should map
            # batch sizes 17-31 to K=3.
            for batch_size in range(
                next_batch_size,
                min(range_start, vllm_max_batch_size + 1),
            ):
                dense_schedule[batch_size] = min(
                    vllm_num_speculative_tokens,
                    last_num_speculative_tokens,
                )

        # Fill the current configured inclusive range with its K value.
        for batch_size in range(
            max(range_start, next_batch_size),
            min(range_end, vllm_max_batch_size) + 1,
        ):
            dense_schedule[batch_size] = min(
                vllm_num_speculative_tokens,
                num_speculative_tokens,
            )

        next_batch_size = max(next_batch_size, range_end + 1)
        last_num_speculative_tokens = num_speculative_tokens

        if next_batch_size > vllm_max_batch_size:
            break

    if last_num_speculative_tokens is None:
        raise ValueError(
            "num_speculative_tokens_per_batch_size must contain at least "
            "one valid batch-size range."
        )

    # Fill the tail after the final configured range by carrying forward the
    # last K through vllm_max_batch_size.
    for batch_size in range(next_batch_size, vllm_max_batch_size + 1):
        dense_schedule[batch_size] = min(
            vllm_num_speculative_tokens,
            last_num_speculative_tokens,
        )

    return dense_schedule


class AcceptanceThrottle:
    """Pause drafting while measured acceptance says it is a net loss.

    A fixed per-batch-size schedule cannot serve both content regimes: the
    same DFlash2 drafter measured mean acceptance 3.7 (of a k=3 max 4) on
    essay prose and 1.2 on Shakespeare verse, which at c8 is +51% with
    drafting on prose and -35% on verse (2026-08-27 sweep,
    perf/results/2026-08-27/nvfp4-baseline/). This throttle keeps drafting
    on by default and reacts to the measured draft efficiency:

    - ``observe()`` folds each step's (drafted, accepted) into an EMA of
      accepted/drafted.
    - ``gate()`` passes the scheduled K through while the EMA is healthy;
      after ``warmup_calls`` observations, an EMA below ``min_ratio``
      pauses drafting for ``pause_steps`` scheduling steps, then re-probes
      (fresh warmup) so drafting recovers when the content changes.

    Break-even is BATCH-DEPENDENT: at batch 1 the verify rows ride the
    small-M GEMV band and are nearly free, so drafting measured net
    -positive (+12%) even at ratio 0.16 on hostile verse -- batches
    below ``min_batch`` are therefore exempt (never gated), which also
    keeps batch-1 runs sha-deterministic. At batch >= min_batch the
    verify rides wider GEMM and ratio 0.33 is ~one accepted token per
    draft call, roughly break-even; prose measures ~0.90 and verse
    ~0.07, so the 0.30 default sits in a wide gap. All knobs are
    env-tunable and the throttle only exists when
    VLLM_SD_ADAPT_THROTTLE=1, so platforms that have not re-gated keep
    their exact scheduler behavior.
    """

    def __init__(
        self,
        min_ratio: float = 0.30,
        pause_steps: int = 96,
        warmup_calls: int = 8,
        ema_alpha: float = 0.1,
        min_batch: int = 2,
    ) -> None:
        self.min_batch = min_batch
        self.min_ratio = min_ratio
        self.pause_steps = pause_steps
        self.warmup_calls = warmup_calls
        self.ema_alpha = ema_alpha
        self._ema: float | None = None
        self._calls_in_mode = 0
        self._pause_remaining = 0

    @classmethod
    def from_env(cls) -> "AcceptanceThrottle | None":
        import os

        if os.environ.get("VLLM_SD_ADAPT_THROTTLE", "0") != "1":
            return None
        return cls(
            min_ratio=float(os.environ.get("VLLM_SD_ADAPT_MIN_RATIO", "0.30")),
            pause_steps=int(os.environ.get("VLLM_SD_ADAPT_PAUSE_STEPS", "96")),
            warmup_calls=int(os.environ.get("VLLM_SD_ADAPT_WARMUP_CALLS", "8")),
            min_batch=int(os.environ.get("VLLM_SD_ADAPT_MIN_BATCH", "2")),
        )

    def observe(self, num_draft_tokens: int, num_accepted_tokens: int) -> None:
        if num_draft_tokens <= 0:
            return
        ratio = num_accepted_tokens / num_draft_tokens
        if self._ema is None:
            self._ema = ratio
        else:
            self._ema += self.ema_alpha * (ratio - self._ema)
        self._calls_in_mode += 1

    def gate(self, num_spec_tokens: int, batch_size: int = 0) -> int:
        if num_spec_tokens <= 0:
            return num_spec_tokens
        if 0 < batch_size < self.min_batch:
            # Small-batch verify is nearly free; drafting stays on and
            # the pause clock does not tick (a hostile pause still
            # applies to the next wide-batch step).
            return num_spec_tokens
        if self._pause_remaining > 0:
            self._pause_remaining -= 1
            if self._pause_remaining == 0:
                # Re-probe: draft again with a fresh warmup so a content
                # change can lift the pause; a still-hostile stream just
                # pauses again after warmup_calls cheap steps.
                self._calls_in_mode = 0
                self._ema = None
            return 0
        if (
            self._calls_in_mode >= self.warmup_calls
            and self._ema is not None
            and self._ema < self.min_ratio
        ):
            # The triggering step is the first paused step, so a pause is
            # exactly pause_steps scheduling steps long.
            self._pause_remaining = self.pause_steps - 1
            if self._pause_remaining == 0:
                self._calls_in_mode = 0
                self._ema = None
            return 0
        return num_spec_tokens
