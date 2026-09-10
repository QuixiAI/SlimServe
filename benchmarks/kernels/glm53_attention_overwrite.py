# SPDX-License-Identifier: Apache-2.0
"""KV-only extra-launch diagnostic; not a production optimization."""


class KVOnlyOverwrite:
    """Original combo ABI, followed by split KV on its original output address.

    Launchers must already be source/config/binary-qualified. This class does
    not load kernels, tune them, copy tensors or install itself into serving.
    Host-call counters include capture, but not GPU-only graph replays.
    """

    def __init__(self, combo, split_kv):
        if not callable(combo) or not callable(split_kv) or combo is split_kv:
            raise ValueError("two distinct precompiled launchers required")
        self.combo = combo
        self.split_kv = split_kv
        self.combo_calls = self.kv_calls = 0

    def run(self, *args, stream):
        if len(args) != 11:
            raise ValueError("original combo ABI requires eleven positional arguments")
        rows = args[8]
        if type(rows) is not int or rows <= 0 or args[9:11] != (rows, rows):
            raise ValueError("matching positive row counts required")
        result = self.combo(*args, stream=stream)
        self.combo_calls += 1
        # packed projection, KV norm weight, KV destination, row/column counts.
        self.split_kv(args[0], args[1], args[5], rows, 512, stream=stream)
        self.kv_calls += 1
        return result

    __call__ = run
