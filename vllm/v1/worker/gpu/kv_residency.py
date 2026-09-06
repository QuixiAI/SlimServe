# SPDX-License-Identifier: Apache-2.0
"""Host-resident main KV: GPU hot window + pinned-host rows, one offset table.

docs/host_resident_kv_design.md. The QSA main-KV layers leave the packed slab
for a tensor whose logical blocks are backed by pinned host rows (readable by
the sparse gather over PCIe through UVA) with a small window of GPU rows for
the blocks being written. This object owns both backings and the per-block
location, and each step it

1. binds every block the step writes to a GPU row (demoting the coldest
   unprotected GPU-resident block to its host row first when the window is
   full - a copy on the current stream, so no kernel observes a page mid-move),
2. refreshes the per-request page-offset table the gather kernel reads
   (``qsa_sparse_paged_attention(page_offsets=...)``), and
3. remaps the group's slot mapping from logical blocks to GPU rows for the
   cache store.

All decisions are made from CPU-side block tables (no device sync); device
tables are updated with a handful of small index writes per step.
"""

from __future__ import annotations

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

PAGE_ALIGN = 4096
_HOST_REGISTER_PORTABLE = 0x01
_HOST_REGISTER_MAPPED = 0x02
# Mirrors ops/qsa.py::PTR_SENTINEL (imported lazily there to avoid a cycle).
PTR_SENTINEL = -(1 << 62)
PAD_SLOT_ID = -1

_ACTIVE: "MainKVResidency | None" = None


def get_main_kv_residency() -> "MainKVResidency | None":
    return _ACTIVE


class MainKVResidency:
    def __init__(
        self,
        num_blocks: int,
        gpu_rows: int,
        row_bytes: int,
        device: torch.device,
    ) -> None:
        assert num_blocks > 0 and gpu_rows >= 2 and row_bytes % 16 == 0
        self.num_blocks = num_blocks
        self.gpu_rows = gpu_rows
        self.row_bytes = row_bytes
        self.device = device
        # GPU hot window. Row 0 is scratch: dummy/capture runs write there.
        self.gpu = torch.zeros((gpu_rows, row_bytes), dtype=torch.int8, device=device)
        # Pinned host rows, page-aligned, registered PORTABLE|MAPPED so the
        # gather kernel can read them through the host pointer (UVA).
        total = num_blocks * row_bytes
        self._raw = torch.empty(total + PAGE_ALIGN, dtype=torch.int8)
        off = (-self._raw.data_ptr()) % PAGE_ALIGN
        self.host = self._raw[off : off + total].view(num_blocks, row_bytes)
        rc = torch.cuda.cudart().cudaHostRegister(
            self._raw.data_ptr(),
            self._raw.numel(),
            _HOST_REGISTER_PORTABLE | _HOST_REGISTER_MAPPED,
        )
        if rc != 0:
            raise RuntimeError(f"main-kv: cudaHostRegister failed ({rc})")
        self._registered = True
        self.host.zero_()
        self.gpu_base = self.gpu.data_ptr()
        self.host_base = self.host.data_ptr()

        # CPU bookkeeping.
        # Tier integration (milestone 4): a scheduler block's home is a main-KV
        # tier slot (byte address of its first sub-row) when the tier reserved
        # one, else the residency's own pool row; dirty marks GPU rows whose
        # bytes are not yet at the home.
        self.home_addr: dict[int, int] = {}  # scheduler block -> slot address
        self.dirty = np.zeros(num_blocks, dtype=bool)  # per residency row
        self.tier_arena: torch.Tensor | None = None
        self._tier_raw: torch.Tensor | None = None
        self.row_of_block = np.full(num_blocks, -1, dtype=np.int32)
        self.block_of_row = np.full(gpu_rows, -1, dtype=np.int32)
        self.free_rows: list[int] = list(range(gpu_rows - 1, 0, -1))
        self.last_use = np.zeros(num_blocks, dtype=np.int64)
        self.step = 0
        self.demotions = 0
        self.binds = 0
        # Device tables: byte offset of each block's location relative to
        # the GPU window base (negative allowed), and its GPU row (-1: host).
        host_delta = (self.host_base - self.gpu_base) + np.arange(num_blocks, dtype=np.int64) * row_bytes
        self.page_delta = torch.from_numpy(host_delta).to(device)
        self.row_of_block_dev = torch.full((num_blocks,), -1, dtype=torch.int32, device=device)
        # Per-step outputs (persistent, sized on first use).
        self.page_offsets: torch.Tensor | None = None
        self.slot_mapping: torch.Tensor | None = None
        self.block_size = 0  # tokens per residency row (kernel page)
        self.manager_block_size = 0  # tokens per scheduler block
        self.sub_blocks = 1  # rows per scheduler block
        self.group_id = -1
        logger.info(
            "main-kv: %d logical blocks x %d bytes: %d GPU hot rows (%.2f GiB), "
            "%.2f GiB pinned host rows",
            num_blocks,
            row_bytes,
            gpu_rows,
            gpu_rows * row_bytes / (1 << 30),
            total / (1 << 30),
        )

    # ---------------------------------------------------------------- binding

    def bind(self, group_id: int, block_size: int, max_reqs: int, table_width: int, max_tokens: int) -> None:
        self.group_id = group_id
        self.block_size = block_size
        if not self.manager_block_size:
            self.manager_block_size = block_size * self.sub_blocks
        self.page_offsets = torch.full(
            (max_reqs, table_width * self.sub_blocks),
            PTR_SENTINEL,
            dtype=torch.int64,
            device=self.device,
        )
        self.slot_mapping = torch.full((max_tokens,), PAD_SLOT_ID, dtype=torch.int64, device=self.device)

    # ------------------------------------------------------- tier integration

    def attach_tier_arena(self, num_slots: int, slot_bytes: int) -> int:
        """Pinned MAPPED host arena of main-KV tier slots (one scheduler block
        of sub-rows per slot). Returns its base address."""
        assert slot_bytes == self.row_bytes * self.sub_blocks
        total = num_slots * slot_bytes
        self._tier_raw = torch.empty(total + PAGE_ALIGN, dtype=torch.int8)
        off = (-self._tier_raw.data_ptr()) % PAGE_ALIGN
        self.tier_arena = self._tier_raw[off : off + total].view(num_slots, slot_bytes)
        rc = torch.cuda.cudart().cudaHostRegister(
            self._tier_raw.data_ptr(),
            self._tier_raw.numel(),
            _HOST_REGISTER_PORTABLE | _HOST_REGISTER_MAPPED,
        )
        if rc != 0:
            raise RuntimeError(f"main-kv tier arena: cudaHostRegister failed ({rc})")
        self._tier_registered = True
        logger.info(
            "main-kv tier arena: %d slots x %d bytes (%.2f GiB pinned)",
            num_slots, slot_bytes, total / (1 << 30),
        )
        return self.tier_arena.data_ptr()

    def _home_row_addr(self, row: int) -> int:
        """Where residency row `row` lives when not on the GPU."""
        block = row // self.sub_blocks
        home = self.home_addr.get(block)
        if home is not None:
            return home + (row % self.sub_blocks) * self.row_bytes
        return self.host_base + row * self.row_bytes

    def _host_view(self, addr: int) -> torch.Tensor:
        """A [row_bytes] int8 view of pinned host memory at `addr` (pool or
        tier arena)."""
        if self.tier_arena is not None:
            base = self.tier_arena.data_ptr()
            if base <= addr < base + self.tier_arena.numel():
                off = addr - base
                return self.tier_arena.view(-1)[off : off + self.row_bytes]
        off = addr - self.host_base
        assert 0 <= off < self.host.numel()
        return self.host.view(-1)[off : off + self.row_bytes]

    def set_home(self, block: int, slot_addr: int) -> None:
        """Scheduler block `block` now demotes into the tier slot at
        `slot_addr` (called when the block is allocated, before writes)."""
        self.home_addr[block] = slot_addr
        # Rows already demoted to the pool (window pressure before the home
        # arrived) move to the slot so the slot ends up complete.
        for k in range(self.sub_blocks):
            row = block * self.sub_blocks + k
            if self.row_of_block[row] < 0:
                cur = int(self.page_delta[row].item()) + self.gpu_base
                if cur == self.host_base + row * self.row_bytes and self.dirty[row]:
                    dst = slot_addr + k * self.row_bytes
                    self._host_view(dst).copy_(self._host_view(cur))
                    self.page_delta[row] = dst - self.gpu_base
                    self.dirty[row] = False

    def clear_home(self, block: int) -> None:
        self.home_addr.pop(block, None)

    def flush(self, block: int) -> None:
        """The block is full: its GPU-resident dirty rows are copied into
        the home slot (rows stay hot); pool-resident dirty rows move."""
        home = self.home_addr.get(block)
        if home is None:
            return
        for k in range(self.sub_blocks):
            row = block * self.sub_blocks + k
            dst = home + k * self.row_bytes
            gpu_row = int(self.row_of_block[row])
            if gpu_row >= 0:
                if self.dirty[row]:
                    self._host_view(dst).copy_(self.gpu[gpu_row], non_blocking=True)
                    self.dirty[row] = False
            else:
                cur = int(self.page_delta[row].item()) + self.gpu_base
                if cur != dst:
                    self._host_view(dst).copy_(self._host_view(cur))
                    self.page_delta[row] = dst - self.gpu_base
                self.dirty[row] = False

    def rebind(self, block: int, slot_addr: int) -> None:
        """A restored block: point its rows at the tier slot (read-only, no
        copy). Any GPU row still bound to a previous use is released."""
        rows_freed = []
        for k in range(self.sub_blocks):
            row = block * self.sub_blocks + k
            gpu_row = int(self.row_of_block[row])
            if gpu_row >= 0:
                self.row_of_block[row] = -1
                self.block_of_row[gpu_row] = -1
                self.free_rows.append(gpu_row)
                rows_freed.append(row)
            self.page_delta[row] = slot_addr + k * self.row_bytes - self.gpu_base
            self.dirty[row] = False
        self.home_addr[block] = slot_addr
        if rows_freed:
            idx = torch.tensor(rows_freed, dtype=torch.int64, device=self.device)
            self.row_of_block_dev[idx] = -1

    def layer_raw(self) -> torch.Tensor:
        """The GPU window as the layers' raw backing ([gpu_rows * row_bytes])."""
        return self.gpu.view(-1)

    # ------------------------------------------------------------------- step

    def _bind_block(self, block: int, protected: set[int]) -> None:
        if self.row_of_block[block] >= 0:
            return
        if self.free_rows:
            row = self.free_rows.pop()
        else:
            # Demote the coldest unprotected GPU-resident block.
            victim, victim_row, oldest = -1, -1, None
            for r in range(1, self.gpu_rows):
                b = int(self.block_of_row[r])
                if b < 0 or b in protected:
                    continue
                if oldest is None or self.last_use[b] < oldest:
                    victim, victim_row, oldest = b, r, self.last_use[b]
            if victim < 0:
                raise RuntimeError(
                    "main-kv: GPU hot window exhausted by protected blocks; "
                    "raise main_kv_gpu_rows"
                )
            dst = self._home_row_addr(victim)
            if self.dirty[victim]:
                self._host_view(dst).copy_(self.gpu[victim_row], non_blocking=True)
                self.dirty[victim] = False
            self.page_delta[victim] = dst - self.gpu_base
            self.row_of_block[victim] = -1
            self.block_of_row[victim_row] = -1
            self._changed.append(victim)
            self.demotions += 1
            row = victim_row
        # Fresh content is about to be written: a stale row must not leak
        # another block's bytes into unwritten positions.
        self.gpu[row].zero_()
        self.row_of_block[block] = row
        self.block_of_row[row] = block
        self.dirty[block] = True
        self._changed.append(block)
        self.binds += 1

    def prepare_step(
        self,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
        written_blocks: list[int],
        protected_blocks: list[int],
        dummy: bool = False,
    ) -> None:
        """Bind this step's written blocks, refresh the device tables.

        ``block_table``/``slot_mapping`` are this group's persistent device
        buffers for the step (the same objects the metadata builders see).
        """
        assert self.page_offsets is not None and self.slot_mapping is not None
        import time as _time

        _t0 = _time.perf_counter()
        self.step += 1
        num_reqs, width = block_table.shape
        num_tokens = slot_mapping.shape[0]
        if dummy:
            # Profiling and graph capture: every page resolves to scratch row
            # 0 and every write lands there.
            self.page_offsets[:num_reqs, : width * self.sub_blocks].zero_()
            slots = slot_mapping.to(torch.int64)
            self.slot_mapping[:num_tokens] = torch.where(
                slots >= 0, slots % self.block_size, slots
            )
            return
        self._changed: list[int] = []
        if self.step % 500 == 0:
            logger.info("main-kv residency: %s", self.stats())
        protected = set(protected_blocks) | set(written_blocks)
        for b in sorted(set(written_blocks)):
            self._bind_block(b, protected)
            self.last_use[b] = self.step
        if self._changed:
            changed = sorted(set(self._changed))
            gpu_bound = [b for b in changed if self.row_of_block[b] >= 0]
            if gpu_bound:
                idx = torch.tensor(gpu_bound, dtype=torch.int64)
                rows = torch.from_numpy(self.row_of_block[idx.numpy()])
                idx_dev = idx.to(self.device, non_blocking=True)
                self.page_delta[idx_dev] = (rows.to(torch.int64) * self.row_bytes).to(
                    self.device, non_blocking=True
                )
            idx_all = torch.tensor(changed, dtype=torch.int64)
            self.row_of_block_dev[idx_all.to(self.device, non_blocking=True)] = torch.from_numpy(
                self.row_of_block[idx_all.numpy()]
            ).to(self.device, non_blocking=True)
        # Page offsets at row (kernel page) granularity: scheduler block b
        # owns rows b*sub .. b*sub+sub-1; null entries keep the sentinel.
        sub = self.sub_blocks
        table = block_table.to(torch.int64)
        rows_idx = (table.clamp_min(0) * sub).unsqueeze(-1) + torch.arange(
            sub, device=self.device, dtype=torch.int64
        )
        offs = self.page_delta[rows_idx].reshape(num_reqs, width * sub)
        null = (table < 0).unsqueeze(-1).expand(-1, -1, sub).reshape(num_reqs, width * sub)
        self.page_offsets[:num_reqs, : width * sub] = torch.where(
            null, torch.full_like(offs, PTR_SENTINEL), offs
        )
        # Slot mapping: logical block -> GPU row.
        slots = slot_mapping.to(torch.int64)
        logical = torch.where(slots >= 0, slots // self.block_size, torch.zeros_like(slots))
        rows = self.row_of_block_dev[logical].to(torch.int64)
        phys = rows * self.block_size + slots % self.block_size
        self.slot_mapping[:num_tokens] = torch.where((slots >= 0) & (rows >= 0), phys, torch.full_like(slots, PAD_SLOT_ID))
        self.cpu_seconds = getattr(self, "cpu_seconds", 0.0) + (_time.perf_counter() - _t0)

    def stats(self) -> dict[str, int]:
        return {
            "gpu_rows": self.gpu_rows,
            "resident": int((self.row_of_block >= 0).sum()),
            "binds": self.binds,
            "demotions": self.demotions,
            "steps": self.step,
            "cpu_ms_per_step": round(1000 * getattr(self, "cpu_seconds", 0.0) / max(1, self.step), 3),
        }

    def release(self) -> None:
        if getattr(self, "_registered", False):
            torch.cuda.synchronize(self.device)
            torch.cuda.cudart().cudaHostUnregister(self._raw.data_ptr())
            self._registered = False
        if getattr(self, "_tier_registered", False):
            torch.cuda.cudart().cudaHostUnregister(self._tier_raw.data_ptr())
            self._tier_registered = False

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:  # noqa: BLE001
            pass


def set_main_kv_residency(residency: "MainKVResidency | None") -> None:
    global _ACTIVE
    _ACTIVE = residency
