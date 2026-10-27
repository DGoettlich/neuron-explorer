"""Conservative local MPS worker budget, applied before model allocation."""
import os

from neuron_compat.buffers import install

install()

from neuron_compat.metal_worker import MPSWorker


class BoundedMPSWorker(MPSWorker):
    def init_device(self):
        import psutil
        import torch

        # These must also be present in the launching process before importing torch.
        if os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO") != "0.9" or os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO") != "0.8":
            raise RuntimeError("Launch with the required MPS watermarks (0.9 / 0.8)")
        self._memory_budget = 3 * 1024**3
        if psutil.virtual_memory().available < self._memory_budget + 512 * 1024**2:
            raise RuntimeError("Not enough free memory for the smoke run plus 512 MiB system headroom")
        torch.mps.set_per_process_memory_fraction(
            min(0.9, self._memory_budget / torch.mps.recommended_max_memory())
        )
        from neuron_compat.engine import install

        install()
        super().init_device()
        print(f"MPS startup buffers: {torch.mps.driver_allocated_memory() / 1024**3:.2f} GiB; allocator cap: 3 GiB", flush=True)

    def determine_available_memory(self):
        import torch

        available = super().determine_available_memory()
        # Keep 512 MiB of the worker budget for activations and steering.
        remaining = self._memory_budget - torch.mps.driver_allocated_memory() - 512 * 1024**2
        if available > remaining:
            raise RuntimeError("Requested KV cache would exceed the worker's MPS budget")
        return available
