"""Bridge the steering fork's older staging-buffer API to current Metal."""
import torch


def install():
    from vllm.v1.worker.gpu import buffer_utils
    if hasattr(buffer_utils, "NonUvaBuffer"):
        return

    # Same explicit-copy fallback as upstream vLLM 0.32 buffer_utils.
    class NonUvaBuffer:
        def __init__(self, size, dtype):
            from vllm.platforms import current_platform
            self.cpu = torch.zeros(size, dtype=dtype, device="cpu")
            self.np = self.cpu.numpy()
            self._uva = torch.zeros(size, dtype=dtype, device=current_platform.device_type)

        def uva(self, n=None):
            if n is None:
                return self._uva.copy_(self.cpu, non_blocking=True)
            return self._uva[:n].copy_(self.cpu[:n], non_blocking=True)

    # The older pool reads .uva as a property rather than calling .uva(n).
    class MPSBuffer(NonUvaBuffer):
        def __init__(self, size, dtype):
            super().__init__(size, dtype)
            self._uva = self._uva.to("mps")

        @property
        def uva(self):
            result = super().uva()
            result._metal_cpu = self.cpu
            return result

    buffer_utils.NonUvaBuffer = NonUvaBuffer
    buffer_utils.UvaBuffer = MPSBuffer


def copy_to_gpu(x, out=None, device=None):
    source = torch.as_tensor(x)
    result = source.to(device) if out is None else out.copy_(source)
    result._metal_cpu = source
    return result
