"""MPS dispatch and capture transport for the unchanged steering engine."""
import torch

_LIBRARY = None


def install():
    global _LIBRARY
    if _LIBRARY is not None:
        return
    from vllm.model_executor.layers import activation_capture, steering, clamp
    from vllm.v1.capture.manager import CaptureManager

    _LIBRARY = torch.library.Library("vllm", "FRAGMENT")
    for name, function in (
        ("apply_steering", steering.apply_steering),
        ("apply_clamp", clamp.apply_clamp),
        ("apply_clamp_block", clamp.apply_clamp_block),
        ("steering_monitor", steering.steering_monitor),
        ("capture_residual", activation_capture._capture_residual_impl),
        ("capture_residual_add", activation_capture._capture_residual_add_impl),
    ):
        _LIBRARY.impl(name, function, dispatch_key="MPS")

    enqueue = CaptureManager._enqueue_packet

    def enqueue_cpu_snapshot(self, packet):
        # Synchronous copy completes before the worker reuses capture buffers.
        packet.scratch_pinned = {
            key: (pinned, tensor.cpu() if tensor.device.type == "mps" else tensor)
            for key, (pinned, tensor) in packet.scratch_pinned.items()
        }
        return enqueue(self, packet)

    CaptureManager._enqueue_packet = enqueue_cpu_snapshot
