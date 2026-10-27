"""Install the older engine's worker interface before Metal config validation."""
import os
import sys


def register():
    if os.environ.get("VLLM_METAL_BACKEND") != "mps":
        return
    from types import SimpleNamespace
    from vllm.config import VllmConfig
    # This older fork predates optional routed-expert auxiliary outputs.
    if not hasattr(VllmConfig, "aux_output_config"):
        VllmConfig.aux_output_config = SimpleNamespace(enabled=False)
    from neuron_compat.buffers import install, copy_to_gpu
    install()
    from neuron_compat import metal_worker
    from vllm.v1.worker.gpu import buffer_utils, model_runner, block_table
    # All groups in the supported Qwen3 model have attention KV slots.
    if not hasattr(block_table.BlockTables, "_slot_mapping_enabled"):
        block_table.BlockTables._slot_mapping_enabled = property(
            lambda self: [True] * len(self.block_sizes)
        )
    buffer_utils.async_copy_to_gpu = copy_to_gpu
    model_runner.async_copy_to_gpu = copy_to_gpu
    # MetalPlatform imports configure_mps from this module at config time.
    sys.modules["vllm_metal.pytorch_backend.worker"] = metal_worker
    from vllm_metal.pytorch_backend.attention import MPSAttentionBackend
    # The older engine asks the backend for the same logical cache shape.
    MPSAttentionBackend.get_kv_cache_shape = staticmethod(
        lambda num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto":
        (num_blocks, num_kv_heads, block_size, 2 * head_size)
    )
    # Match Metal's LBNHC physical order: block, token, head, packed K/V.
    MPSAttentionBackend.get_kv_cache_stride_order = staticmethod(
        lambda include_num_layers_dimension=False:
        (0, 1, 3, 2, 4) if include_num_layers_dimension else (0, 2, 1, 3)
    )
