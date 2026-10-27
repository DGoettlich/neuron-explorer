"""Single-device MPS control-plane adapter for the steering fork."""
import time
import torch
from vllm.distributed import init_distributed_environment, ensure_model_parallel_initialized
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.worker_base import WorkerBase, CompilationTimes


def init_worker_distributed_environment(config, rank, method, local_rank):
    p = config.parallel_config
    init_distributed_environment(p.world_size, rank, method, local_rank, backend="gloo")
    ensure_model_parallel_initialized(p.tensor_parallel_size, p.pipeline_parallel_size)


class MetalWorker(WorkerBase):
    def load_model(self):
        self.model_runner.load_model()

    def get_kv_cache_spec(self):
        return self.model_runner.get_kv_cache_spec()

    def initialize_from_config(self, config):
        self.model_runner.initialize_kv_cache(config)

    def compile_or_warm_up_model(self):
        set_random_seed(self.model_config.seed)
        start = time.perf_counter()
        self.model_runner.warm_up()
        return CompilationTimes(time.perf_counter() - start, 0.0)

    def drain_pending_capture_results(self):
        return self.model_runner.drain_pending_capture_results()

    def execute_model(self, scheduler_output):
        return self.model_runner.execute_model(scheduler_output)

    def sample_tokens(self, grammar_output):
        return self.model_runner.sample_tokens(grammar_output)

    def get_model(self):
        return self.model_runner.model

    def get_cache_block_size_bytes(self):
        return self.model_runner.get_cache_block_size_bytes()

    def get_supported_tasks(self):
        return self.model_runner.get_supported_tasks()

    def take_draft_token_ids(self):
        return None

    def shutdown(self):
        # Flush queued capture writes before the worker exits.
        manager = getattr(self.model_runner, "_capture_manager", None)
        if manager is not None:
            manager.shutdown()
        torch.mps.empty_cache()
