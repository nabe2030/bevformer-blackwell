# torch_profiler_hook.py
# mmcv 1.x Hook wrapping torch.profiler for Tensor Core kernel measurement (NCU alternative).
# Used in Run B/C as fallback when NCU profiling blocked by container ERR_NVGPUCTRPERM.

import os
import torch
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class TorchProfilerHook(Hook):
    """
    Profile training iterations using torch.profiler. Outputs tensorboard trace +
    kernel time summary for Tensor Core utilization analysis.

    Schedule: skip `wait` iters → warmup `warmup` iters → record `active` iters → repeat `repeat` times.
    With BEVFormer mini benchmark, default (wait=50, warmup=10, active=40) profiles iters 60-99.
    """

    def __init__(self,
                 output_dir,
                 wait=50,
                 warmup=10,
                 active=40,
                 repeat=1,
                 record_shapes=False,
                 with_stack=False):
        self.output_dir = output_dir
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self.record_shapes = record_shapes
        self.with_stack = with_stack
        self.profiler = None

    def before_run(self, runner):
        os.makedirs(self.output_dir, exist_ok=True)
        # Only rank 0 writes profiler output to avoid contention
        if runner.rank != 0:
            return
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=self.wait,
                warmup=self.warmup,
                active=self.active,
                repeat=self.repeat,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.output_dir),
            record_shapes=self.record_shapes,
            with_stack=self.with_stack,
        )
        self.profiler.start()
        runner.logger.info(
            f'TorchProfilerHook started: output={self.output_dir} '
            f'schedule(wait={self.wait}, warmup={self.warmup}, '
            f'active={self.active}, repeat={self.repeat})')

    def after_train_iter(self, runner):
        if self.profiler is not None:
            self.profiler.step()

    def after_run(self, runner):
        if self.profiler is not None:
            self.profiler.stop()
            # Export key averages summary as TSV for easy host-side parsing
            try:
                key_averages = self.profiler.key_averages()
                summary_path = os.path.join(self.output_dir, 'kernel_summary.tsv')
                with open(summary_path, 'w') as f:
                    f.write('name\tcuda_time_us\tcpu_time_us\tcalls\n')
                    for ev in key_averages:
                        name = getattr(ev, 'key', None) or ev.node_id
                        cuda_us = getattr(ev, 'cuda_time_total', 0)
                        cpu_us = getattr(ev, 'cpu_time_total', 0)
                        calls = getattr(ev, 'count', 0)
                        f.write(f'{name}\t{cuda_us}\t{cpu_us}\t{calls}\n')
                runner.logger.info(f'TorchProfilerHook kernel summary: {summary_path}')
            except Exception as e:
                runner.logger.warning(f'TorchProfilerHook summary export failed: {e}')
            self.profiler = None
