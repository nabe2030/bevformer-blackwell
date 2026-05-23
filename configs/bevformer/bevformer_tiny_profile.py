# bevformer_tiny_profile.py — TorchProfilerHook を有効化した bevformer_tiny。
#
# Used by the public release image (nabe2030/bevformer:blackwell-pt2.8) for the
# "Measure Tensor Core utilization" recipe in README.
#
# Phase 4-5 で実証済の hook 設定:
#   wait=20  → 最初 20 iter を warmup 飛ばし
#   warmup=5 → 5 iter warmup (record しない)
#   active=30 → 30 iter を record
#   repeat=1 → 1 cycle のみ
#
# Output: /workspace/runs/profile/torch_profile/<host>_<pid>...pt.trace.json
# Parse:  python /opt/bevformer/scripts/tensor_core_ratio.py <trace.json>

_base_ = ['./bevformer_tiny.py']

# AMP ON (FP16 で TC kernel utilization を測定 — FP32 では TC kernel は限定的)
fp16 = dict(loss_scale=512.)

# TorchProfilerHook を custom_hooks に注入。
# Hook 実装は projects/mmdet3d_plugin/bevformer/hooks/torch_profiler_hook.py
custom_hooks = [
    dict(
        type='TorchProfilerHook',
        output_dir='/workspace/runs/profile/torch_profile',
        wait=20,
        warmup=5,
        active=30,
        repeat=1,
        record_shapes=False,
        with_stack=False,
    ),
]
