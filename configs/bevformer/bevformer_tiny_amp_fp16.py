# bevformer_tiny_amp_fp16.py — AMP ON (FP16) variant of bevformer_tiny.
#
# Inherits bevformer_tiny.py verbatim and adds the mmcv 1.x `fp16` key, which
# activates Fp16OptimizerHook (autocast). Symmetric to bevformer_tiny_amp_fp32.py;
# the only diff is the `fp16 = dict(loss_scale=512.)` line.
#
# Dataset resolution: same as bevformer_tiny.py — cwd=/workspace.

_base_ = ['./bevformer_tiny.py']

# AMP ON — mmcv 1.x Fp16OptimizerHook
fp16 = dict(loss_scale=512.)
