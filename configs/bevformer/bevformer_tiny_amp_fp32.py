# bevformer_tiny_amp_fp32.py — AMP A/B baseline (AMP OFF, FP32)
#
# Inherits bevformer_tiny.py verbatim (data path + model + runner + log_config)
# to match the dataset resolution path that Cmd 1 (training) was verified with.
# The only purpose of this file is to be the FP32 baseline in the AMP A/B
# benchmark — symmetric to bevformer_tiny_amp_fp16.py which adds `fp16 = ...`.
#
# Dataset resolution: cwd MUST be /workspace at invocation time, because
# bevformer_tiny.py uses `data_root = 'data/nuscenes/'` (relative). Same
# requirement as Cmd 1 (training).
#
# No `fp16 = ...` key — AMP OFF is identified by its absence.

_base_ = ['./bevformer_tiny.py']
