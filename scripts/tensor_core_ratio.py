#!/usr/bin/env python3
"""tensor_core_ratio.py — Aggregate torch.profiler chrome trace JSON to
extract Tensor Core kernel utilization ratio.

This is the Phase 4 / Phase 5 aggregator (previously executed inline via
`python -c '...'` on the Pod) packaged as a standalone script for the release
image. Output mirrors the data published in the FINAL_REPORT (B200×1: 8.08%,
H200×2: 8.49% for bevformer_tiny FP16 AMP).

Usage:
    python /opt/bevformer/scripts/tensor_core_ratio.py <trace.json>

The trace.json file is produced by TorchProfilerHook (see
bevformer/hooks/torch_profiler_hook.py and configs/bevformer/bevformer_tiny_profile.py).
Default output directory: /workspace/runs/profile/torch_profile/
"""
import json
import re
import sys
from collections import defaultdict


# Tensor Core kernel name patterns (cutlass/cudnn/cublas tensorop, etc.).
# Matches kernels that execute on the Tensor Cores across SM75/80/89/90/100/120.
TC_PATTERNS = [
    r'gemm.*tensorop', r'tensorop.*gemm',
    r'sm[0-9]+_xmma',
    r'cutlass.*tensor', r'cudnn.*conv.*tensor',
    r'cublasGemmEx',
    r'wgrad.*tensor', r'dgrad.*tensor',
    r'volta_sgemm', r'volta_hgemm',
    r'ampere_sgemm', r'ampere_hgemm', r'ampere_fp16',
    r'turing_h884',
    r'hopper_gemm', r'blackwell_gemm',
    r'flash.*attn',
    r'tensor_op', r'_tensorop_',
    r'cutlass.*sm[0-9]',
]
TC_RE = re.compile('|'.join(TC_PATTERNS), re.IGNORECASE)


def aggregate(trace_path):
    with open(trace_path) as f:
        data = json.load(f)
    events = [e for e in data.get('traceEvents', []) if e.get('cat') == 'kernel']

    agg = defaultdict(lambda: [0, 0])  # name -> [total_us, count]
    total_us = 0
    tc_us = 0
    for e in events:
        name = e.get('name', '')
        dur = e.get('dur', 0)
        agg[name][0] += dur
        agg[name][1] += 1
        total_us += dur
        if TC_RE.search(name):
            tc_us += dur

    return events, agg, total_us, tc_us


def main(trace_path):
    events, agg, total_us, tc_us = aggregate(trace_path)
    ratio = (tc_us / total_us * 100) if total_us > 0 else 0.0

    print(f'trace: {trace_path}')
    print(f'total_kernel_events={len(events)}')
    print(f'total_cuda_ms={total_us / 1000:.1f}')
    print(f'tc_cuda_ms={tc_us / 1000:.1f}')
    print(f'tc_ratio={ratio:.2f}%')
    print()
    print('=== Top 20 kernels by CUDA time ===')
    sorted_kernels = sorted(agg.items(), key=lambda x: -x[1][0])
    for name, (us, calls) in sorted_kernels[:20]:
        is_tc = 'TC' if TC_RE.search(name) else '  '
        pct = (us / total_us * 100) if total_us > 0 else 0
        print(f'  [{is_tc}] {us / 1000:8.1f}ms ({pct:5.2f}%) calls={calls:6} {name[:90]}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
