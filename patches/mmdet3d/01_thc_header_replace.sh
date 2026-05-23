#!/bin/bash
# patches/mmdet3d/01_thc_header_replace.sh
# Purpose: #include <THC/THC.h> を #include <ATen/cuda/CUDAContext.h> に置換
# Targets: ATen include を持たない 3 ファイル (src/CLAUDE.md §3 参照)
# Why: PyTorch 2.0+ で THC/THC.h ヘッダが削除された。
#      ATen include を持たないファイルは「置換」、持つファイルは「削除」(→ 02)
set -e
TARGET=${1:-/tmp/mmdetection3d}
FILES=(
    "mmdet3d/ops/ball_query/src/ball_query.cpp"
    "mmdet3d/ops/interpolate/src/interpolate.cpp"
    "mmdet3d/ops/group_points/src/group_points.cpp"
)
for f in "${FILES[@]}"; do
    full="$TARGET/$f"
    [ -f "$full" ] || { echo "FAIL: $full not found"; exit 1; }
    grep -q "#include <THC/THC.h>" "$full" \
        || { echo "FAIL: THC/THC.h not present in $f"; exit 1; }
    sed -i 's|#include <THC/THC\.h>|#include <ATen/cuda/CUDAContext.h>|' "$full"
    grep -q "#include <ATen/cuda/CUDAContext.h>" "$full" \
        || { echo "FAIL: replacement not applied in $f"; exit 1; }
    grep -q "#include <THC/THC.h>" "$full" \
        && { echo "FAIL: THC/THC.h still present in $f"; exit 1; } || true
    echo "OK: $f"
done
echo "OK: patch 01 (THC header replace) applied to ${#FILES[@]} files"
