#!/bin/bash
# patches/mmdet3d/02_thc_header_delete.sh
# Purpose: #include <THC/THC.h> 行を削除
# Targets: 既に ATen include を持つ 3 ファイル (src/CLAUDE.md §3 参照)
# Why: PyTorch 2.0+ で THC/THC.h ヘッダが削除された。
#      既存の ATen include で必要な機能はカバーされるので、THC 行は削除のみ
set -e
TARGET=${1:-/tmp/mmdetection3d}
FILES=(
    "mmdet3d/ops/knn/src/knn.cpp"
    "mmdet3d/ops/gather_points/src/gather_points.cpp"
    "mmdet3d/ops/furthest_point_sample/src/furthest_point_sample.cpp"
)
for f in "${FILES[@]}"; do
    full="$TARGET/$f"
    [ -f "$full" ] || { echo "FAIL: $full not found"; exit 1; }
    grep -q "#include <THC/THC.h>" "$full" \
        || { echo "FAIL: THC/THC.h not present in $f"; exit 1; }
    sed -i '/#include <THC\/THC\.h>/d' "$full"
    grep -q "#include <THC/THC.h>" "$full" \
        && { echo "FAIL: THC/THC.h still present in $f"; exit 1; } || true
    echo "OK: $f"
done
echo "OK: patch 02 (THC header delete) applied to ${#FILES[@]} files"
