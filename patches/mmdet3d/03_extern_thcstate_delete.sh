#!/bin/bash
# patches/mmdet3d/03_extern_thcstate_delete.sh
# Purpose: 'extern THCState *state;' 行を削除 (全 6 ファイル)
# Why: PyTorch 2.0+ で THCState 型が削除された。
#      src/CLAUDE.md §3 のとおり、6 ファイルとも extern 宣言のみで実際の参照は無いので、
#      行ごと削除で安全。
set -e
TARGET=${1:-/tmp/mmdetection3d}
FILES=(
    "mmdet3d/ops/ball_query/src/ball_query.cpp"
    "mmdet3d/ops/knn/src/knn.cpp"
    "mmdet3d/ops/interpolate/src/interpolate.cpp"
    "mmdet3d/ops/group_points/src/group_points.cpp"
    "mmdet3d/ops/gather_points/src/gather_points.cpp"
    "mmdet3d/ops/furthest_point_sample/src/furthest_point_sample.cpp"
)
for f in "${FILES[@]}"; do
    full="$TARGET/$f"
    [ -f "$full" ] || { echo "FAIL: $full not found"; exit 1; }
    grep -q "extern THCState" "$full" \
        || { echo "FAIL: 'extern THCState' not present in $f"; exit 1; }
    sed -i '/extern THCState/d' "$full"
    grep -q "extern THCState" "$full" \
        && { echo "FAIL: 'extern THCState' still present in $f"; exit 1; } || true
    echo "OK: $f"
done
echo "OK: patch 03 (extern THCState delete) applied to ${#FILES[@]} files"
