#!/bin/bash
# patches/mmdet3d/04_data_to_data_ptr.sh
# Purpose: Tensor::.data<T>() → .data_ptr<T>() に置換
# Target: ops/furthest_point_sample/src/furthest_point_sample.cpp
#         .data<T>() 全 6 件 (post-patch で L34-36, L48-50)
#         - furthest_point_sampling_wrapper の 3 件 (points/temp/idx)
#         - furthest_point_sampling_with_dist_wrapper の 3 件 (points/temp/idx)
#         Note: src/CLAUDE.md §3 では「L50-52」と記述されているが、実際は 2 関数で
#         合計 6 件 (v18 ground truth verify で確定)。/g flag で全件置換する
# Why: PyTorch 2.0+ で Tensor::data<T>() が削除された
set -e
TARGET=${1:-/tmp/mmdetection3d}
FILE="$TARGET/mmdet3d/ops/furthest_point_sample/src/furthest_point_sample.cpp"
[ -f "$FILE" ] || { echo "FAIL: $FILE not found"; exit 1; }
echo "=== Before patch 04: .data< occurrences ==="
grep -n "\.data<" "$FILE" || echo "(none found)"
grep -q "\.data<" "$FILE" || { echo "FAIL: no .data< in $FILE"; exit 1; }
sed -i -E 's/\.data<([^>]+)>\(\)/.data_ptr<\1>()/g' "$FILE"
echo "=== After patch 04: .data_ptr< occurrences ==="
grep -n "\.data_ptr<" "$FILE"
grep -q "\.data<" "$FILE" \
    && { echo "FAIL: .data< still present"; exit 1; } || true
grep -q "\.data_ptr<" "$FILE" \
    || { echo "FAIL: .data_ptr< not present after sed"; exit 1; }
echo "OK: patch 04 (.data → .data_ptr) applied"
