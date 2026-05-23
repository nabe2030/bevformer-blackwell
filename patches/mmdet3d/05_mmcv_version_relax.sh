#!/bin/bash
# patches/mmdet3d/05_mmcv_version_relax.sh
# Purpose: mmdet3d/__init__.py の mmcv バージョン上限を '1.4.0' → '1.8.0' に緩和
# Why: mmcv-full 1.7.2 (PyTorch 2.x 対応の mmcv 1.x 系最終版) を使うため。
#      src/CLAUDE.md §2 で確認済の唯一の version-check 緩和。
#      v18 ground truth で post-patch 形を verify 済 (mmcv_maximum_version = '1.8.0')。
set -e
TARGET=${1:-/tmp/mmdetection3d}
FILE="$TARGET/mmdet3d/__init__.py"
[ -f "$FILE" ] || { echo "FAIL: $FILE not found"; exit 1; }
echo "=== Before patch 05: mmcv_maximum_version line ==="
grep -n "mmcv_maximum_version" "$FILE" \
    || { echo "FAIL: 'mmcv_maximum_version' not found in $FILE"; exit 1; }
grep -q "mmcv_maximum_version = '1\.4\.0'" "$FILE" \
    || { echo "FAIL: expected mmcv_maximum_version = '1.4.0' not present"; exit 1; }
sed -i "s/mmcv_maximum_version = '1\.4\.0'/mmcv_maximum_version = '1.8.0'/" "$FILE"
echo "=== After patch 05: mmcv_maximum_version line ==="
grep -n "mmcv_maximum_version" "$FILE"
grep -q "mmcv_maximum_version = '1\.8\.0'" "$FILE" \
    || { echo "FAIL: mmcv_maximum_version not updated to '1.8.0'"; exit 1; }
echo "OK: patch 05 (mmcv version relax 1.4.0 → 1.8.0) applied"
