#!/bin/bash
# patches/mmdet3d/06_numba_errors_path.sh
# Purpose: from numba.errors import → from numba.core.errors import に変更
# Target: mmdet3d/datasets/pipelines/data_augment_utils.py (L5、唯一の該当行)
# Why: numba 0.50 (2020) で `numba.errors` namespace が `numba.core.errors` に移動。
#      旧 path は shim として残っていたが、後の version で removed。
#      v5 では PIP_CONSTRAINT で numba==0.58.1 に固定しており、0.58.1 は旧 path で
#      ImportError (deprecation でなく) になるため必須。
# Source: upstream v0.17.1 vs v18 ground truth の diff で対象 1 行と確定
#         (mmdet3d/ 配下に numba.errors / numba.core.errors の occurrence は各 1 件のみ)
set -e
TARGET=${1:-/tmp/mmdetection3d}
FILE="$TARGET/mmdet3d/datasets/pipelines/data_augment_utils.py"
[ -f "$FILE" ] || { echo "FAIL: $FILE not found"; exit 1; }
echo "=== Before patch 06: target line ==="
grep -n "numba\.errors\|numba\.core\.errors" "$FILE" \
    || { echo "FAIL: no numba.errors/numba.core.errors in $FILE"; exit 1; }
grep -q "^from numba\.errors import NumbaPerformanceWarning$" "$FILE" \
    || { echo "FAIL: expected pre-patch line not present"; exit 1; }
sed -i 's|^from numba\.errors import NumbaPerformanceWarning$|from numba.core.errors import NumbaPerformanceWarning|' "$FILE"
echo "=== After patch 06: target line ==="
grep -n "numba\.core\.errors" "$FILE"
grep -q "^from numba\.errors import" "$FILE" \
    && { echo "FAIL: old numba.errors path still present"; exit 1; } || true
grep -q "^from numba\.core\.errors import NumbaPerformanceWarning$" "$FILE" \
    || { echo "FAIL: post-patch line not present"; exit 1; }
echo "OK: patch 06 (numba.errors → numba.core.errors) applied"
