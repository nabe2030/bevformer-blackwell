#!/bin/bash
# patches/mmdet3d/13_spconv_cxx17.sh
# Purpose: spconv の '-std=c++14' を '-std=c++17' に変更
# Why: PyTorch 2.7.0+ ヘッダが C++17 必須
#      (mmdet3d 0.17.1 setup.py:244 が唯一の c++14 hardcode)
# Source: docs/mmdet3d_0.17.1_analysis.md §2 で grep "std=c++" → setup.py:244 1 件のみと確定
set -e
TARGET=${1:-/tmp/mmdetection3d}
SETUP_PY="$TARGET/setup.py"
[ -f "$SETUP_PY" ] || { echo "FAIL: $SETUP_PY not found"; exit 1; }
echo "=== Before patch 13: c++14 occurrences in setup.py ==="
grep -n "c++14" "$SETUP_PY" || echo "(none found)"
sed -i "s/'-std=c++14'/'-std=c++17'/g" "$SETUP_PY"
echo "=== After patch 13: c++17 occurrences in setup.py ==="
grep -n "c++17" "$SETUP_PY"
grep -q "'-std=c++14'" "$SETUP_PY" && { echo "FAIL: c++14 remains"; exit 1; } || true
grep -q "'-std=c++17'" "$SETUP_PY" || { echo "FAIL: c++17 not applied"; exit 1; }
echo "OK: patch 13 (spconv c++17) applied successfully"
