#!/bin/bash
# patches/mmdet3d/07_spconv_force_register.sh
# Purpose: ops/spconv/conv.py の @CONV_LAYERS.register_module() 全 10 件を
#          @CONV_LAYERS.register_module(force=True) に変更
# Target: $TARGET/mmdet3d/ops/spconv/conv.py
# Why: mmcv 1.5+ で CONV_LAYERS registry が "duplicate registration" に対し KeyError
#      を投げるようになった。mmdet3d 0.17.1 の spconv は SparseConv2d/3d/4d など
#      10 個の class を CONV_LAYERS に register するが、mmcv-full 1.7.2 (v5 使用版) が
#      同名 (一部) を既に持っており、register_module() のデフォルト force=False では
#      実行時に KeyError で落ちる。force=True で上書き許可する。
# Source: upstream v0.17.1 vs v18 ground truth の diff で 10 行同一パターンと確定
#         (対象 class: SparseConv2d/3d/4d, SparseConvTranspose2d/3d,
#          SparseInverseConv2d/3d, SubMConv2d/3d/4d)
set -e
TARGET=${1:-/tmp/mmdetection3d}
FILE="$TARGET/mmdet3d/ops/spconv/conv.py"
[ -f "$FILE" ] || { echo "FAIL: $FILE not found"; exit 1; }
echo "=== Before patch 07: register_module occurrences ==="
grep -c "^@CONV_LAYERS\.register_module()$" "$FILE" \
    | awk '{print $1 " (no-force form)"}'
grep -c "^@CONV_LAYERS\.register_module(force=True)$" "$FILE" \
    | awk '{print $1 " (force=True form)"}'
# pre-condition: 期待値ちょうど 10 件の no-force form
N_PRE=$(grep -c "^@CONV_LAYERS\.register_module()$" "$FILE" || true)
[ "$N_PRE" = "10" ] \
    || { echo "FAIL: expected 10 no-force @CONV_LAYERS.register_module() lines, found $N_PRE"; exit 1; }
sed -i 's|^@CONV_LAYERS\.register_module()$|@CONV_LAYERS.register_module(force=True)|' "$FILE"
echo "=== After patch 07: register_module occurrences ==="
grep -c "^@CONV_LAYERS\.register_module()$" "$FILE" \
    | awk '{print $1 " (no-force form)"}'
grep -c "^@CONV_LAYERS\.register_module(force=True)$" "$FILE" \
    | awk '{print $1 " (force=True form)"}'
# post-condition: 0 件 no-force / 10 件 force=True
N_POST_NO=$(grep -c "^@CONV_LAYERS\.register_module()$" "$FILE" || true)
N_POST_F=$(grep -c "^@CONV_LAYERS\.register_module(force=True)$" "$FILE" || true)
[ "$N_POST_NO" = "0" ] \
    || { echo "FAIL: no-force form still remains ($N_POST_NO)"; exit 1; }
[ "$N_POST_F" = "10" ] \
    || { echo "FAIL: expected 10 force=True lines, got $N_POST_F"; exit 1; }
echo "OK: patch 07 (spconv force register, 10 sites) applied"
