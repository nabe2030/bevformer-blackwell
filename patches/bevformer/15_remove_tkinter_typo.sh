#!/bin/bash
# patches/bevformer/15_remove_tkinter_typo.sh
# Purpose: bevformer_fp16.py:7 の誤 import `from tkinter.messagebox import NO` を削除
# Target: $TARGET/projects/mmdet3d_plugin/bevformer/detectors/bevformer_fp16.py
# Why: 明らかな IDE 自動 import の事故。`NO` シンボルはファイル内で全く使われていない
#      (grep '\bNO\b' で line 7 のみ確認)。tkinter は BEVFormer の依存に含まれず、
#      headless container でも tkinter 自体が無くて ImportError になる可能性。不要 line。
# Note: source file has CRLF line terminators (upstream BEVFormer の Windows 由来)。
#       GNU grep / sed BRE では `\r` は literal `r` (CR ではない)。literal CR byte を
#       printf '\r' で注入する必要あり。ugrep は `\r` を CR 扱いするが container は GNU grep。
# Investigation: src/ 全体 grep で tkinter 参照は本ファイル 1 行のみ
set -e
TARGET=${1:-/workspace/bevformer}
FILE="$TARGET/projects/mmdet3d_plugin/bevformer/detectors/bevformer_fp16.py"
[ -f "$FILE" ] || { echo "FAIL: $FILE not found"; exit 1; }
echo "=== Before patch 15: tkinter line ==="
grep -n "tkinter" "$FILE" \
    || { echo "FAIL: 'tkinter' not present in $FILE"; exit 1; }
CR=$(printf '\r')
grep -q "^from tkinter\.messagebox import NO${CR}\?$" "$FILE" \
    || { echo "FAIL: expected exact line 'from tkinter.messagebox import NO' not present"; exit 1; }
sed -i "/^from tkinter\.messagebox import NO${CR}\?\$/d" "$FILE"
echo "=== After patch 15: tkinter line ==="
grep -n "tkinter" "$FILE" \
    && { echo "FAIL: tkinter still present in $FILE"; exit 1; } \
    || echo "(none) — OK"
echo "OK: patch 15 (tkinter typo removed) applied"
