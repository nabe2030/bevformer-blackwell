#!/bin/bash
# patches/bevformer/09_dd3d_init_empty.sh
# Purpose: projects/mmdet3d_plugin/dd3d/__init__.py を empty 化 (Detectron2 依存の遮断)
# Target: $TARGET/projects/mmdet3d_plugin/dd3d/__init__.py
# Why: 元 __init__.py は `from .modeling import *` を実行 → dd3d/modeling 配下の
#      Detectron2 依存コードを引き込む。BEVFormer 推論パスでは dd3d 不要、
#      かつ v5 では Detectron2 を install していないので、plugin load 時に
#      ImportError になる。empty 化して「import 可能だが何も export しない」状態に。
set -e
TARGET=${1:-/workspace/bevformer}
FILE="$TARGET/projects/mmdet3d_plugin/dd3d/__init__.py"
[ -f "$FILE" ] || { echo "FAIL: $FILE not found"; exit 1; }
echo "=== Before patch 09: $FILE ==="
cat "$FILE"; echo
# pre-condition: 元 __init__.py に modeling import が含まれていること
grep -q "^from \.modeling import" "$FILE" \
    || { echo "FAIL: expected 'from .modeling import' not present"; exit 1; }
# empty 化 (explanatory comment 付き、Python 的には empty と同等)
cat > "$FILE" <<'EOF'
# Emptied by patches/bevformer/09_dd3d_init_empty.sh
# Reason: dd3d/modeling は Detectron2 依存。BEVFormer 推論パスでは dd3d 不要、
#         かつ v5 image は Detectron2 未 install。plugin load 時の ImportError 回避。
EOF
echo "=== After patch 09: $FILE ==="
cat "$FILE"
# post-condition: modeling import が無いこと
grep -q "^from \.modeling import" "$FILE" \
    && { echo "FAIL: 'from .modeling import' still present"; exit 1; } || true
echo "OK: patch 09 (dd3d __init__.py emptied) applied"
