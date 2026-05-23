#!/bin/bash
# patches/bevformer/11_pipelines_init_guard.sh
# Purpose: pipelines/__init__.py の DD3DMapper import を try/except で guard
# Target: $TARGET/projects/mmdet3d_plugin/datasets/pipelines/__init__.py
# Why: DD3DMapper は dd3d.datasets.transform_utils / dd3d.structures.pose /
#      dd3d.utils.tasks を引き、その先で Detectron2 を必須とする。
#      v5 image は Detectron2 未 install。Plugin load 時の ImportError 回避。
# Investigation: dd3d_mapper.py の L6-L8 が dd3d サブパス 3 件を import。
#      他 3 import (transform_3d / formating / augmentation) は mmcv/mmdet/numpy のみで
#      Detectron2 非依存 → guard 不要。
set -e
TARGET=${1:-/workspace/bevformer}
FILE="$TARGET/projects/mmdet3d_plugin/datasets/pipelines/__init__.py"
[ -f "$FILE" ] || { echo "FAIL: $FILE not found"; exit 1; }
echo "=== Before patch 11: $FILE ==="
cat "$FILE"; echo "==="
# pre-condition: 行頭 dd3d_mapper import が存在 (未 patch 状態)
grep -q "^from \.dd3d_mapper import DD3DMapper$" "$FILE" \
    || { echo "FAIL: expected dd3d_mapper import line not present (already patched?)"; exit 1; }
# multi-line edit は python3 で実施
python3 - "$FILE" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path) as f:
    src = f.read()
old = "from .dd3d_mapper import DD3DMapper"
new = (
    "try:\n"
    "    from .dd3d_mapper import DD3DMapper\n"
    "except ImportError as _e:\n"
    "    # Guarded by patches/bevformer/11_pipelines_init_guard.sh\n"
    "    # Reason: DD3DMapper pulls dd3d.{datasets,structures,utils} -> Detectron2 (un-installed in v5)\n"
    "    import warnings\n"
    "    warnings.warn(f'DD3DMapper unavailable: {_e}')"
)
assert src.count(old) == 1, f"Expected exactly 1 occurrence of dd3d_mapper import, found {src.count(old)}"
src = src.replace(old, new, 1)
with open(path, "w") as f:
    f.write(src)
PYEOF
echo "=== After patch 11: $FILE ==="
cat "$FILE"; echo "==="
# post-condition: try / indented import / except の 3 行が揃っていること
grep -q "^try:$" "$FILE" \
    || { echo "FAIL: 'try:' line not present"; exit 1; }
grep -q "^    from \.dd3d_mapper import DD3DMapper$" "$FILE" \
    || { echo "FAIL: indented dd3d_mapper import not present"; exit 1; }
grep -q "^except ImportError" "$FILE" \
    || { echo "FAIL: 'except ImportError' not present"; exit 1; }
echo "OK: patch 11 (pipelines __init__.py DD3DMapper guard) applied"
