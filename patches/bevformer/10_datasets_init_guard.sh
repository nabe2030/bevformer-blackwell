#!/bin/bash
# patches/bevformer/10_datasets_init_guard.sh
# Purpose: datasets/__init__.py の v2 dataset import を try/except で guard
# Target: $TARGET/projects/mmdet3d_plugin/datasets/__init__.py
# Why: CustomNuScenesDatasetV2 は内部で dd3d.datasets.nuscenes を import し、
#      その先で detectron2.structures.boxes を引く。v5 image は Detectron2 未 install。
#      patch 09 は dd3d/__init__.py を empty 化するが、サブパス
#      (dd3d.datasets.nuscenes) は __init__.py を通らないので遮断不可。
#      ここで v2 のみ try/except し、v1 と builder は無傷で残す。
set -e
TARGET=${1:-/workspace/bevformer}
FILE="$TARGET/projects/mmdet3d_plugin/datasets/__init__.py"
[ -f "$FILE" ] || { echo "FAIL: $FILE not found"; exit 1; }
echo "=== Before patch 10: $FILE ==="
cat "$FILE"; echo "==="
# pre-condition: 行頭 v2 import が存在 (未 patch 状態)
grep -q "^from \.nuscenes_dataset_v2 import CustomNuScenesDatasetV2$" "$FILE" \
    || { echo "FAIL: expected v2 import line not present (already patched?)"; exit 1; }
# multi-line edit は python3 で実施 (sed の改行挿入は非可搬)
python3 - "$FILE" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path) as f:
    src = f.read()
old = "from .nuscenes_dataset_v2 import CustomNuScenesDatasetV2"
new = (
    "try:\n"
    "    from .nuscenes_dataset_v2 import CustomNuScenesDatasetV2\n"
    "except ImportError as _e:\n"
    "    # Guarded by patches/bevformer/10_datasets_init_guard.sh\n"
    "    # Reason: v2 pulls dd3d.datasets.nuscenes -> Detectron2 (un-installed in v5)\n"
    "    import warnings\n"
    "    warnings.warn(f'CustomNuScenesDatasetV2 unavailable: {_e}')"
)
assert src.count(old) == 1, f"Expected exactly 1 occurrence of v2 import, found {src.count(old)}"
src = src.replace(old, new, 1)
with open(path, "w") as f:
    f.write(src)
PYEOF
echo "=== After patch 10: $FILE ==="
cat "$FILE"; echo "==="
# post-condition: try / indented import / except の 3 行が揃っていること
grep -q "^try:$" "$FILE" \
    || { echo "FAIL: 'try:' line not present"; exit 1; }
grep -q "^    from \.nuscenes_dataset_v2 import CustomNuScenesDatasetV2$" "$FILE" \
    || { echo "FAIL: indented v2 import not present"; exit 1; }
grep -q "^except ImportError" "$FILE" \
    || { echo "FAIL: 'except ImportError' not present"; exit 1; }
echo "OK: patch 10 (datasets __init__.py v2 import guard) applied"
