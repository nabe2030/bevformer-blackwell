#!/bin/bash
# patches/bevformer/12_plugin_init_guard.sh
# Purpose: plugin root __init__.py の `from .dd3d import *` を try/except で guard
# Target: $TARGET/projects/mmdet3d_plugin/__init__.py
# Why: 外部 (plugin root) から dd3d/ にアクセスする唯一の経路。dd3d/__init__.py 自体は
#      patch 09 で empty 化されるが、patch 09 が未適用な build 環境や future change で
#      dd3d/ に再び import が入った場合の防御。patch 10/11 と一貫した defense-in-depth。
# Investigation: plugin-wide grep で確認 (excluding dd3d/ itself):
#   - detectron2/tridet 参照は dd3d/ 内のみ
#   - dd3d への外部 import は nuscenes_dataset_v2.py (patch 10) と dd3d_mapper.py (patch 11) のみ
#   → plugin root で残る dd3d 接点は `from .dd3d import *` の 1 行
set -e
TARGET=${1:-/workspace/bevformer}
FILE="$TARGET/projects/mmdet3d_plugin/__init__.py"
[ -f "$FILE" ] || { echo "FAIL: $FILE not found"; exit 1; }
echo "=== Before patch 12: $FILE ==="
cat "$FILE"; echo "==="
# pre-condition: 行頭 `from .dd3d import *` が存在 (未 patch 状態)
grep -q "^from \.dd3d import \*$" "$FILE" \
    || { echo "FAIL: expected 'from .dd3d import *' line not present (already patched?)"; exit 1; }
python3 - "$FILE" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path) as f:
    src = f.read()
old = "from .dd3d import *"
new = (
    "try:\n"
    "    from .dd3d import *\n"
    "except ImportError as _e:\n"
    "    # Guarded by patches/bevformer/12_plugin_init_guard.sh\n"
    "    # Reason: dd3d/ subpackage may pull Detectron2 (un-installed in v5).\n"
    "    # Normally patch 09 empties dd3d/__init__.py (no-op), this guards against\n"
    "    # patch-application atomicity / future changes adding dd3d imports.\n"
    "    import warnings\n"
    "    warnings.warn(f'dd3d subpackage unavailable: {_e}')"
)
assert src.count(old) == 1, f"Expected exactly 1 occurrence, found {src.count(old)}"
src = src.replace(old, new, 1)
with open(path, "w") as f:
    f.write(src)
PYEOF
echo "=== After patch 12: $FILE ==="
cat "$FILE"; echo "==="
# post-condition
grep -q "^try:$" "$FILE" \
    || { echo "FAIL: 'try:' line not present"; exit 1; }
grep -q "^    from \.dd3d import \*$" "$FILE" \
    || { echo "FAIL: indented 'from .dd3d import *' not present"; exit 1; }
grep -q "^except ImportError" "$FILE" \
    || { echo "FAIL: 'except ImportError' not present"; exit 1; }
echo "OK: patch 12 (plugin __init__.py dd3d star-import guard) applied"
