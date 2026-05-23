#!/bin/bash
# patches/misc/nuscenes_dict_keys.sh
# Purpose: nuscenes-devkit data_classes.py の dict_keys 直接代入を list() で包む
# Target: <site-packages>/nuscenes/eval/detection/data_classes.py
#         (Python で動的解決 — runpod/pytorch base + Python 3.11 で path 変動)
# Why: GitHub Issue #1155 - DetectionConfig.__init__ で self.class_names に dict_keys が
#      代入され、multiproc DataLoader (num_workers > 0) の pickle で
#      `TypeError: cannot pickle 'dict_keys'` を引き起こす。
# History: v18 (Python 3.8, miniconda) で同じ sed を実証
#          (Dockerfile.runpod-base.sourcebuild Stage 11.5、v18f で 1 epoch DDP 完走、
#           loss 17.81→9.68)
# Note: Dockerfile Stage 9.5 から TARGET arg なしで呼ばれる前提
#       (他 patch と異なり site-packages 配下を編集するため)
set -e
# nuscenes-devkit install path を python で動的解決
TARGET=$(python -c "import nuscenes.eval.detection.data_classes as m; print(m.__file__)" 2>&1)
[ -f "$TARGET" ] \
    || { echo "FAIL: nuscenes data_classes.py not found (got '$TARGET')"; exit 1; }
echo "=== Target resolved: $TARGET ==="
echo "=== Before patch: target line ==="
grep -n "class_names = self\.class_range" "$TARGET" \
    || { echo "FAIL: 'class_names = self.class_range' not present (already patched?)"; exit 1; }
grep -q "self\.class_names = self\.class_range\.keys()$" "$TARGET" \
    || { echo "FAIL: expected unwrapped 'self.class_range.keys()' not present"; exit 1; }
sed -i 's|self\.class_names = self\.class_range\.keys()|self.class_names = list(self.class_range.keys())|' "$TARGET"
echo "=== After patch: target line ==="
# NB: post-state grep — pattern targets `class_names = list(self.class_range`
# (not the pre-state `class_names = self.class_range`). Pre-pattern fails to match
# post-state because `=` is followed by `list(` instead of `self.` after sed wrap.
# Active verification under `set -e`: sed silent fail → no match → exit 1 (alert).
grep -n "class_names = list(self\.class_range" "$TARGET"
grep -q "self\.class_names = list(self\.class_range\.keys())" "$TARGET" \
    || { echo "FAIL: post-patch wrapped form not present"; exit 1; }
grep -q "self\.class_names = self\.class_range\.keys()$" "$TARGET" \
    && { echo "FAIL: unwrapped form still present"; exit 1; } || true
echo "OK: patch nuscenes_dict_keys applied to $TARGET"
