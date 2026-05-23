#!/bin/bash
# patches/bevformer/17_test_py_single_gpu.sh
# Strategy C1 patch 17: replace tools/test.py `assert False` (single-GPU 禁止)
# with MMDataParallel + single_gpu_test path so B200 x1 smoke works.
#
# Target: $TARGET/tools/test.py
# Why: Phase 3 で B200 x1 smoke する可能性、`not distributed:` 分岐の
#      `assert False` が単 GPU 実行を blockする。
# Source: design doc STRATEGY_C1_ANALYSIS.md section 5 + Nabe Phase 2 instr.
set -e
TARGET=${1:-/workspace/bevformer}
FILE="$TARGET/tools/test.py"
[ -f "$FILE" ] || { echo "FAIL: $FILE not found"; exit 1; }

echo "=== Before patch 17: relevant lines ==="
grep -n "assert False\|if not distributed:\|single_gpu_test" "$FILE" | head -10

# Backup
cp "$FILE" "${FILE}.pre-c1-17.bak"

# pre-condition: anchor 確認
grep -q "^        assert False$" "$FILE" \
    || { echo "FAIL: 'assert False' anchor not present (already patched?)"; exit 1; }
grep -q "single_gpu_test" "$FILE" \
    || echo "NOTE: 'single_gpu_test' not yet imported (will inject via patch)"

# Apply patch via python (multi-line, safe)
python3 - "$FILE" <<'PYEOF'
import sys
path = sys.argv[1]
src = open(path).read()

# 1. Replace `if not distributed:\n    assert False\n    # ...` block
old_block = '''    if not distributed:
        assert False
        # model = MMDataParallel(model, device_ids=[0])
        # outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:'''
new_block = '''    if not distributed:
        # Strategy C1 patch 17: single-GPU support (was: assert False)
        # See ~/work/bevformer/docs/STRATEGY_C1_ANALYSIS.md section 5.
        # Uses pre-existing `from mmdet3d.apis import single_gpu_test` at top.
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(
            model, data_loader,
            getattr(args, 'show', False),
            getattr(args, 'show_dir', None),
        )
    else:'''
if old_block not in src:
    print(f"FAIL: anchor block not found in {path}", file=sys.stderr)
    sys.exit(1)
src = src.replace(old_block, new_block)

open(path, 'w').write(src)
print('OK: single-GPU branch enabled with MMDataParallel + single_gpu_test')
PYEOF

echo "=== After patch 17: relevant lines ==="
grep -n "Strategy C1 patch 17\|single_gpu_test\|if not distributed:" "$FILE" | head -10

# post-condition: assert False line gone, single_gpu_test usage present
if grep -q "^        assert False$" "$FILE"; then
  # there could be another assert False elsewhere (e.g. line 244 for args.out)
  # check that the FIRST one in single-GPU branch is gone
  N=$(grep -c "^        assert False$" "$FILE")
  echo "NOTE: $N 'assert False' lines remain (acceptable if not in single-GPU branch)"
fi
# post-condition: multi-line call では tail args が改行を跨ぐので、関数 call 開始 +
# 直近 5 行に data_loader 引数が出現するかを確認 (-F で `(` を literal 扱い)
if grep -qF "outputs = single_gpu_test(" "$FILE" \
   && grep -A 5 -F "outputs = single_gpu_test(" "$FILE" | grep -q "data_loader"; then
  echo "OK: single_gpu_test call found (multi-line, data_loader arg verified)"
else
  echo "FAIL: single_gpu_test call not present after patch"
  exit 1
fi
echo "OK: patch 17 (tools/test.py single-GPU enabled) applied"
