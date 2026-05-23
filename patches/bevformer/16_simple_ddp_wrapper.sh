#!/bin/bash
# patches/bevformer/16_simple_ddp_wrapper.sh
# Strategy C1 patch 16: replace mmcv 1.7.2 MMDistributedDataParallel with
# SimpleDDP v3 equivalent compatible with PyTorch 2.x.
#
# Design history (Phase 3 で確定):
#   v1 "swap forward"     — _to_kwargs が dict positional を unpack で fail
#   v2 "bypass forward"   — DataContainer unwrap 欠落で AttributeError
#   v3 scatter_kwargs+direct call — Phase 3-5 で実証済の最終版
#
# Target: <site-packages>/mmcv/parallel/distributed.py (replace verbatim)
# Why: open-mmlab/mmcv#636 — reducer direct access broke in PT 2.x.
#      See ~/work/bevformer/docs/STRATEGY_C1_ANALYSIS.md for design.

set -e
TARGET=$(python3 -c "import mmcv.parallel.distributed as m; print(m.__file__)" 2>/dev/null)
[ -f "$TARGET" ] || { echo "FAIL: mmcv distributed.py not found via python import"; exit 1; }
echo "=== patch 16: replacing $TARGET ==="

# Backup
cp "$TARGET" "${TARGET}.pre-c1.bak"

cat > "$TARGET" <<'PYEOF'
"""SimpleDDP v3 — Strategy C1 (Phase 3 で確定).

Drop-in replacement for mmcv 1.7.2 MMDistributedDataParallel that works on
PyTorch 2.x without touching any private PyTorch reducer API (_rebuild_buckets,
_sync_params, _module_copies, _DDPSink — all removed or refactored between
PT 2.0 and 2.8).

Design history:
- v1 "swap forward" routed through stock DDP.forward() but _pre_forward's
  _to_kwargs unpacked dict positional args into kwargs → TypeError.
- v2 "bypass forward" called module.train_step() directly but skipped
  DataContainer unwrap → 'DataContainer' is not subscriptable.
- v3 (current): mmcv scatter_kwargs unwraps DataContainer, then directly
  calls self.module.train_step()/val_step(). Gradient sync still works via
  parameter hooks registered in DDP __init__.

Compatible with PyTorch 2.0-2.12+. Verified in Phase 3-5:
- B200×2 Run C 3-epoch loss 23.26 → 16.34 (no NaN)
- H200×2 Run A/B loss curve matches B200, grad_norm ~42 stable
"""
# Copyright (c) OpenMMLab. All rights reserved.

# Preserve _find_tensors export for downstream re-import (mmcv internal)
from torch.nn.parallel.distributed import (  # noqa: F401
    DistributedDataParallel,
    _find_tensors,
)

from .scatter_gather import scatter_kwargs


class MMDistributedDataParallel(DistributedDataParallel):
    """mmcv 1.7.2 MMDistributedDataParallel — PyTorch 2.x-compatible reimpl (v3).

    Public surface preserved:
    - ``train_step(*inputs, **kwargs)`` → ``self.module.train_step(...)``
    - ``val_step(*inputs, **kwargs)`` → ``self.module.val_step(...)``

    DataContainer is unwrapped via mmcv's own scatter_kwargs (which knows
    about DataContainer); then we call ``self.module.<method>`` directly.
    Gradient sync still works through parameter hooks set in DDP __init__.
    """

    def train_step(self, *inputs, **kwargs):
        if self.device_ids:
            inputs, kwargs = scatter_kwargs(inputs, kwargs, self.device_ids, dim=self.dim)
            return self.module.train_step(*inputs[0], **kwargs[0])
        return self.module.train_step(*inputs, **kwargs)

    def val_step(self, *inputs, **kwargs):
        if self.device_ids:
            inputs, kwargs = scatter_kwargs(inputs, kwargs, self.device_ids, dim=self.dim)
            return self.module.val_step(*inputs[0], **kwargs[0])
        return self.module.val_step(*inputs, **kwargs)
PYEOF

echo "=== verify ==="
python3 -c "
from mmcv.parallel import MMDistributedDataParallel
import inspect
src = inspect.getsource(MMDistributedDataParallel)
assert 'scatter_kwargs(inputs, kwargs' in src, 'FAIL: SimpleDDP v3 marker not found'
assert 'train_step' in src and 'val_step' in src, 'FAIL: train_step/val_step missing'
bases = MMDistributedDataParallel.__bases__
assert any('DistributedDataParallel' in b.__name__ for b in bases), \
    f'FAIL: base classes incorrect: {bases}'
print('OK: MMDistributedDataParallel patched to SimpleDDP v3')
print(f'  bases: {[b.__name__ for b in bases]}')
print(f'  has train_step: {hasattr(MMDistributedDataParallel, \"train_step\")}')
print(f'  has val_step:   {hasattr(MMDistributedDataParallel, \"val_step\")}')
print(f'  source lines: {len(src.splitlines())}')
"
echo "OK: patch 16 (mmcv SimpleDDP v3 wrapper) applied"
