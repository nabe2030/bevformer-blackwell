#!/usr/bin/env python3
"""patches/mmdet3d/08_visualizer_import_guard.py

Purpose:
    mmdet3d/core/__init__.py の `from .visualizer import ...` 行を
    try/except ImportError で guard。

Why:
    visualizer/show_result.py が `import trimesh` を素で行うため、
    trimesh 未 install 環境では plugin load 時に ImportError になる。
    BEVFormer 推論パスでは visualizer 不要なので silent fallback で防御。

Source:
    旧 Dockerfile.runpod-base / Dockerfile.runpod-base.sourcebuild の
    Step 8b inline heredoc (PATCH8_EOF) を逐語移植。
    pre/post condition と idempotency check を追加。

Usage:
    python /tmp/patches/mmdet3d/08_visualizer_import_guard.py /tmp/mmdetection3d
"""
import pathlib
import sys


def main() -> None:
    if len(sys.argv) < 2:
        print("FAIL: usage: 08_visualizer_import_guard.py <mmdetection3d_root>",
              file=sys.stderr)
        sys.exit(1)
    root = pathlib.Path(sys.argv[1])
    if not root.is_dir():
        print(f"FAIL: not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    p = root / "mmdet3d" / "core" / "__init__.py"
    if not p.is_file():
        print(f"FAIL: not a file: {p}", file=sys.stderr)
        sys.exit(1)

    text = p.read_text()
    lines = text.splitlines(keepends=True)

    # pre-condition: 行頭 `from .visualizer import` が 1 件以上存在
    target_count = sum(
        1 for ln in lines
        if ln.rstrip("\n").startswith("from .visualizer import")
    )
    if target_count == 0:
        print(f"FAIL: no 'from .visualizer import' line found in {p}",
              file=sys.stderr)
        sys.exit(1)

    # idempotency: 既にインデント済 (4 空白) で start すれば patched 済
    indented_count = sum(
        1 for ln in lines
        if ln.rstrip("\n").startswith("    from .visualizer import")
    )
    if indented_count > 0:
        print(f"FAIL: '    from .visualizer import' already present in {p} "
              f"(already patched?)", file=sys.stderr)
        sys.exit(1)

    # apply (verbatim from old Dockerfile Step 8b)
    patched = []
    for line in lines:
        stripped = line.rstrip("\n")
        if stripped.startswith("from .visualizer import"):
            patched.append("try:\n")
            patched.append("    " + line)
            patched.append("except ImportError:\n")
            patched.append("    pass\n")
        else:
            patched.append(line)
    p.write_text("".join(patched))

    # post-condition
    new_text = p.read_text()
    if "try:\n    from .visualizer import" not in new_text:
        print(f"FAIL: post-patch 'try:\\n    from .visualizer import' "
              f"not found in {p}", file=sys.stderr)
        sys.exit(1)
    if "except ImportError:\n    pass" not in new_text:
        print(f"FAIL: post-patch 'except ImportError:\\n    pass' "
              f"not found in {p}", file=sys.stderr)
        sys.exit(1)

    print(f"OK: patch 08 (visualizer import guarded) applied to {p}")


if __name__ == "__main__":
    main()
