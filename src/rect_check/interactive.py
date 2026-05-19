"""Chạy nhập tọa độ tương tác: python -m rect_check.interactive"""

from __future__ import annotations

import sys
import time

from .batch import batch_summary, evaluate_batch
from .input_helpers import input_objects_interactive, input_polygon_interactive
from .reporting import print_batch_report


def run_interactive() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 50)
    print("  Kiểm tra hình chữ nhật trong đa giác (OpenCV)")
    print("=" * 50)

    try:
        polygon = input_polygon_interactive()
        objects = input_objects_interactive()
    except (ValueError, KeyboardInterrupt) as e:
        if isinstance(e, KeyboardInterrupt):
            print("\nĐã hủy.")
        else:
            print(f"\nLỗi: {e}")
        return 1

    t0 = time.perf_counter()
    results = evaluate_batch(polygon, objects)
    elapsed = time.perf_counter() - t0
    summary = batch_summary(results)

    print("\n" + "=" * 50)
    print("  Kết quả")
    print("=" * 50)
    if len(objects) > 1:
        print(f"Thời gian xử lý: {elapsed * 1000:.1f} ms")
    print_batch_report(
        polygon,
        results,
        summary,
        show_polygon=True,
        preview=10 if len(objects) > 1 else 1,
    )

    again = input("\nNhập lại? (y/N): ").strip().lower()
    if again in ("y", "yes", "có"):
        return run_interactive()
    return 0


def main() -> int:
    return run_interactive()


if __name__ == "__main__":
    sys.exit(main())
