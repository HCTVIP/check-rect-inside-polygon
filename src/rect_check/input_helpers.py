"""Nhập tọa độ polygon và hình chữ nhật từ bàn phím."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import List, Optional, Tuple

from .batch import ObjectInput, object_from_yolo_xyxy, parse_objects
from .geometry import looks_like_yolo_xyxy
from .geometry import parse_polygon

Point = Tuple[float, float]


def parse_point(text: str) -> Optional[Point]:
    """
    Parse một điểm: "100,100" | "100 100" | "(100, 100)".
    Trả về None nếu chuỗi rỗng.
    """
    text = text.strip()
    if not text:
        return None
    text = text.strip("()[]")
    if "," in text:
        parts = [p.strip() for p in text.split(",", 1)]
    else:
        parts = re.split(r"\s+", text, maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"Không đọc được điểm: {text!r} (dùng x,y hoặc x y)")
    return float(parts[0]), float(parts[1])


def input_polygon_interactive() -> List[Point]:
    """
    Nhập từng đỉnh đa giác.

    Cách 1: từng dòng x,y (dòng trống để kết thúc)
    Cách 2: một dòng nhiều điểm: 100,100 500,120 ...
    """
    print("\n--- Vùng (polygon) ---")
    print("Nhập đỉnh đa giác (ít nhất 3 điểm).")
    print("  • Từng dòng: x,y  (Enter trống để xong)")
    print('  • Hoặc một dòng: 100,100 500,120 600,300 ...')
    print('  • Hoặc JSON: [[100,100],[500,120],...]')

    first = input("> ").strip()
    if not first:
        raise ValueError("Chưa nhập polygon.")

    if first.startswith("["):
        return parse_polygon(first)

    if " " in first and "," in first.split()[0]:
        return parse_polygon(first)

    points: List[Point] = []
    line = first
    while True:
        pt = parse_point(line)
        if pt is None:
            break
        points.append(pt)
        line = input("> ").strip()
        if not line:
            break

    if len(points) < 3:
        raise ValueError(f"Polygon cần ≥ 3 đỉnh, hiện có {len(points)}")
    return points


def input_rectangle_interactive(*, quiet: bool = False) -> Tuple[float, float, float, float]:
    """Nhập tâm và kích thước hình chữ nhật."""
    if not quiet:
        print("\n--- Vật thể (hình chữ nhật) ---")
        print("Nhập tâm (x_center, y_center) và kích thước (w, h).")

    while True:
        raw = input("x_center, y_center (hoặc x y): ").strip()
        try:
            pt = parse_point(raw)
            if pt is None:
                raise ValueError("empty")
            x_center, y_center = pt
            break
        except (ValueError, TypeError):
            print("  Lỗi: nhập ví dụ 210,275 hoặc 210 275")

    while True:
        raw = input("w, h (chiều rộng, chiều cao): ").strip()
        try:
            pt = parse_point(raw)
            if pt is None:
                raise ValueError("empty")
            w, h = pt
            if w <= 0 or h <= 0:
                raise ValueError("size")
            break
        except (ValueError, TypeError):
            print("  Lỗi: nhập ví dụ 180,450 — w và h phải > 0")

    return float(x_center), float(y_center), float(w), float(h)


def _parse_object_line(line: str, index: int) -> ObjectInput:
    """
    Một dòng:
      - yolo:x1,y1,x2,y2  (YOLOv8)
      - center:x_center,y_center,w,h
      - x1,y1,x2,y2  (tự nhận nếu x2>x1 và y2>y1)
      - x_center,y_center,w,h
      - id,x1,y1,x2,y2 hoặc id,x_center,y_center,w,h
    """
    line = line.strip()
    if not line or line.startswith("#"):
        raise ValueError("empty")

    fmt: str | None = None
    lower = line.lower()
    for prefix in ("yolo:", "xyxy:", "center:", "c:"):
        if lower.startswith(prefix):
            fmt = "yolo" if prefix in ("yolo:", "xyxy:") else "center"
            line = line[len(prefix) :].strip()
            break

    parts = [p.strip() for p in re.split(r"[,;\s]+", line) if p.strip()]
    if len(parts) == 4:
        a, b, c, d = (float(v) for v in parts)
        if fmt == "yolo" or (fmt is None and looks_like_yolo_xyxy(a, b, c, d)):
            return object_from_yolo_xyxy(a, b, c, d)
        return ObjectInput(x_center=a, y_center=b, w=c, h=d)
    if len(parts) == 5:
        oid = parts[0]
        a, b, c, d = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        if fmt == "yolo" or (fmt is None and looks_like_yolo_xyxy(a, b, c, d)):
            return object_from_yolo_xyxy(a, b, c, d, id=oid)
        return ObjectInput(id=oid, x_center=a, y_center=b, w=c, h=d)
    raise ValueError(
        f"dòng #{index}: dùng x1,y1,x2,y2 (YOLO), yolo:..., "
        "x_center,y_center,w,h, hoặc center:..."
    )


def _load_objects_file(path: str) -> List[ObjectInput]:
    p = Path(path.strip().lstrip("@"))
    if not p.is_file():
        raise ValueError(f"Không tìm thấy file: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "objects" in data:
        return parse_objects(data)
    return parse_objects(data)


def _generate_fake_objects(count: int, seed: int = 42) -> List[ObjectInput]:
    rng = random.Random(seed)
    return [
        ObjectInput(
            id=f"fake-{i:04d}",
            x_center=rng.uniform(50, 550),
            y_center=rng.uniform(50, 450),
            w=rng.uniform(10, 80),
            h=rng.uniform(10, 120),
        )
        for i in range(count)
    ]


def input_objects_interactive() -> List[ObjectInput]:
    """
    Nhập một hoặc nhiều vật thể (có thể 1000+).

    - Enter trống ở dòng đầu → 1 vật (tâm + w,h như cũ)
    - Từng dòng: x_center,y_center,w,h  (Enter trống để xong)
    - JSON một dòng hoặc file @path.json
    - !random 1000  → tạo 1000 vật thể giả
    """
    print("\n--- Vật thể (hình chữ nhật) ---")
    print("Nhập một hoặc nhiều vật thể:")
    print("  • Enter trống → 1 vật (tâm + w, h)")
    print("  • YOLOv8: x1,y1,x2,y2  hoặc yolo:x1,y1,x2,y2")
    print("  • Tâm+kích thước: x_center,y_center,w,h  hoặc center:...")
    print('  • JSON YOLO: [{"x1":10,"y1":20,"x2":100,"y2":200}, ...]')
    print('  • JSON tâm: [{"x_center":210,"y_center":275,"w":180,"h":450}]')
    print("  • File: @duong/dan/file.json")
    print("  • Giả lập: !random 1000")

    first = input("> ").strip()

    if not first:
        xc, yc, w, h = input_rectangle_interactive(quiet=True)
        return [ObjectInput(x_center=xc, y_center=yc, w=w, h=h)]

    if first.lower().startswith("!random"):
        parts = first.split()
        if len(parts) != 2:
            raise ValueError("Dùng: !random 1000")
        count = int(parts[1])
        if count <= 0:
            raise ValueError("Số vật thể phải > 0")
        seed_raw = input("seed [42]: ").strip() or "42"
        objs = _generate_fake_objects(count, int(seed_raw))
        print(f"  Đã tạo {len(objs)} vật thể giả (seed={seed_raw}).")
        return objs

    if first.startswith("@"):
        objs = _load_objects_file(first)
        print(f"  Đọc {len(objs)} vật thể từ file.")
        return objs

    if first.startswith("[") or first.startswith("{"):
        return parse_objects(json.loads(first))

    objects: List[ObjectInput] = []
    line = first
    line_no = 1
    while True:
        try:
            obj = _parse_object_line(line, line_no)
            objects.append(obj)
        except ValueError as e:
            if str(e) == "empty":
                break
            raise
        line_no += 1
        line = input("> ").strip()
        if not line:
            break

    if not objects:
        raise ValueError("Chưa nhập vật thể nào.")
    print(f"  Đã nhập {len(objects)} vật thể.")
    return objects
