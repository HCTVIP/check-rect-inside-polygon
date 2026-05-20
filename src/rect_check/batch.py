"""Kiểm tra hàng loạt nhiều vật thể với một polygon."""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple, TypedDict

import cv2
import numpy as np

from .geometry import (
    Polygon,
    RectanglePolygonResult,
    classify_polygon_relation,
    looks_like_yolo_xyxy,
    polygon_to_contour,
    yolo_xyxy_to_center_size,
)
from .reporting import POLYGON_RELATION_MESSAGES

Point = Tuple[float, float]


class ObjectInput(TypedDict, total=False):
    id: str
    x_center: float
    y_center: float
    w: float
    h: float
    x1: float
    y1: float
    x2: float
    y2: float


class ObjectResult(TypedDict, total=False):
    index: int
    id: str
    x_center: float
    y_center: float
    w: float
    h: float
    x1: float
    y1: float
    x2: float
    y2: float
    fully_inside: bool
    overlaps: bool
    relation: str
    relation_message: str
    error: str


def object_from_yolo_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    id: str = "",
) -> ObjectInput:
    """Tạo ObjectInput từ bbox YOLOv8 (x1,y1,x2,y2)."""
    xc, yc, w, h = yolo_xyxy_to_center_size(x1, y1, x2, y2)
    x1f, y1f, x2f, y2f = float(x1), float(y1), float(x2), float(y2)
    if x2f < x1f:
        x1f, x2f = x2f, x1f
    if y2f < y1f:
        y1f, y2f = y2f, y1f
    row: ObjectInput = {
        "x_center": xc,
        "y_center": yc,
        "w": w,
        "h": h,
        "x1": x1f,
        "y1": y1f,
        "x2": x2f,
        "y2": y2f,
    }
    if id:
        row["id"] = id
    return row


def _parse_object(raw: Any, index: int) -> ObjectInput:
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        a, b, c, d = (float(v) for v in raw)
        if looks_like_yolo_xyxy(a, b, c, d):
            return object_from_yolo_xyxy(a, b, c, d)
        return ObjectInput(x_center=a, y_center=b, w=c, h=d)

    if not isinstance(raw, dict):
        raise ValueError(
            f"Vật thể #{index}: cần object, [x1,y1,x2,y2] YOLO, hoặc [x_center,y_center,w,h]"
        )

    obj = dict(raw)
    if "id" in obj:
        oid = str(obj["id"])
    else:
        oid = ""

    if all(k in obj for k in ("x1", "y1", "x2", "y2")):
        return object_from_yolo_xyxy(
            obj["x1"], obj["y1"], obj["x2"], obj["y2"], id=oid
        )

    if "xyxy" in obj and len(obj["xyxy"]) == 4:
        x1, y1, x2, y2 = obj["xyxy"]
        return object_from_yolo_xyxy(x1, y1, x2, y2, id=oid)

    if "center" in obj and "size" in obj:
        cx, cy = obj["center"]
        w, h = obj["size"]
        obj.setdefault("x_center", cx)
        obj.setdefault("y_center", cy)
        obj.setdefault("w", w)
        obj.setdefault("h", h)

    for key in ("x_center", "y_center", "w", "h"):
        if key not in obj:
            raise ValueError(
                f"Vật thể #{index}: thiếu {key} (hoặc dùng x1,y1,x2,y2 cho YOLO)"
            )
    row = ObjectInput(
        id=oid,
        x_center=float(obj["x_center"]),
        y_center=float(obj["y_center"]),
        w=float(obj["w"]),
        h=float(obj["h"]),
    )
    return row


def parse_objects(data: Any) -> List[ObjectInput]:
    """Parse danh sách vật thể từ JSON (mảng hoặc {objects: [...]})."""
    if isinstance(data, dict) and "objects" in data:
        items = data["objects"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError('Cần mảng objects hoặc {"objects": [...]}')
    return [_parse_object(item, i) for i, item in enumerate(items)]


class _PolygonMaskCache:
    """Vẽ mask polygon một lần — dùng cho nhiều vật thể."""

    def __init__(self, polygon: Polygon, objects: Sequence[ObjectInput], margin: int = 2):
        half_ws = [o["w"] / 2.0 for o in objects]
        half_hs = [o["h"] / 2.0 for o in objects]
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        for o, hw, hh in zip(objects, half_ws, half_hs):
            xs.extend([o["x_center"] - hw, o["x_center"] + hw])
            ys.extend([o["y_center"] - hh, o["y_center"] + hh])

        self.ox = int(np.floor(min(xs))) - margin
        self.oy = int(np.floor(min(ys))) - margin
        max_x = int(np.ceil(max(xs))) + margin
        max_y = int(np.ceil(max(ys))) + margin
        w = max(1, max_x - self.ox + 1)
        h = max(1, max_y - self.oy + 1)
        self.canvas = np.zeros((h, w), dtype=np.uint8)

        contour = polygon_to_contour(polygon).copy()
        contour[:, 0, 0] -= self.ox
        contour[:, 0, 1] -= self.oy
        self.poly_mask = np.zeros_like(self.canvas)
        cv2.fillPoly(self.poly_mask, [contour], 255)

    def evaluate(self, x_center: float, y_center: float, w: float, h: float) -> RectanglePolygonResult:
        half_w, half_h = w / 2.0, h / 2.0
        x1 = int(round(x_center - half_w - self.ox))
        y1 = int(round(y_center - half_h - self.oy))
        x2 = int(round(x_center + half_w - self.ox))
        y2 = int(round(y_center + half_h - self.oy))

        rect_mask = np.zeros_like(self.canvas)
        cv2.rectangle(rect_mask, (x1, y1), (x2, y2), 255, thickness=-1)

        rect_area = cv2.countNonZero(rect_mask)
        if rect_area == 0:
            return RectanglePolygonResult(fully_inside=False, overlaps=False)

        outside = cv2.bitwise_and(rect_mask, cv2.bitwise_not(self.poly_mask))
        fully_inside = cv2.countNonZero(outside) == 0
        overlaps = cv2.countNonZero(cv2.bitwise_and(rect_mask, self.poly_mask)) > 0
        return RectanglePolygonResult(fully_inside=fully_inside, overlaps=overlaps)


def evaluate_batch(
    polygon: Polygon,
    objects: Sequence[ObjectInput],
) -> List[ObjectResult]:
    """Kiểm tra nhiều vật thể với cùng một polygon (polygon mask chỉ build một lần)."""
    if not objects:
        return []

    cache = _PolygonMaskCache(polygon, objects)
    results: List[ObjectResult] = []

    for i, obj in enumerate(objects):
        row: ObjectResult = {
            "index": i,
            "x_center": obj["x_center"],
            "y_center": obj["y_center"],
            "w": obj["w"],
            "h": obj["h"],
        }
        if obj.get("id"):
            row["id"] = obj["id"]
        for k in ("x1", "y1", "x2", "y2"):
            if k in obj:
                row[k] = obj[k]  # type: ignore[literal-required]

        try:
            if obj["w"] <= 0 or obj["h"] <= 0:
                raise ValueError("w và h phải > 0")
            ev = cache.evaluate(
                obj["x_center"], obj["y_center"], obj["w"], obj["h"]
            )
            row["fully_inside"] = ev["fully_inside"]
            row["overlaps"] = ev["overlaps"]
            rel = classify_polygon_relation(ev["fully_inside"], ev["overlaps"])
            row["relation"] = rel
            row["relation_message"] = POLYGON_RELATION_MESSAGES[rel]
        except ValueError as e:
            row["error"] = str(e)
            row["fully_inside"] = False
            row["overlaps"] = False

        results.append(row)
    return results


def batch_summary(results: Sequence[ObjectResult]) -> dict:
    ok = [r for r in results if "error" not in r]
    return {
        "total": len(results),
        "errors": sum(1 for r in results if "error" in r),
        "inside_count": sum(1 for r in ok if r.get("relation") == "inside"),
        "intersect_count": sum(1 for r in ok if r.get("relation") == "intersect"),
        "outside_count": sum(1 for r in ok if r.get("relation") == "outside"),
    }
