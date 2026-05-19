"""Parse polygon và hỗ trợ điểm/góc (OpenCV)."""

from __future__ import annotations

import json
from typing import List, Sequence, Tuple, TypedDict

import cv2
import numpy as np

Point = Tuple[float, float]
Polygon = Sequence[Point]


class RectanglePolygonResult(TypedDict):
    """Kết quả kiểm tra vật thể (hình chữ nhật) với đa giác."""

    fully_inside: bool
    overlaps: bool


def parse_polygon(value: str) -> List[Point]:
    """
    Parse đa giác từ chuỗi.

    - "x1,y1 x2,y2 ..." hoặc phân tách bằng `;`
    - JSON: [[100,100], [500,120], ...]
    """
    value = value.strip()
    if value.startswith("["):
        data = json.loads(value)
        return [(float(p[0]), float(p[1])) for p in data]
    points: List[Point] = []
    for token in value.replace(";", " ").split():
        if not token:
            continue
        parts = token.split(",")
        if len(parts) != 2:
            raise ValueError(f"Điểm không hợp lệ: {token!r} (cần dạng x,y)")
        points.append((float(parts[0]), float(parts[1])))
    if len(points) < 3:
        raise ValueError("Đa giác cần ít nhất 3 đỉnh")
    return points


def polygon_to_contour(polygon: Polygon) -> np.ndarray:
    """Contour OpenCV shape (N, 1, 2) int32."""
    pts = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
    return np.round(pts).astype(np.int32)


def yolo_xyxy_to_center_size(
    x1: float, y1: float, x2: float, y2: float
) -> Tuple[float, float, float, float]:
    """
    Chuyển bbox YOLOv8 (2 góc chéo) → tâm + kích thước.

    Ultralytics / API thường trả x1,y1 = góc trên-trái, x2,y2 = góc dưới-phải (pixel).
    """
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        raise ValueError("BBox YOLO không hợp lệ (x2-x1 và y2-y1 phải > 0)")
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, w, h


def looks_like_yolo_xyxy(a: float, b: float, c: float, d: float) -> bool:
    """4 số giống x1,y1,x2,y2 (góc dưới-phải nằm sau góc trên-trái)."""
    return c > a and d > b


def rect_corners(x_center: float, y_center: float, w: float, h: float) -> List[Point]:
    """4 góc hình chữ nhật từ tâm và kích thước."""
    half_w, half_h = w / 2.0, h / 2.0
    return [
        (x_center - half_w, y_center - half_h),
        (x_center + half_w, y_center - half_h),
        (x_center + half_w, y_center + half_h),
        (x_center - half_w, y_center + half_h),
    ]


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """Điểm có nằm trong đa giác (kể cả trên cạnh)."""
    contour = polygon_to_contour(polygon)
    result = cv2.pointPolygonTest(contour, point, measureDist=False)
    return result >= 0
