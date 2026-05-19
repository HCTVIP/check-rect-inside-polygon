"""In kết quả kiểm tra hình chữ nhật / đa giác."""

from __future__ import annotations

from typing import Sequence

from .geometry import Polygon, RectanglePolygonResult, point_in_polygon, rect_corners


def corner_details(
    x_center: float,
    y_center: float,
    w: float,
    h: float,
    polygon: Polygon,
) -> dict:
    corners = rect_corners(x_center, y_center, w, h)
    return {
        "corners": corners,
        "corners_inside": [point_in_polygon(c, polygon) for c in corners],
        "center_inside": point_in_polygon((x_center, y_center), polygon),
    }


def print_rectangle_report(
    x_center: float,
    y_center: float,
    w: float,
    h: float,
    polygon: Sequence,
    result: RectanglePolygonResult,
    *,
    show_polygon: bool = False,
) -> None:
    details = corner_details(x_center, y_center, w, h, polygon)

    if show_polygon:
        print(f"Polygon ({len(polygon)} đỉnh): {list(polygon)}")
    print(f"Vật thể: tâm=({x_center}, {y_center}), kích thước=({w} x {h})")
    print(f"4 góc: {details['corners']}")
    print(f"Từng góc trong vùng: {details['corners_inside']}")
    print(f"Tâm trong vùng: {details['center_inside']}")
    print()
    print(
        "1. Vật thể có nằm hoàn toàn trong Polygon (khu vực) không? "
        f"{'CÓ' if result['fully_inside'] else 'KHÔNG'}"
    )
    print(
        "2. Vật thể có giao với Polygon (khu vực) không? "
        f"{'CÓ' if result['overlaps'] else 'KHÔNG'}"
    )


def _format_object_label(row: dict) -> str:
    if all(k in row for k in ("x1", "y1", "x2", "y2")):
        return (
            f"YOLO ({row['x1']}, {row['y1']})→({row['x2']}, {row['y2']}) "
            f"· tâm=({row['x_center']}, {row['y_center']}) {row['w']}x{row['h']}"
        )
    return f"tâm=({row['x_center']}, {row['y_center']}), {row['w']}x{row['h']}"


def print_batch_report(
    polygon: Sequence,
    results: Sequence,
    summary: dict,
    *,
    show_polygon: bool = False,
    preview: int = 10,
) -> None:
    if show_polygon:
        print(f"Polygon ({len(polygon)} đỉnh): {list(polygon)}")
    print(f"Số vật thể: {summary['total']}")
    print()
    print(
        f"  Trọn trong vùng: {summary['fully_inside_count']} vật thể"
    )
    print(f"  Có giao vùng:     {summary['overlaps_count']} vật thể")
    print(f"  Không giao:       {summary['no_overlap_count']} vật thể")
    if summary.get("errors"):
        print(f"  Lỗi:              {summary['errors']} vật thể")

    if not results:
        return

    if len(results) == 1:
        r = results[0]
        print()
        print_rectangle_report(
            r["x_center"],
            r["y_center"],
            r["w"],
            r["h"],
            polygon,
            {
                "fully_inside": bool(r.get("fully_inside")),
                "overlaps": bool(r.get("overlaps")),
            },
        )
        return

    n = min(preview, len(results))
    print(f"\nChi tiết ({n}/{len(results)} dòng đầu):")
    for row in results[:n]:
        label = row.get("id") or f"#{row['index']}"
        if row.get("error"):
            print(f"  {label}: LỖI — {row['error']}")
            continue
        print(
            f"  {label}: {_format_object_label(row)} → "
            f"trọn={'CÓ' if row.get('fully_inside') else 'KHÔNG'}, "
            f"giao={'CÓ' if row.get('overlaps') else 'KHÔNG'}"
        )
    if len(results) > n:
        print(f"  ... và {len(results) - n} vật thể nữa")
