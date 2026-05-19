# Rect Inside Polygon (OpenCV)

Kiểm tra hình chữ nhật (tâm + `w`, `h`) với đa giác — chạy tương tác qua bàn phím.

Hệ tọa độ ảnh: gốc trên-trái, trục **y** tăng xuống dưới.

## Cài đặt

```bash
cd C:\check-rect-inside-polygon
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Không bắt buộc `pip install -e .` — `interactive.py` tự thêm `src/` vào path.

## Chạy

```bash
python interactive.py
```

### Nhập polygon

- Từng dòng `x,y`, Enter trống để xong
- Hoặc một dòng / JSON: `100,100 500,120 ...` hoặc `[[100,100],[500,120],...]`

### Nhập vật thể

| Cách | Ví dụ |
|------|--------|
| **YOLOv8** (2 góc chéo) | `100,50,200,150` hoặc `yolo:100,50,200,150` |
| Tâm + kích thước | `210,275,180,450` hoặc `center:210,275,180,450` |
| JSON từ API YOLO | `[{"x1":10,"y1":20,"x2":100,"y2":200,"confidence":0.9}]` |
| Enter trống (dòng đầu) | 1 vật: tâm + `w, h` |
| Giả lập | `!random 1000` |
| File JSON | `@data/objects.json` |

Nếu 4 số thỏa `x2>x1` và `y2>y1` → coi là **YOLO**; ngược lại → **tâm + w,h** (vd. `210,275,180,450`).

Trong code (sau YOLO detect):

```python
from rect_check.batch import evaluate_batch, object_from_yolo_xyxy, parse_objects

polygon = [(100, 100), (500, 120), (600, 300)]
boxes = [{"x1": b["x1"], "y1": b["y1"], "x2": b["x2"], "y2": b["y2"]} for b in yolo_result["boxes"]]
results = evaluate_batch(polygon, parse_objects(boxes))
# fully_inside = trọn trong vùng nguy hiểm, overlaps = có giao (cảnh báo)
```

### Kết quả

1. Vật thể có **nằm hoàn toàn** trong polygon không?
2. Vật thể có **giao** với polygon không?

Nhiều vật thể: tóm tắt + 10 dòng chi tiết đầu.

## Cấu trúc project

```
check-rect-inside-polygon/
├── interactive.py          # entry: python interactive.py
├── _bootstrap.py           # thêm src/ vào path
├── requirements.txt
└── src/rect_check/
    ├── interactive.py      # vòng lặp nhập + in kết quả
    ├── input_helpers.py    # nhập polygon / vật thể
    ├── batch.py            # evaluate_batch (nhiều vật)
    ├── geometry.py         # OpenCV mask / polygon
    └── reporting.py        # in kết quả ra console
```
