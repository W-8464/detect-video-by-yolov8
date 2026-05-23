import cv2
import numpy as np
from ultralytics import YOLOWorld
from pathlib import Path
import json
import argparse

ROOT = Path(__file__).resolve().parent

# 1. Khởi tạo mô hình YOLO-World
# Sẽ tự động tải file yolov8s-worldv2.pt nếu chưa có trong máy
yolo_world_model_path = "yolov8s-worldv2.pt"
model = YOLOWorld(yolo_world_model_path)

# 2. Cài đặt các class bằng văn bản (Prompt Engineering)
# CHÚ Ý: Đây là thủ thuật quan trọng cho YOLO-World.
# Từ khóa càng chi tiết, nhận diện càng chính xác (Ví dụ: thay vì 'board', ta dùng 'green printed circuit board')

CLASSES_TEXT = [
    'hand', 'tray', 'board', 'jig'
]

# Đây là bộ từ khóa mô tả mà YOLO-World sẽ đọc (Prompt)
CLASSES_PROMPTS = [
    'human hand',                  # 0: hand
    'black plastic parts tray',          # 1: tray
    'green printed circuit board', # 2: board (tránh hiểu lầm là bảng đen, thớt gỗ)
    'black plastic parts tray', # 3: jig
]

# Đưa Prompt vào model thay vì chữ viết tắt
model.set_classes(CLASSES_PROMPTS)

def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO-World inference on a video.")
    parser.add_argument(
        "--input",
        type=str,
        default=str((ROOT / "datasets/xb10_14.mp4").resolve()),
        help="Input video path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output video path (default: <input_stem>_world_output.mp4)",
    )
    parser.add_argument(
        "--detections",
        type=str,
        default="",
        help="Output detections jsonl path (default: <input_stem>_world_detections.jsonl)",
    )
    return parser.parse_args()

args = parse_args()
input_video_path = str(Path(args.input).resolve())
input_stem = Path(input_video_path).stem
output_video_path = str(Path(args.output).resolve()) if args.output else str((ROOT / f"{input_stem}_world_output.mp4").resolve())
detections_output_path = str(Path(args.detections).resolve()) if args.detections else str((ROOT / f"{input_stem}_world_detections.jsonl").resolve())

# 3. Bảng màu cho từng class (BGR) - giữ nguyên như file cũ
CLASS_COLORS = {
    0: (0, 255, 0),      # hand
    1: (255, 0, 0),      # tray
    2: (0, 0, 255),      # board
    3: (255, 255, 0),    # jig
}

def xyxy_to_poly(x1, y1, x2, y2):
    """
    YOLO-World trả về HBB (xyxy). Để tương thích với kiến trúc cũ dùng OBB (poly),
    ta mô phỏng 4 đỉnh của hình hộp chữ nhật để module segment_actions.py vẫn chạy bình thường.
    """
    return np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ], dtype=np.float32)

def draw_polys_smart(frame, detections):
    """ Giữ nguyên code clean vẽ boxes """
    canvas = frame.copy()
    if not detections:
        return canvas

    best_idx = {}
    for i, d in enumerate(detections):
        c = int(d["cls_id"])
        conf = float(d.get("conf", 0.0))
        if c not in best_idx or conf > float(detections[best_idx[c]].get("conf", 0.0)):
            best_idx[c] = i

    for i, d in enumerate(detections):
        cls_id = int(d["cls_id"])
        conf = float(d.get("conf", 0.0))
        track_id = d.get("track_id", None)
        color = CLASS_COLORS.get(cls_id, (200, 200, 200))

        pts = d["poly"].astype(np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)

        if best_idx.get(cls_id) != i:
            continue

        cls_name = CLASSES_TEXT[cls_id]
        if track_id is not None:
            label = f"{cls_name} #{track_id} {conf:.2f}"
        else:
            label = f"{cls_name} {conf:.2f}"

        top_idx = int(np.argmin(pts[:, 1]))
        text_x = int(pts[top_idx, 0])
        text_y = max(20, int(pts[top_idx, 1]) - 8)

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(
            canvas,
            (text_x - 1, text_y - th - 4),
            (text_x + tw + 4, text_y + baseline + 2),
            color,
            -1,
        )
        cv2.putText(canvas, label, (text_x + 1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    return canvas

# 4. Open video
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print(f"Lỗi: Không thể mở video: {input_video_path}")
    raise SystemExit

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 25

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

print(f"Bắt đầu xử lý video bằng YOLO-World... {width}x{height}, FPS={fps}")
print(f"Ghi detections JSONL: {detections_output_path}")

frame_idx = 0
det_file = open(detections_output_path, "w", encoding="utf-8")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame_idx += 1

    # Thay vì 2 stage, YOLO-World có thể detect mọi thứ chung 1 lần quét
    results = model.track(
        frame,
        persist=True,
        conf=0.1,  # Đặt mức Confidence thấp để nhận diện nhiều vật thể zero-shot hơn
        iou=0.4,
        verbose=False
    )

    result = results[0]
    detections = []

    if result.boxes is not None and result.boxes.cls is not None:
        cls_ids = result.boxes.cls.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else None
        xyxys = result.boxes.xyxy.cpu().numpy()

        for i in range(len(cls_ids)):
            cls_id = int(cls_ids[i])
            conf = float(confs[i])
            track_id = int(ids[i]) if ids is not None else None
            x1, y1, x2, y2 = xyxys[i]
            
            # Quy đổi sang poly để lưu json tương thích với segment_actions.py
            poly = xyxy_to_poly(x1, y1, x2, y2)

            detections.append({
                "cls_id": cls_id, 
                "conf": conf, 
                "poly": poly, 
                "track_id": track_id
            })

    # Vẽ bounding box (đã giả lập poly)
    annotated_frame = draw_polys_smart(frame, detections)
    out.write(annotated_frame)

    # Ghi file json chứa tracking
    frame_payload = {"frame": frame_idx, "detections": []}
    for d in detections:
        cls_id = int(d["cls_id"])
        poly = d["poly"]
        frame_payload["detections"].append(
            {
                "class": CLASSES_TEXT[cls_id],
                "cls_id": cls_id,
                "conf": float(d.get("conf", 0.0)),
                "track_id": d.get("track_id", None),
                "poly": [[float(x), float(y)] for x, y in poly.tolist()],
            }
        )
    det_file.write(json.dumps(frame_payload, ensure_ascii=False) + "\n")

    # In log trạng thái
    for d in detections:
        cls_id = int(d["cls_id"])
        conf = float(d.get("conf", 0.0))
        cls_name = CLASSES_TEXT[cls_id]
        track_id = d.get("track_id", None)
        poly = d["poly"]
        cx = float(poly[:, 0].mean())
        cy = float(poly[:, 1].mean())
        print(f"frame={frame_idx}, id={track_id}, class={cls_name}, conf={conf:.3f}, cx={cx:.1f}, cy={cy:.1f}")

cap.release()
out.release()
det_file.close()
cv2.destroyAllWindows()

print(f"Hoàn tất! Video kết quả: {output_video_path}")
