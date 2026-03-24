import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import json
import argparse

# 1. Load models
# - full-frame: dùng để track hand/tweezers lấy ROI
# - ROI model: detect shielding/gasket trong ROI (in-hand)
ROOT = Path(__file__).resolve().parent
full_model_path = str((ROOT / 'runs/obb/action_model_v62/weights/best.pt').resolve())
roi_model_path = str((ROOT / 'runs/obb/action_model_roi_v1/weights/best.pt').resolve())

full_model = YOLO(full_model_path)
roi_model = YOLO(roi_model_path)

def parse_args():
    parser = argparse.ArgumentParser(description="Run two-stage OBB inference on a video.")
    parser.add_argument(
        "--input",
        type=str,
        default=str((ROOT / "datasets/xb10_6.mp4").resolve()),
        help="Input video path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output video path (default: <input_stem>_output.mp4 in project root)",
    )
    parser.add_argument(
        "--detections",
        type=str,
        default="",
        help="Output detections jsonl path (default: <input_stem>_detections.jsonl in project root)",
    )
    return parser.parse_args()


args = parse_args()
input_video_path = str(Path(args.input).resolve())
input_stem = Path(input_video_path).stem
output_video_path = str(Path(args.output).resolve()) if args.output else str((ROOT / f"{input_stem}_output.mp4").resolve())
detections_output_path = str(Path(args.detections).resolve()) if args.detections else str((ROOT / f"{input_stem}_detections.jsonl").resolve())

# 3. Bảng màu cho từng class (BGR) - thứ tự theo data.yaml
CLASS_COLORS = {
    0: (0, 255, 0),      # hand - xanh lá
    1: (255, 0, 0),      # tray - xanh dương
    2: (0, 0, 255),      # board - đỏ
    3: (255, 255, 0),    # jig - cyan
    4: (0, 255, 255),    # liner - vàng
    5: (128, 255, 128),  # tweezers - xanh lá nhạt
    6: (255, 128, 0),    # shielding - cam
    7: (255, 0, 255),    # PCIe cable - hồng
    8: (128, 0, 255),    # gasket - tím
}

HAND_CLS = 0
TRAY_CLS = 1
BOARD_CLS = 2
JIG_CLS = 3
LINER_CLS = 4
TWEEZERS_CLS = 5
PCIE_CABLE_CLS = 7
ROI_PAD = 1.6  # crop rộng hơn bbox tay/nhíp

# ROI model được train với data_roi.yaml (re-index): 0 hand, 1 tweezers, 2 shielding, 3 gasket
ROI_TO_FULL_CLASS = {
    0: HAND_CLS,
    1: TWEEZERS_CLS,
    2: 6,  # shielding
    3: 8,  # gasket
}

# Các class muốn track/vẽ từ full-frame model (action_model_v62)
TRACK_FROM_FULL = [HAND_CLS, TWEEZERS_CLS, TRAY_CLS, BOARD_CLS, JIG_CLS, LINER_CLS, PCIE_CABLE_CLS]

# Khi log, ta in conf các class này; shielding/gasket vẫn áp dụng threshold riêng bên dưới.
LOG_FROM_FULL = set(TRACK_FROM_FULL) | {6, 8}


def poly_aabb(poly: np.ndarray):
    xs = poly[:, 0]
    ys = poly[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def expand_clip_aabb(aabb, w, h, pad):
    x1, y1, x2, y2 = aabb
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = max(2.0, (x2 - x1) * pad)
    bh = max(2.0, (y2 - y1) * pad)
    nx1 = int(round(cx - bw / 2.0))
    ny1 = int(round(cy - bh / 2.0))
    nx2 = int(round(cx + bw / 2.0))
    ny2 = int(round(cy + bh / 2.0))
    nx1 = max(0, min(w - 2, nx1))
    ny1 = max(0, min(h - 2, ny1))
    nx2 = max(nx1 + 2, min(w, nx2))
    ny2 = max(ny1 + 2, min(h, ny2))
    return nx1, ny1, nx2, ny2


def draw_polys_smart(frame, detections, names_by_cls):
    """
    detections: list of dict {cls_id, conf, poly(4,2 int), track_id(optional)}
    Chỉ dán nhãn 1 box/class (conf cao nhất) để đỡ rối.
    """
    canvas = frame.copy()
    if not detections:
        return canvas

    # best per class
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

        cls_name = names_by_cls.get(cls_id, str(cls_id))
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


def draw_obb_smart(frame, result, model):
    """
    Vẽ OBB boxes lên frame.
    Mỗi class chỉ hiển thị TEXT LABEL 1 lần (box có conf cao nhất).
    Các box còn lại chỉ vẽ viền, không dán nhãn → tránh rối.
    """
    if result.obb is None or result.obb.cls is None:
        return frame

    canvas = frame.copy()

    cls_ids = result.obb.cls.cpu().numpy()
    confs = result.obb.conf.cpu().numpy() if result.obb.conf is not None else np.zeros(len(cls_ids))
    ids = result.obb.id.cpu().numpy() if result.obb.id is not None else None

    # Lấy tọa độ 4 đỉnh polygon cho mỗi box
    # xyxyxyxy: shape (N, 4, 2) - 4 góc, mỗi góc (x, y)
    polys = result.obb.xyxyxyxy.cpu().numpy() if result.obb.xyxyxyxy is not None else None
    if polys is None:
        return canvas

    # Bước 1: Tìm box có conf cao nhất cho mỗi class → sẽ được dán nhãn
    best_per_class = {}  # class_id -> index (chỉ index có conf cao nhất)
    for i in range(len(cls_ids)):
        cls_id = int(cls_ids[i])
        conf = float(confs[i])
        if cls_id not in best_per_class or conf > confs[best_per_class[cls_id]]:
            best_per_class[cls_id] = i

    # Bước 2: Vẽ tất cả boxes
    for i in range(len(cls_ids)):
        cls_id = int(cls_ids[i])
        conf = float(confs[i])
        track_id = int(ids[i]) if ids is not None else None
        color = CLASS_COLORS.get(cls_id, (200, 200, 200))

        # Lấy 4 đỉnh polygon
        pts = polys[i].astype(np.int32)  # shape (4, 2)

        # Vẽ viền box
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)

        # Chỉ dán nhãn cho box có conf cao nhất trong mỗi class
        is_best = (best_per_class.get(cls_id) == i)
        if is_best:
            cls_name = model.names[cls_id]
            if track_id is not None:
                label = f"{cls_name} #{track_id} {conf:.2f}"
            else:
                label = f"{cls_name} {conf:.2f}"

            # Vị trí text: đỉnh cao nhất (y nhỏ nhất)
            top_idx = np.argmin(pts[:, 1])
            text_x = int(pts[top_idx, 0])
            text_y = int(pts[top_idx, 1]) - 8
            text_y = max(20, text_y)

            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            # Nền label
            cv2.rectangle(canvas,
                          (text_x - 1, text_y - th - 4),
                          (text_x + tw + 4, text_y + baseline + 2),
                          color, -1)
            # Text
            cv2.putText(canvas, label, (text_x + 1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

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

print(f"Bắt đầu xử lý video... {width}x{height}, FPS={fps}")
print(f"Ghi detections JSONL: {detections_output_path}")

frame_idx = 0
det_file = open(detections_output_path, "w", encoding="utf-8")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame_idx += 1

    # Stage 1: track hand/tweezers để lấy ROI
    full_results = full_model.track(
        frame,
        persist=True,
        conf=0.2,
        iou=0.4,
        verbose=False,
        classes=TRACK_FROM_FULL,
    )

    full_result = full_results[0]

    detections = []
    names_by_cls = {i: n for i, n in full_model.names.items()} if hasattr(full_model, "names") else {}

    # Thu thập detections từ full model:
    # - hand/tweezers: dùng để tạo ROI
    # - tray/board/jig/PCIe cable: chỉ để vẽ/log theo yêu cầu
    hand_rois = []
    if full_result.obb is not None and full_result.obb.cls is not None and full_result.obb.xyxyxyxy is not None:
        cls_ids = full_result.obb.cls.cpu().numpy()
        confs = full_result.obb.conf.cpu().numpy() if full_result.obb.conf is not None else np.zeros(len(cls_ids))
        ids = full_result.obb.id.cpu().numpy() if full_result.obb.id is not None else None
        polys = full_result.obb.xyxyxyxy.cpu().numpy()

        for i in range(len(cls_ids)):
            cls_id = int(cls_ids[i])
            if cls_id not in set(TRACK_FROM_FULL):
                continue
            conf = float(confs[i])
            track_id = int(ids[i]) if ids is not None else None
            poly = polys[i]

            detections.append({"cls_id": cls_id, "conf": conf, "poly": poly, "track_id": track_id})

            if cls_id in (HAND_CLS, TWEEZERS_CLS):
                aabb = poly_aabb(poly)
                rx1, ry1, rx2, ry2 = expand_clip_aabb(aabb, width, height, ROI_PAD)
                hand_rois.append((rx1, ry1, rx2, ry2, track_id))

    # Stage 2: chạy ROI model trên crop để detect shielding/gasket "in-hand"
    for rx1, ry1, rx2, ry2, parent_id in hand_rois:
        crop = frame[ry1:ry2, rx1:rx2]
        if crop.size == 0:
            continue

        roi_results = roi_model.predict(crop, conf=0.25, iou=0.4, verbose=False)
        roi_res = roi_results[0]
        if roi_res.obb is None or roi_res.obb.cls is None or roi_res.obb.xyxyxyxy is None:
            continue

        roi_cls_ids = roi_res.obb.cls.cpu().numpy()
        roi_confs = roi_res.obb.conf.cpu().numpy() if roi_res.obb.conf is not None else np.zeros(len(roi_cls_ids))
        roi_polys = roi_res.obb.xyxyxyxy.cpu().numpy()

        for j in range(len(roi_cls_ids)):
            roi_cls = int(roi_cls_ids[j])
            full_cls = ROI_TO_FULL_CLASS.get(roi_cls, None)
            if full_cls not in (6, 8):  # chỉ quan tâm shielding/gasket
                continue

            poly = roi_polys[j].copy()
            poly[:, 0] += float(rx1)
            poly[:, 1] += float(ry1)

            detections.append(
                {
                    "cls_id": int(full_cls),
                    "conf": float(roi_confs[j]),
                    "poly": poly,
                    "track_id": parent_id,
                }
            )

    # Giảm rối: với shielding/gasket, giữ top-k theo mỗi track_id (tay/nhíp)
    TOPK_PER_PARENT = 3
    filtered = []
    small_by_key = {}
    for d in detections:
        cls_id = int(d["cls_id"])
        if cls_id not in (6, 8):
            filtered.append(d)
            continue
        key = (d.get("track_id", None), cls_id)
        small_by_key.setdefault(key, []).append(d)

    for key, items in small_by_key.items():
        items_sorted = sorted(items, key=lambda x: float(x.get("conf", 0.0)), reverse=True)
        filtered.extend(items_sorted[:TOPK_PER_PARENT])

    detections = filtered

    # Vẽ detections tổng hợp (tay/nhíp + shielding/gasket in-hand)
    annotated_frame = draw_polys_smart(frame, detections, names_by_cls)
    out.write(annotated_frame)

    # Lưu detections theo frame để segment_actions.py dùng trực tiếp
    frame_payload = {"frame": frame_idx, "detections": []}
    for d in detections:
        cls_id = int(d["cls_id"])
        poly = d["poly"]
        frame_payload["detections"].append(
            {
                "class": names_by_cls.get(cls_id, str(cls_id)),
                "cls_id": cls_id,
                "conf": float(d.get("conf", 0.0)),
                "track_id": d.get("track_id", None),
                "poly": [[float(x), float(y)] for x, y in poly.tolist()],
            }
        )
    det_file.write(json.dumps(frame_payload, ensure_ascii=False) + "\n")

    # In log kết quả OBB
    for d in detections:
        cls_id = int(d["cls_id"])
        if cls_id not in LOG_FROM_FULL:
            continue
        conf = float(d.get("conf", 0.0))
        if cls_id in (6, 8) and conf < 0.5:
            continue
        cls_name = names_by_cls.get(cls_id, str(cls_id))
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