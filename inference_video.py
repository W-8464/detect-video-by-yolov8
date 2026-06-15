import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import json
import argparse
import yaml

# 1. Load models
# - full-frame: dùng để track hand/tweezers lấy ROI
# - ROI model: detect shielding/gasket/bracket trong ROI (in-hand)
# - Pose model: detect hand keypoints
ROOT = Path(__file__).resolve().parent
full_model_path = str((ROOT / 'runs/obb/obb_full_frame_v1/weights/best.pt').resolve())
roi_model_path = str((ROOT / 'runs/obb/obb_roi_v1/weights/best.pt').resolve())
pose_model_path = str((ROOT / 'runs/pose/pose_hand_v7/weights/best.pt').resolve())

full_model = YOLO(full_model_path)
roi_model = YOLO(roi_model_path)
pose_model = YOLO(pose_model_path)


def _load_class_mapping():
    data_yaml_path = ROOT / "data.yaml"
    global_yaml_path = ROOT / "sop_global_shared.yaml"

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    class_names = data_cfg.get("names", [])
    CLASS_ID_TO_NAME = {i: str(name) for i, name in enumerate(class_names)}
    CLASS_NAME_TO_ID = {str(name): i for i, name in enumerate(class_names)}

    global_cfg = {}
    if global_yaml_path.exists():
        with open(global_yaml_path, "r", encoding="utf-8") as f:
            global_cfg = yaml.safe_load(f).get("global", {})

    raw_colors = global_cfg.get("class_colors", {})
    CLASS_COLORS = {}
    for name, color in raw_colors.items():
        cls_id = CLASS_NAME_TO_ID.get(str(name))
        if cls_id is not None:
            CLASS_COLORS[cls_id] = tuple(color)

    roi_mapping = global_cfg.get("roi_model_mapping", {})
    ROI_TO_FULL_CLASS = {}
    for roi_id_str, class_name in roi_mapping.items():
        full_id = CLASS_NAME_TO_ID.get(str(class_name))
        if full_id is not None:
            ROI_TO_FULL_CLASS[int(roi_id_str)] = full_id

    roi_small_names = global_cfg.get("roi_small_classes", [])
    ROI_SMALL_CLASSES = {CLASS_NAME_TO_ID[n] for n in roi_small_names if n in CLASS_NAME_TO_ID}

    tuning = global_cfg.get("tuning", {})
    ROI_PAD = float(tuning.get("roi_pad", 1.6))

    pose_cfg = tuning.get("pose", {})
    POSE_IMGSZ = int(pose_cfg.get("imgsz", 416))
    POSE_CONF = float(pose_cfg.get("conf", 0.25))
    POSE_IOU = float(pose_cfg.get("iou", 0.5))
    POSE_KPTS_SMOOTH_ALPHA = float(pose_cfg.get("kpts_smooth_alpha", 0.4))
    POSE_KPTS_DISPLAY_CONF = float(pose_cfg.get("kpts_display_conf", 0.3))

    return (
        CLASS_ID_TO_NAME,
        CLASS_NAME_TO_ID,
        CLASS_COLORS,
        ROI_TO_FULL_CLASS,
        ROI_SMALL_CLASSES,
        ROI_PAD,
        POSE_IMGSZ,
        POSE_CONF,
        POSE_IOU,
        POSE_KPTS_SMOOTH_ALPHA,
        POSE_KPTS_DISPLAY_CONF,
    )


(
    CLASS_ID_TO_NAME,
    CLASS_NAME_TO_ID,
    CLASS_COLORS,
    ROI_TO_FULL_CLASS,
    ROI_SMALL_CLASSES,
    ROI_PAD,
    POSE_IMGSZ,
    POSE_CONF,
    POSE_IOU,
    POSE_KPTS_SMOOTH_ALPHA,
    POSE_KPTS_DISPLAY_CONF,
) = _load_class_mapping()

HAND_CLS = CLASS_NAME_TO_ID.get("hand", 0)
TRAY_CLS = CLASS_NAME_TO_ID.get("tray", 1)
BOARD_CLS = CLASS_NAME_TO_ID.get("board", 2)
JIG_CLS = CLASS_NAME_TO_ID.get("jig", 3)
LINER_CLS = CLASS_NAME_TO_ID.get("liner", 4)
TWEEZERS_CLS = CLASS_NAME_TO_ID.get("tweezers", 5)
PCIE_CABLE_CLS = CLASS_NAME_TO_ID.get("PCIe cable", 7)
BRACKET_CLS = CLASS_NAME_TO_ID.get("bracket", 9)
HEATSINK_CLS = CLASS_NAME_TO_ID.get("heatsink", 10)
SCREW_CLS = CLASS_NAME_TO_ID.get("screw", 11)
SCREW_DRIVER_CLS = CLASS_NAME_TO_ID.get("screw driver", 12)
SCREW_MACHINE_CLS = CLASS_NAME_TO_ID.get("screw machine", 13)
FIXTURE_CLS = CLASS_NAME_TO_ID.get("fixture", 14)

# Các class muốn track/vẽ từ full-frame model
TRACK_FROM_FULL = [
    HAND_CLS, TWEEZERS_CLS, TRAY_CLS, BOARD_CLS, JIG_CLS, LINER_CLS,
    PCIE_CABLE_CLS, BRACKET_CLS, HEATSINK_CLS, SCREW_CLS,
    SCREW_DRIVER_CLS, SCREW_MACHINE_CLS, FIXTURE_CLS
]

# Khi log, ta in conf các class này; shielding/gasket vẫn áp dụng threshold riêng bên dưới.
LOG_FROM_FULL = set(TRACK_FROM_FULL) | ROI_SMALL_CLASSES

def parse_args():
    parser = argparse.ArgumentParser(description="Run two-stage OBB inference on a video.")
    parser.add_argument(
        "--input",
        type=str,
        default=str((ROOT / "datasets/xb10_5_2.mp4").resolve()),
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
    detections: list of dict {cls_id, conf, poly(4,2 int), track_id(optional), keypoints(optional)}
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

    # SKELETON for 16 keypoints (Wrist + 5 fingers x 3 joints)
    # 0: Wrist
    # 1,2,3: Thumb
    # 4,5,6: Index
    # 7,8,9: Middle
    # 10,11,12: Ring
    # 13,14,15: Pinky
    SKELETON = [
        [0, 1], [1, 2], [2, 3],      # Thumb
        [0, 4], [4, 5], [5, 6],      # Index
        [0, 7], [7, 8], [8, 9],      # Middle
        [0, 10], [10, 11], [11, 12], # Ring
        [0, 13], [13, 14], [14, 15]  # Pinky
    ]

    for i, d in enumerate(detections):
        cls_id = int(d["cls_id"])
        conf = float(d.get("conf", 0.0))
        track_id = d.get("track_id", None)
        color = CLASS_COLORS.get(cls_id, (200, 200, 200))

        pts = d["poly"].astype(np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)

        # Draw keypoints if available (Pose)
        if "keypoints" in d and d["keypoints"] is not None:
            kpts = d["keypoints"]  # (N, 3) or (N, 2)
            for kp in kpts:
                kx, ky = int(kp[0]), int(kp[1])
                conf_kp = kp[2] if len(kp) > 2 else 1.0
                if conf_kp > POSE_KPTS_DISPLAY_CONF:
                    cv2.circle(canvas, (kx, ky), 4, (0, 255, 0), -1)
            
            # Draw skeleton
            for edge in SKELETON:
                p1_idx, p2_idx = edge
                if p1_idx < len(kpts) and p2_idx < len(kpts):
                    p1 = kpts[p1_idx]
                    p2 = kpts[p2_idx]
                    conf1 = p1[2] if len(p1) > 2 else 1.0
                    conf2 = p2[2] if len(p2) > 2 else 1.0
                    if conf1 > POSE_KPTS_DISPLAY_CONF and conf2 > POSE_KPTS_DISPLAY_CONF:
                        cv2.line(canvas, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 255), 2)

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

# Smooth hand ROI per track_id để giảm giật crop window
roi_history = {}  # track_id -> (rx1, ry1, rx2, ry2)
ROI_SMOOTH_ALPHA = 0.45

# Smooth keypoints per track_id để giảm giật keypoints giữa các frame
kpts_history = {}  # track_id -> (N, 3) numpy array [x, y, conf]

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
                # Giữ raw ROI cho pose inference (không smoothing → keypoint khớp hand box)
                raw_rx1, raw_ry1, raw_rx2, raw_ry2 = rx1, ry1, rx2, ry2
                # Smooth ROI cho ROI model (shielding/gasket) để crop window ổn định
                if track_id is not None and track_id in roi_history:
                    prx1, pry1, prx2, pry2 = roi_history[track_id]
                    a = ROI_SMOOTH_ALPHA
                    rx1 = int(a * rx1 + (1 - a) * prx1)
                    ry1 = int(a * ry1 + (1 - a) * pry1)
                    rx2 = int(a * rx2 + (1 - a) * prx2)
                    ry2 = int(a * ry2 + (1 - a) * pry2)
                if track_id is not None:
                    roi_history[track_id] = (rx1, ry1, rx2, ry2)
                hand_rois.append((rx1, ry1, rx2, ry2, raw_rx1, raw_ry1, raw_rx2, raw_ry2, track_id))

    # Stage 2: chạy ROI model trên crop để detect shielding/gasket "in-hand"
    for rx1, ry1, rx2, ry2, raw_rx1, raw_ry1, raw_rx2, raw_ry2, parent_id in hand_rois:
        crop_smoothed = frame[ry1:ry2, rx1:rx2]
        if crop_smoothed.size == 0:
            continue

        # --- Pose Inference (dùng raw crop, không smoothing → keypoint khớp hand box) ---
        target_hand_det = None
        for d in detections:
            if d.get("track_id") == parent_id and d["cls_id"] == HAND_CLS:
                target_hand_det = d
                break
        
        if target_hand_det is not None:
            crop_raw = frame[raw_ry1:raw_ry2, raw_rx1:raw_rx2]
            if crop_raw.size > 0:
                crop_h, crop_w = crop_raw.shape[:2]
                raw_imgsz = max(320, min(POSE_IMGSZ, max(crop_h, crop_w)))
                dynamic_imgsz = int(round(raw_imgsz / 32.0) * 32)

                pose_results = pose_model.predict(
                    crop_raw, imgsz=dynamic_imgsz, conf=POSE_CONF, iou=POSE_IOU, verbose=False
                )
                pose_res = pose_results[0]
                if pose_res.keypoints is not None and len(pose_res.keypoints) > 0:
                    kpts_all = pose_res.keypoints.data.cpu().numpy()
                    box_all = pose_res.boxes

                    crop_cx, crop_cy = crop_w / 2.0, crop_h / 2.0
                    best_idx = 0
                    if len(kpts_all) > 1 and box_all is not None:
                        boxes_xyxy = box_all.xyxy.cpu().numpy()
                        best_dist = float("inf")
                        for m in range(len(kpts_all)):
                            bx_cx = (boxes_xyxy[m][0] + boxes_xyxy[m][2]) / 2.0
                            bx_cy = (boxes_xyxy[m][1] + boxes_xyxy[m][3]) / 2.0
                            dist = (bx_cx - crop_cx) ** 2 + (bx_cy - crop_cy) ** 2
                            if dist < best_dist:
                                best_dist = dist
                                best_idx = m

                    kpts_data = kpts_all[best_idx]
                    kpts_full = kpts_data.copy()
                    kpts_full[:, 0] += float(raw_rx1)
                    kpts_full[:, 1] += float(raw_ry1)

                    if parent_id is not None and parent_id in kpts_history:
                        prev_kpts = kpts_history[parent_id]
                        a = POSE_KPTS_SMOOTH_ALPHA
                        valid_mask = (kpts_full[:, 2] > 0.0) & (prev_kpts[:, 2] > 0.0)
                        kpts_full[valid_mask, 0] = (
                            a * kpts_full[valid_mask, 0] + (1 - a) * prev_kpts[valid_mask, 0]
                        )
                        kpts_full[valid_mask, 1] = (
                            a * kpts_full[valid_mask, 1] + (1 - a) * prev_kpts[valid_mask, 1]
                        )
                    if parent_id is not None:
                        kpts_history[parent_id] = kpts_full.copy()

                    target_hand_det["keypoints"] = kpts_full

        # --- ROI OBB Inference (dùng smoothed crop) ---
        roi_results = roi_model.predict(crop_smoothed, conf=0.25, iou=0.4, verbose=False)
        roi_res = roi_results[0]
        if roi_res.obb is None or roi_res.obb.cls is None or roi_res.obb.xyxyxyxy is None:
            continue

        roi_cls_ids = roi_res.obb.cls.cpu().numpy()
        roi_confs = roi_res.obb.conf.cpu().numpy() if roi_res.obb.conf is not None else np.zeros(len(roi_cls_ids))
        roi_polys = roi_res.obb.xyxyxyxy.cpu().numpy()

        for j in range(len(roi_cls_ids)):
            roi_cls = int(roi_cls_ids[j])
            full_cls = ROI_TO_FULL_CLASS.get(roi_cls, None)
            if full_cls not in ROI_SMALL_CLASSES:  # chỉ quan tâm shielding/gasket/bracket
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

    # Dọn dẹp roi_history và kpts_history cho track không còn xuất hiện
    active_ids = {d.get("track_id") for d in detections if d.get("track_id") is not None}
    for tid in list(roi_history.keys()):
        if tid not in active_ids:
            del roi_history[tid]
    for tid in list(kpts_history.keys()):
        if tid not in active_ids:
            del kpts_history[tid]

    # Giảm rối: với shielding/gasket, giữ top-k theo mỗi track_id (tay/nhíp)
    TOPK_PER_PARENT = 3
    filtered = []
    small_by_key = {}
    for d in detections:
        cls_id = int(d["cls_id"])
        if cls_id not in ROI_SMALL_CLASSES:
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
        det_data = {
            "class": names_by_cls.get(cls_id, str(cls_id)),
            "cls_id": cls_id,
            "conf": float(d.get("conf", 0.0)),
            "track_id": d.get("track_id", None),
            "poly": [[float(x), float(y)] for x, y in poly.tolist()],
        }
        if "keypoints" in d and d["keypoints"] is not None:
            det_data["keypoints"] = [[float(kp[0]), float(kp[1]), float(kp[2])] for kp in d["keypoints"]]
        
        frame_payload["detections"].append(det_data)
    det_file.write(json.dumps(frame_payload, ensure_ascii=False) + "\n")

    # In log kết quả OBB
    for d in detections:
        cls_id = int(d["cls_id"])
        if cls_id not in LOG_FROM_FULL:
            continue
        conf = float(d.get("conf", 0.0))
        if cls_id in ROI_SMALL_CLASSES and conf < 0.5:
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