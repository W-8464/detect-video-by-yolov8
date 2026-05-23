"""
Real-time camera inference using two-stage OBB detection.
- Stage 1: full model (action_model_v7) → track hand, tweezers, tray, board, jig, liner, PCIe cable, bracket
- Stage 2: ROI model (action_model_roi_v4) → detect shielding/gasket/bracket in hand/tweezers ROI

Usage:
    python inference_camera.py                              # webcam mặc định (index 0)
    python inference_camera.py --camera 0                   # webcam theo index
    python inference_camera.py --camera rtsp://...          # RTSP stream
    python inference_camera.py --video path/to/vid.mp4      # test với video file
    python inference_camera.py --screen                     # chụp toàn màn hình
    python inference_camera.py --screen --monitor 1         # chọn monitor cụ thể
    python inference_camera.py --screen --region 0 0 1280 720  # crop vùng (x y w h)
    python inference_camera.py --save                       # lưu video output
    python inference_camera.py --save --detections out.jsonl  # lưu detections

Controls:
    q / ESC  : thoát
    s        : screenshot (lưu frame hiện tại)
    f        : toggle fullscreen
    +/-      : tăng/giảm tốc độ phát (chỉ khi --video)
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path
import json
import argparse
import time
import signal
import threading
from datetime import datetime
import csv
try:
    from segment_actions_refactored import Detection
    from streaming_engine import StreamingActionEngine
    HAS_ACTION_ENGINE = True
except ImportError:
    HAS_ACTION_ENGINE = False

try:
    import mss
    import mss.tools
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False

# ─── Load models ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
full_model_path = str((ROOT / 'runs/obb/action_model_v7/weights/best.pt').resolve())
roi_model_path = str((ROOT / 'runs/obb/action_model_roi_v4/weights/best.pt').resolve())

MODEL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
full_model = YOLO(full_model_path).to(MODEL_DEVICE)
roi_model = YOLO(roi_model_path).to(MODEL_DEVICE)

# ─── Class mapping & colors (giống inference_video.py) ────────────────────────
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
    9: (0, 128, 255),    # bracket - cam nhạt
}

HAND_CLS = 0
TRAY_CLS = 1
BOARD_CLS = 2
JIG_CLS = 3
LINER_CLS = 4
TWEEZERS_CLS = 5
PCIE_CABLE_CLS = 7
BRACKET_CLS = 9
ROI_PAD = 1.6

ROI_TO_FULL_CLASS = {
    0: HAND_CLS,
    1: TWEEZERS_CLS,
    2: 6,  # shielding
    3: 8,  # gasket
}

# ROI small-object classes (detect từ ROI model, map về full-frame class id)
ROI_SMALL_CLASSES = {6, 8}  # shielding, gasket

TRACK_FROM_FULL = [HAND_CLS, TWEEZERS_CLS, TRAY_CLS, BOARD_CLS, JIG_CLS, LINER_CLS, PCIE_CABLE_CLS, BRACKET_CLS]
LOG_FROM_FULL = set(TRACK_FROM_FULL) | ROI_SMALL_CLASSES


# ─── Geometry helpers ─────────────────────────────────────────────────────────
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


# ─── Drawing ──────────────────────────────────────────────────────────────────
def draw_polys_smart(frame, detections, names_by_cls):
    """
    Vẽ OBB polygons lên frame.
    Mỗi class chỉ hiển thị label 1 lần (conf cao nhất) để tránh rối.
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


def draw_fps_overlay(frame, fps, frame_idx):
    """Vẽ FPS và frame counter lên góc trái trên."""
    overlay_text = f"FPS: {fps:.1f} | Frame: {frame_idx}"
    (tw, th), baseline = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (8, 8), (tw + 16, th + baseline + 16), (0, 0, 0), -1)
    cv2.putText(frame, overlay_text, (12, th + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return frame


def run_two_stage_inference(frame, width, height, names_by_cls):
    """
    Chạy two-stage OBB inference trên 1 frame:
    1. full_model.track() -> hand/tweezers/tray/board/jig/liner/PCIe cable
    2. roi_model.predict() BATCH trên tất cả hand/tweezers.
    3. Filter: Lấy top-k shielding/gasket tốt nhất cho mỗi bàn tay (track_id).
    """
    # Stage 1: track trên full frame
    full_results = full_model.track(
        frame,
        persist=True,
        conf=0.4,
        iou=0.4,
        verbose=False,
        classes=TRACK_FROM_FULL,
    )
    full_result = full_results[0]

    detections = []
    hand_rois = []

    if full_result.obb is not None and full_result.obb.cls is not None and full_result.obb.xyxyxyxy is not None:
        cls_ids = full_result.obb.cls.cpu().numpy()
        confs = full_result.obb.conf.cpu().numpy()
        ids = full_result.obb.id.cpu().numpy() if full_result.obb.id is not None else None
        polys = full_result.obb.xyxyxyxy.cpu().numpy()

        for i in range(len(cls_ids)):
            cls_id = int(cls_ids[i])
            if cls_id not in set(TRACK_FROM_FULL): continue
            conf = float(confs[i])
            track_id = int(ids[i]) if ids is not None else None
            poly = polys[i]
            detections.append({"cls_id": cls_id, "conf": conf, "poly": poly, "track_id": track_id})

            if cls_id in (HAND_CLS, TWEEZERS_CLS):
                aabb = poly_aabb(poly)
                rx1, ry1, rx2, ry2 = expand_clip_aabb(aabb, width, height, ROI_PAD)
                hand_rois.append((rx1, ry1, rx2, ry2, track_id))

    # Stage 2: Batch ROI model
    crops = []
    roi_meta = []
    for rx1, ry1, rx2, ry2, parent_id in hand_rois:
        crop = frame[ry1:ry2, rx1:rx2]
        if crop.size == 0: continue
        crops.append(crop)
        roi_meta.append((rx1, ry1, parent_id))
    
    if crops:
        roi_results = roi_model.predict(crops, conf=0.25, iou=0.4, verbose=False)
        for idx, roi_res in enumerate(roi_results):
            rx1, ry1, p_id = roi_meta[idx]
            if roi_res.obb is not None and roi_res.obb.cls is not None:
                r_cls_ids = roi_res.obb.cls.cpu().numpy()
                r_confs = roi_res.obb.conf.cpu().numpy()
                r_polys = roi_res.obb.xyxyxyxy.cpu().numpy()
                for j in range(len(r_cls_ids)):
                    r_cls = int(r_cls_ids[j])
                    f_cls = ROI_TO_FULL_CLASS.get(r_cls)
                    if f_cls not in ROI_SMALL_CLASSES: continue
                    poly = r_polys[j].copy()
                    poly[:, 0] += float(rx1)
                    poly[:, 1] += float(ry1)
                    detections.append({"cls_id": int(f_cls), "conf": float(r_confs[j]), "poly": poly, "track_id": p_id})

    # Filter: top-k shielding/gasket per parent track_id
    TOPK_PER_PARENT = 2
    filtered = []
    small_by_key = {}
    for d in detections:
        cid = int(d["cls_id"])
        if cid not in ROI_SMALL_CLASSES:
            filtered.append(d)
        else:
            key = (d.get("track_id"), cid)
            small_by_key.setdefault(key, []).append(d)
    
    for key, items in small_by_key.items():
        items_sorted = sorted(items, key=lambda x: float(x.get("conf", 0.0)), reverse=True)
        filtered.extend(items_sorted[:TOPK_PER_PARENT])
    
    return filtered

class VideoGet:
    """Threaded capture với bộ đếm frame để tránh xử lý trùng (optimizing for slow cap)."""
    def __init__(self, cap, stop_on_failure=False):
        self.cap = cap
        self.ret, self.frame = self.cap.read()
        self.frame_count = 0
        self.stopped = False
        self.stop_on_failure = stop_on_failure
        self.reached_eof = False
        self.lock = threading.Lock()
    def start(self):
        t = threading.Thread(target=self.get, args=())
        t.daemon = True
        t.start()
        return self
    def get(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret, self.frame = ret, frame
                    self.frame_count += 1
            else:
                if self.stop_on_failure:
                    with self.lock:
                        self.ret = False
                        self.reached_eof = True
                        self.stopped = True
                    break
                time.sleep(0.01) # Small sleep if cap is failing
    def read(self):
        with self.lock:
            return self.ret, self.frame, self.frame_count, self.reached_eof
    def stop(self):
        self.stopped = True


# ─── Video re-encode helper ───────────────────────────────────────────────────
def _reencode_video_fps(src_path: str, dst_path: str, actual_fps: float) -> bool:
    """
    Đọc video tại src_path (đã ghi với fps sai) rồi ghi lại ra dst_path
    với actual_fps đúng. Trả về True nếu thành công.
    Không cần ffmpeg — dùng thuần cv2.
    """
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        return False
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(dst_path, fourcc, actual_fps, (w, h))
    if not out.isOpened():
        cap.release()
        return False
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.write(frame)
        written += 1
        if written % 50 == 0:
            print(f"   Re-encoding... {written}/{total} frames", end="\r")
    cap.release()
    out.release()
    if written > 0:
        print(f"   Re-encoding hoàn tất: {written} frames @ {actual_fps:.2f} fps{' '*20}")
    return written > 0


# ─── Screen capture helper ────────────────────────────────────────────────────
try:
    from PIL import ImageGrab as _ImageGrab
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

class ScreenCapture:
    """
    Dual-backend screen capture:
      - Thử mss trước (nhanh, X11)
      - Tự fallback sang PIL.ImageGrab nếu mss fail (tương thích Wayland)
    API giống cv2.VideoCapture: .read() trả về (True, frame_bgr)
    """
    def __init__(self, monitor_idx: int = 1, region=None):
        self._sct      = None
        self._mon      = None
        self._pil_bbox = None  # (left, top, right, bottom)
        self._backend  = None  # "mss" | "pil"
        self.fps       = 30.0

        # ── Thử mss (X11) ─────────────────────────────────────────
        if MSS_AVAILABLE:
            try:
                sct = mss.mss()
                monitors = sct.monitors
                if monitor_idx < 0 or monitor_idx >= len(monitors):
                    print(f"⚠️  Monitor {monitor_idx} không tồn tại, dùng primary (1).")
                    monitor_idx = 1
                base = monitors[monitor_idx]
                mon = dict(base)
                if region is not None:
                    mon = {
                        "left":   base["left"] + region["left"],
                        "top":    base["top"]  + region["top"],
                        "width":  region["width"],
                        "height": region["height"],
                    }
                sct.grab(mon)  # test grab – nếu X11 fail sẽ throw ở đây
                self._sct     = sct
                self._mon     = mon
                self._backend = "mss"
                self.width    = int(mon["width"])
                self.height   = int(mon["height"])
                print(f"   Backend: mss (X11)")
                print(f"   Monitor [{monitor_idx}]: {monitors[monitor_idx]}")
                print(f"   Capture region: {mon}")
                return
            except Exception as ex:
                print(f"   mss không hoạt động ({type(ex).__name__}: {ex})")
                print(f"   Chuyển sang PIL.ImageGrab (Wayland backend)...")

        # ── Fallback: PIL.ImageGrab ────────────────────────────────
        if not _PIL_AVAILABLE:
            raise RuntimeError(
                "Cả mss lẫn Pillow đều không khả dụng.\n"
                "Chạy: pip install mss   hoặc   pip install Pillow"
            )
        if region is not None:
            x, y, w, h = region["left"], region["top"], region["width"], region["height"]
            self._pil_bbox = (x, y, x + w, y + h)
            self.width  = w
            self.height = h
        else:
            self._pil_bbox = None
            test_img = _ImageGrab.grab()
            self.width, self.height = test_img.size

        self._backend = "pil"
        print(f"   Backend: PIL.ImageGrab (Wayland compatible)")
        print(f"   Capture bbox: {self._pil_bbox or 'full screen'}")
        print(f"   Size: {self.width}x{self.height}")

    def read(self):
        try:
            if self._backend == "mss":
                # Khởi tạo mss() trong chính luồng gọi read (thread-local) để tránh lỗi X11
                if not hasattr(self, "_thread_sct"):
                    from mss import mss
                    self._thread_sct = mss()
                raw = self._thread_sct.grab(self._mon)
                frame = cv2.cvtColor(np.array(raw), cv2.COLOR_BGRA2BGR)
            else:
                raw = _ImageGrab.grab(bbox=self._pil_bbox)
                frame = cv2.cvtColor(np.array(raw), cv2.COLOR_RGB2BGR)
            return True, frame
        except Exception as e:
            print(f"⚠️ Capture read error: {e}")
            return False, None

    def release(self):
        if self._sct is not None:
            self._sct.close()

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FPS:          return self.fps
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:  return float(self.width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT: return float(self.height)
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:  return 0.0
        return 0.0

    def isOpened(self):
        return self._backend is not None


# ─── Parse arguments ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time camera inference with two-stage OBB detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference_camera.py                              # webcam index 0
  python inference_camera.py --camera 1                   # webcam index 1
  python inference_camera.py --camera rtsp://192.168.1.10:554/stream  # RTSP
  python inference_camera.py --camera http://192.168.1.10:8080/video  # HTTP
  python inference_camera.py --save                       # lưu video
  python inference_camera.py --width 1280 --height 720    # set resolution
        """,
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="0",
        help="Camera source: integer index (0,1,...) hoặc URL (rtsp://, http://). Default: 0",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Set camera width (0 = giữ nguyên mặc định camera)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=0,
        help="Set camera height (0 = giữ nguyên mặc định camera)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Lưu video output ra file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Đường dẫn video output (default: camera_output_<timestamp>.mp4)",
    )
    parser.add_argument(
        "--detections",
        type=str,
        default="",
        help="Đường dẫn file detections JSONL output",
    )
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        default=["actions_config_refactored.yaml"],
        help="Một hoặc nhiều file YAML cấu hình action. Actions sẽ được ghép theo thứ tự truyền vào. "
             "Ví dụ: --config put_board.yaml take_component.yaml return_board.yaml"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Không hiển thị cửa sổ (headless mode, chỉ dùng khi --save)",
    )
    parser.add_argument(
        "--video",
        type=str,
        default="",
        help="Đường dẫn video file để test streaming pipeline (thay thế camera). Ví dụ: --video datasets/xb10_15.mp4",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Tốc độ phát lại khi dùng --video (1.0 = realtime, 2.0 = 2x, 0.5 = chậm lại). Default: 1.0",
    )
    parser.add_argument(
        "--screen",
        action="store_true",
        help="Chụp màn hình máy tính và chạy inference realtime (cần cài mss).",
    )
    parser.add_argument(
        "--monitor",
        type=int,
        default=1,
        help="Index monitor khi dùng --screen (1 = primary, 2 = secondary, ...). Default: 1",
    )
    parser.add_argument(
        "--region",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        default=None,
        help="Crop vùng màn hình khi dùng --screen: X Y W H (tính từ góc trên-trái của monitor). Ví dụ: --region 0 0 1280 720",
    )

    return parser.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Xác định chế độ nguồn: screen > video > camera
    screen_mode = bool(args.screen)
    video_mode  = bool(args.video) and not screen_mode

    if screen_mode:
        # ── Screen capture mode ──────────────────────────────────
        if not MSS_AVAILABLE:
            print("❌ Lỗi: Chưa cài thư viện 'mss'. Chạy: pip install mss")
            return
        region_dict = None
        if args.region:
            x, y, w, h = args.region
            region_dict = {"left": x, "top": y, "width": w, "height": h}
        print(f"🖥️  Chế độ SCREEN CAPTURE")
        print(f"   Monitor index: {args.monitor}")
        if region_dict:
            print(f"   Region (x,y,w,h): {args.region}")
        try:
            cap = ScreenCapture(monitor_idx=args.monitor, region=region_dict)
        except Exception as e:
            print(f"❌ Lỗi khởi tạo ScreenCapture: {e}")
            return

    elif video_mode:
        # ── Video file mode ──────────────────────────────────────
        source = str(Path(args.video).resolve())
        print(f"🎬 Chế độ VIDEO FILE: {source}")
        print(f"   Tốc độ phát: {args.speed}x")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"❌ Lỗi: Không thể mở video file: {source}")
            return

    else:
        # ── Camera mode ──────────────────────────────────────────
        camera_source = args.camera
        try:
            camera_source = int(camera_source)
        except ValueError:
            pass  # giữ nguyên string (rtsp://, http://)
        source = camera_source
        print(f"📷 Đang mở camera: {source} ...")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"❌ Lỗi: Không thể mở camera: {source}")
            print("Kiểm tra:")
            print("  - Camera có được kết nối không?")
            print("  - Index camera có đúng không? (thử --camera 0, --camera 1, ...)")
            print("  - Nếu dùng RTSP/HTTP, URL có đúng không?")
            return
        # Set resolution (chỉ áp dụng cho camera)
        if args.width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        if args.height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Đọc frame đầu tiên để lấy kích thước thực tế
    ret, test_frame = cap.read()
    if not ret:
        print("❌ Lỗi: Không thể đọc frame đầu tiên!")
        cap.release()
        return

    height, width = test_frame.shape[:2]
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not fps or fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    playback_speed = max(0.1, args.speed) if video_mode else 1.0
    frame_delay_s  = (1.0 / fps) / playback_speed if video_mode else 0.0

    if screen_mode:
        print(f"✅ Screen capture sẵn sàng: {width}x{height}")
    elif video_mode:
        print(f"✅ Video mở thành công: {width}x{height}, FPS={fps:.1f}, Tổng frames: {total_frames}")
    else:
        print(f"✅ Camera mở thành công: {width}x{height}, FPS={fps:.1f}")

    # Setup class names
    names_by_cls = {i: n for i, n in full_model.names.items()} if hasattr(full_model, "names") else {}

    # Setup video writer (nếu cần save)
    out_writer = None
    det_file = None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    WRITER_FPS = 30.0  # fps cố định khi ghi temp — sẽ fix sau
    if args.save:
        output_path = args.output if args.output else str(ROOT / f"camera_output_{timestamp}.mp4")
        tmp_output_path = output_path + ".tmp.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(tmp_output_path, fourcc, WRITER_FPS, (width, height))
        print(f"📹 Lưu video: {output_path}  (sẽ fix fps sau khi dừng stream)")
    else:
        output_path = ""
        tmp_output_path = ""

    if args.detections:
        det_path = args.detections
        det_file = open(det_path, "w", encoding="utf-8")
        print(f"📝 Lưu detections: {det_path}")

    # Setup StreamingEngine nếu có
    engine = None
    if HAS_ACTION_ENGINE and args.config:
        # args.config là list khi nargs="+", kiểm tra từng file trong list
        config_list = args.config if isinstance(args.config, list) else [args.config]
        missing = [p for p in config_list if not Path(p).exists()]
        if missing:
            print(f"⚠️  Không tìm thấy file config: {missing}")
        else:
            engine = StreamingActionEngine(config_list)
            label = ", ".join(config_list)
            print(f"⚙️  Khởi tạo StreamingActionEngine (configs: {label})")

    # Bắt đầu luồng capture bất đồng bộ (Chỉ dùng cho camera/video để tránh lỗi X11)
    vget = None
    if not screen_mode and not video_mode:
        vget = VideoGet(cap, stop_on_failure=video_mode).start()

    # Setup display window
    window_name = "YOLOv8 Real-time Detection"
    is_fullscreen = False
    prev_time = time.time()
    stream_start_time = time.time()
    fps_smooth = 0.0
    frame_idx = 0
    last_f_count = -1

    if not args.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(width, 1280), min(height, 720))

    print("\n" + "=" * 60)
    if screen_mode:
        print("  SCREEN CAPTURE INFERENCE")
        print(f"  Monitor: {args.monitor}  Region: {args.region or 'full'}")
        print("  Controls: [q/ESC] Thoát | [s] Screenshot | [f] Fullscreen")
    elif video_mode:
        print("  VIDEO FILE STREAMING TEST")
        print(f"  Source: {args.video}")
        print(f"  Controls: [q/ESC] Thoát | [s] Screenshot | [f] Fullscreen | [+] Nhanh hơn | [-] Chậm hơn")
    else:
        print("  REAL-TIME CAMERA INFERENCE")
        print("  Controls: [q/ESC] Thoát | [s] Screenshot | [f] Fullscreen")
    print("=" * 60 + "\n")


    # ── Signal handler: bắt Ctrl+C dừng stream graceful ──────────────
    _stop = False
    def _handle_sigint(sig, frame):
        nonlocal _stop
        print("\n⚠️  Ctrl+C nhận được — đang dừng stream...")
        _stop = True
    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        while not _stop:
            # 1. Đọc từ capture (threaded cho camera, đồng bộ cho screen)
            if vget is not None:
                ret, frame, f_count, _ = vget.read()
                if not ret or f_count <= last_f_count:
                    if not args.no_display:
                        cv2.waitKey(1)
                    time.sleep(0.005)
                    continue
            else:
                ret, frame = cap.read()
                if not ret:
                    if video_mode:
                        print("\n🏁 Đã đọc hết video input (EOF).")
                        break
                    if not args.no_display:
                        cv2.waitKey(1)
                    time.sleep(0.005)
                    continue
                f_count = last_f_count + 1

            if f_count <= last_f_count:
                if not args.no_display:
                    cv2.waitKey(1)
                time.sleep(0.005)
                continue

            # 2. Xử lý AI khi có frame mới
            last_f_count = f_count
            frame_idx += 1

            # Tính FPS xử lý AI
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            instant_fps = 1.0 / max(dt, 1e-6)
            fps_smooth = 0.9 * fps_smooth + 0.1 * instant_fps if fps_smooth > 0 else instant_fps

            # Inference Stage 1 & 2
            detections = run_two_stage_inference(frame, width, height, names_by_cls)

            # Vẽ annotated frame
            annotated_frame = draw_polys_smart(frame, detections, names_by_cls)
            annotated_frame = draw_fps_overlay(annotated_frame, fps_smooth, frame_idx)


            # Cập nhật Engine (nếu có) và vẽ overlay
            if engine is not None:
                engine_dets = []
                for d in detections:
                    engine_dets.append(Detection(
                        cls=names_by_cls.get(int(d["cls_id"]), str(d["cls_id"])),
                        conf=float(d.get("conf", 0.0)),
                        poly=d["poly"]
                    ))
                engine_state = engine.process_frame(frame_idx, engine_dets)
                
                # Vẽ bảng Overlay
                lines = []
                history = engine_state.get("history", [])
                now = engine_state.get("now", time.time())

                for row in history[-3:]:
                    aid = row["action_id"]
                    t_start = row.get("start_time")
                    t_end = row.get("end_time")
                    elapsed = (t_end - t_start) if (t_start and t_end) else 0.0
                    lines.append(f"[{row['outcome'].upper()}] {aid}  {elapsed:.1f}s")
                    
                status = engine_state.get("status")
                if status == "done":
                    lines.append("> ALL COMPLETED!")
                    
                    try:
                        out_dir = Path("runs/camera_sessions")
                        out_dir.mkdir(parents=True, exist_ok=True)
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        csv_path = out_dir / f"session_{ts}.csv"
                        with open(csv_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=["action_id", "start_time", "end_time", "outcome", "end_reason"])
                            writer.writeheader()
                            writer.writerows(history)
                        print(f"\n[INFO] Đã hoàn thành 1 chu trình. Lưu kết quả CSV: {csv_path}")
                        
                        engine.reset()
                    except Exception as e:
                        print(f"Lỗi khi save CSV hoặc reset: {e}")
                else:
                    aid = engine_state.get("action_id", "Unknown")
                    sf = engine_state.get("start_time")
                    elapsed = (now - sf) if sf is not None else 0.0
                    pidx = engine_state.get("phase_idx", 0)
                    ptot = engine_state.get("total_phases", 0)
                    lines.append(f"> {aid} [P:{pidx}/{ptot}]  {elapsed:.1f}s")

                # render box overlays
                font, sc, th, pad = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1, 8
                x0, y0 = 10, 80
                line_h = 22
                
                box_w = 0
                for txt in lines:
                    (tw, _), _ = cv2.getTextSize(txt, font, sc, th)
                    box_w = max(box_w, tw)
                box_w += pad * 2
                box_h = len(lines) * line_h + pad * 2

                cv2.rectangle(annotated_frame, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), thickness=-1)
                
                y_curr = y0 + pad + 16
                for txt in lines:
                    color = (0, 255, 255) if txt.startswith(">") else (0, 255, 0) if "COMPLETED" in txt else (200, 200, 200)
                    cv2.putText(annotated_frame, txt, (x0 + pad, y_curr), font, sc, color, th, cv2.LINE_AA)
                    y_curr += line_h

            # Lưu video nếu cần
            if out_writer is not None:
                out_writer.write(annotated_frame)

            # Lưu detections nếu cần
            if det_file is not None:
                frame_payload = {"frame": frame_idx, "detections": []}
                for d in detections:
                    cls_id = int(d["cls_id"])
                    poly = d["poly"]
                    frame_payload["detections"].append({
                        "class": names_by_cls.get(cls_id, str(cls_id)),
                        "cls_id": cls_id,
                        "conf": float(d.get("conf", 0.0)),
                        "track_id": d.get("track_id", None),
                        "poly": [[float(x), float(y)] for x, y in poly.tolist()],
                    })
                det_file.write(json.dumps(frame_payload, ensure_ascii=False) + "\n")

            # Hiển thị
            if not args.no_display:
                cv2.imshow(window_name, annotated_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # q hoặc ESC
                    print("\n👋 Thoát theo yêu cầu người dùng.")
                    break
                elif key == ord('s'):  # screenshot
                    screenshot_path = str(ROOT / f"screenshot_{timestamp}_{frame_idx}.jpg")
                    cv2.imwrite(screenshot_path, annotated_frame)
                    print(f"📸 Screenshot saved: {screenshot_path}")
                elif key == ord('f'):  # fullscreen toggle
                    is_fullscreen = not is_fullscreen
                    if is_fullscreen:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                elif video_mode and key == ord('+'):  # tăng tốc
                    playback_speed = min(playback_speed * 1.5, 16.0)
                    frame_delay_s = (1.0 / fps) / playback_speed
                    print(f"⏩ Tốc độ phát: {playback_speed:.2f}x")
                elif video_mode and key == ord('-'):  # giảm tốc
                    playback_speed = max(playback_speed / 1.5, 0.1)
                    frame_delay_s = (1.0 / fps) / playback_speed
                    print(f"⏪ Tốc độ phát: {playback_speed:.2f}x")

            # Log console (chỉ in khi có detection quan trọng)
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

    finally:
        # Cleanup — luôn chạy dù thoát bằng q, ESC, hay Ctrl+C
        if vget is not None:
            vget.stop()
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        cap.release()
        stream_elapsed = time.time() - stream_start_time

        if out_writer is not None:
            out_writer.release()
            actual_fps = frame_idx / max(stream_elapsed, 1e-3)
            ratio = actual_fps / WRITER_FPS
            print(f"\n⏱️  Stream thực tế: {stream_elapsed:.1f}s | {frame_idx} frames | {actual_fps:.2f} fps")

            if abs(ratio - 1.0) > 0.1:  # lệch >10% → re-encode với fps đúng
                print(f"🔄 Re-encoding với fps đúng ({actual_fps:.2f} fps)...")
                ok = _reencode_video_fps(tmp_output_path, output_path, actual_fps)
                import os
                try:
                    os.remove(tmp_output_path)
                except Exception:
                    pass
                if ok:
                    print(f"✅ Video đã được lưu (đúng tốc độ thực tế): {output_path}")
                else:
                    print(f"⚠️  Re-encode thất bại. File gốc giữ lại tại: {tmp_output_path}")
            else:
                import shutil
                shutil.move(tmp_output_path, output_path)
                print(f"✅ Video đã lưu: {output_path}")

        if det_file is not None:
            det_file.close()
            print("✅ Detections đã lưu.")
        cv2.destroyAllWindows()
        print(f"✅ Hoàn tất! Tổng {frame_idx} frames đã xử lý.")


if __name__ == "__main__":
    main()
