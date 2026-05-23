from ultralytics import YOLO

print("Bắt đầu train mô hình...")

# ============================================================
# --- Train YOLO-Pose (hand keypoints) ---
# ============================================================
pose_model = YOLO('yolov8s-pose.pt')

results = pose_model.train(
    data='datasets/pose_format/data_pose.yaml',
    epochs=150,
    patience=50,
    imgsz=416,           # Sát mean 390×375, không lãng phí như 640
    batch=24,            # Vừa VRAM 2080Ti 11GB
    device=0,
    name='pose_hand_v6',
    lr0=0.001,           # Fine-tune, không phá pretrained weight
    mosaic=0.5,
    close_mosaic=15,     # Tắt mosaic 15 epoch cuối để tinh chỉnh
    degrees=45.0,
    flipud=0.5,
    fliplr=0.5,
    scale=0.5,
    erasing=0.4,         # Giúp model quen với occlusion (43% keypoint bị che)
    pose=20.0,           # Nhấn mạnh keypoint accuracy
    rect=True,           # Giữ aspect ratio gốc, giảm padding vô ích
    workers=4,
    cache=True,
)

# ============================================================
# --- Train mô hình full-frame OBB (model gốc) ---
# ============================================================
# model = YOLO('yolov8s-obb.pt')
# results = model.train(
#     data='data.yaml',
#     epochs=100,
#     patience=25,
#     imgsz=1280,
#     batch=4,
#     device=0,
#     name='action_model_v10',
#     mosaic=1.0,
#     copy_paste=0.2,
#     mixup=0.1,
#     degrees=10.0,
#     lr0=0.01,
#     workers=4,
#     cache=False
# )

# ============================================================
# --- Train mô hình ROI OBB (in-hand) ---
# ============================================================
# results = model.train(
#     data='datasets/roi_format/data_roi.yaml',
#     epochs=100,
#     patience=25,
#     imgsz=640,
#     batch=16,
#     device=0,
#     name='action_model_roi_v5',
#     mosaic=1.0,
#     copy_paste=0.2,
#     mixup=0.1,
#     degrees=10.0,
#     lr0=0.01,
#     workers=4,
#     cache=False
# )

print("Hoàn tất huấn luyện!")