from ultralytics import YOLO

# ====================================================================================
# HƯỚNG DẪN: Bỏ comment (#) ở phần bạn muốn train và comment lại các phần khác.
# Chỉ nên để mở 1 phần duy nhất tại một thời điểm.
# ====================================================================================

# 1️⃣ TRAIN MÔ HÌNH FULL-FRAME OBB (Phát hiện vật thể toàn khung hình)
# ------------------------------------------------------------------------------------
# model = YOLO('yolov8s-obb.pt')
# results = model.train(
#     data='data.yaml',
#     epochs=100,
#     patience=25,
#     imgsz=1280,
#     batch=4,
#     device=0,
#     name='obb_full_frame_v1',
#     mosaic=1.0,
#     copy_paste=0.2,
#     mixup=0.1,
#     degrees=10.0,
#     lr0=0.01,
#     workers=4,
#     cache=True
# )


# 2️⃣ TRAIN MÔ HÌNH ROI OBB (Phát hiện linh kiện nhỏ trong vùng cắt bàn tay)
# ------------------------------------------------------------------------------------
model = YOLO('yolov8s-obb.pt')
results = model.train(
    data='datasets/roi_format/data_roi.yaml',
    epochs=100,
    patience=25,
    imgsz=640,
    batch=16,
    device=0,
    name='obb_roi_v1',
    mosaic=1.0,
    copy_paste=0.3,
    mixup=0.1,
    degrees=15.0,
    lr0=0.01,
    workers=4,
    cache=True
)


# 3️⃣ TRAIN MÔ HÌNH POSE (Phát hiện các điểm khớp bàn tay - Keypoints)
# ------------------------------------------------------------------------------------
# model = YOLO('yolov8s-pose.pt')
# results = model.train(
#     data='datasets/pose_format/data_pose.yaml',
#     epochs=150,
#     patience=50,
#     imgsz=416,           # Tối ưu cho ảnh bàn tay nhỏ (390x375)
#     batch=24,            # Phù hợp GPU 11GB VRAM
#     device=0,
#     name='pose_hand_v7',
#     lr0=0.001,           # Fine-tune để không làm hỏng trọng số pretrained
#     mosaic=0.5,
#     close_mosaic=15,     # Tắt mosaic cuối kỳ để hội tụ tốt hơn
#     degrees=45.0,
#     flipud=0.5,
#     fliplr=0.5,
#     scale=0.5,
#     erasing=0.4,         # Chống overfitting khi bị che khuất
#     pose=20.0,           # Tập trung cao vào độ chính xác keypoint
#     rect=True,           # Giữ tỉ lệ ảnh gốc, giảm padding
#     workers=4,
#     cache=True,
# )

print("✅ Hoàn tất quá trình huấn luyện!")