from ultralytics import YOLO

model = YOLO('yolov8s-obb.pt')

print("Bắt đầu train mô hình...")

# --- Train mô hình full-frame (model gốc) ---
# results = model.train(
#     data='data.yaml',
#     epochs=100,
#     patience=25,
#     imgsz=1280,
#     batch=4,
#     device=0,
#     name='action_model_v6',
#     mosaic=1.0,
#     copy_paste=0.2,
#     mixup=0.1,
#     degrees=10.0,
#     lr0=0.01,
#     workers=4,
#     cache=False
# )

# --- Train mô hình ROI (in-hand) ---
results = model.train(
    data='datasets/roi_format_full/data_roi.yaml',
    epochs=100,
    patience=25,
    imgsz=640,
    batch=16,
    device=0,
    name='action_model_roi_v1',
    mosaic=1.0,
    copy_paste=0.2,
    mixup=0.1,
    degrees=10.0,
    lr0=0.01,
    workers=4,
    cache=False
)

print("Hoàn tất huấn luyện!")