import cv2
import os
from ultralytics import YOLO

model = YOLO('runs/detect/action_model_v1/weights/best.pt') 
cap = cv2.VideoCapture('datasets/video_thao_tac_raw.mp4')

# Tạo thư mục chứa các frame khó để đem đi gán nhãn
os.makedirs('datasets/hard_frames', exist_ok=True)

frame_count = 0
saved_count = 0

print("Đang đi săn các frame lỗi...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame_count += 1

    # Bỏ qua bớt frame cho nhẹ (1 giây chỉ check 2 frame)
    if frame_count % 15 != 0: 
        continue

    results = model(frame, verbose=False)[0]
    
    is_hard_frame = False
    hand_detected = False

    for box in results.boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        
        if cls_id == 0: hand_detected = True
            
        # BẪY 1: Phát hiện vật thể nhưng độ tự tin thấp (từ 25% đến 60%)
        # (Chứng tỏ AI đang phân vân, bị nhòe)
        if 0.25 < conf < 0.60:
            is_hard_frame = True

    # BẪY 2: Nếu trong video luân phiên thao tác mà bỗng nhiên AI không thấy cái tay nào
    if not hand_detected:
        is_hard_frame = True

    # Nếu dính bẫy -> Lưu ảnh lại ngay lập tức
    if is_hard_frame:
        save_path = f"datasets/hard_frames/hard_frame_{frame_count}.jpg"
        cv2.imwrite(save_path, frame)
        saved_count += 1
        print(f"Bắt được 1 frame lỗi tại frame {frame_count}!")

cap.release()
print(f"✅ Đã săn được {saved_count} frame khó. Hãy mang chúng vào LabelImg gán nhãn!")