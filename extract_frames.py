import cv2
import os

def extract_frames(video_path, output_dir, target_fps=3):
    # 1. Tạo thư mục chứa ảnh đầu ra nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)

    # 2. Mở file video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Lỗi: Không thể mở video tại đường dẫn '{video_path}'")
        print("💡 Hãy kiểm tra lại xem file video đã được copy vào đúng vị trí chưa nhé.")
        return

    # 3. Lấy thông số FPS gốc của video
    original_fps = round(cap.get(cv2.CAP_PROP_FPS))
    print(f"🎬 FPS gốc của video: {original_fps}")

    # 4. Tính toán chu kỳ lấy frame 
    # (Ví dụ: Video 30fps, muốn lấy 3fps -> Lấy 1 frame, bỏ qua 9 frames)
    frame_interval = max(1, original_fps // target_fps)
    print(f"⏱️ Sẽ lấy 1 ảnh sau mỗi {frame_interval} frames (Tương đương ~{target_fps} ảnh/giây)")

    pic = 2
    frame_count = 0
    saved_count = 0

    print("Đang tiến hành trích xuất...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break # Hết video

        # Chỉ lưu frame nếu số thứ tự của nó chia hết cho khoảng cách đã tính
        if frame_count % frame_interval == 0:
            # Tạo tên file có padding số 0 (VD: frame_0000.jpg, frame_0010.jpg)
            filename = os.path.join(output_dir, f"frame_{pic}_{frame_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"✅ Hoàn tất! Tổng số ảnh thu được: {saved_count} ảnh.")
    print(f"📁 Dữ liệu đã được lưu tại: {output_dir}")

if __name__ == "__main__":
    # --- CẤU HÌNH ĐƯỜNG DẪN DỰA THEO PROJECT CỦA BẠN ---
    
    # 1. Bạn copy file video quay thao tác vào thư mục datasets và đổi tên file ở dưới
    VIDEO_FILE = "datasets/video_thao_tac_2.mp4" 
    
    # 2. Thư mục chứa ảnh sau khi cắt
    OUTPUT_FOLDER = "datasets/extracted_frames"
    
    # 3. Chạy hàm trích xuất
    extract_frames(VIDEO_FILE, OUTPUT_FOLDER, target_fps=3)