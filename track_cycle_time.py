import cv2
import csv
from ultralytics import YOLO
from collections import deque

model = YOLO('runs/detect/action_model_v6/weights/best.pt') 

video_path = 'datasets/video_thao_tac_raw.mp4'
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('cycle_time_1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# --- HÀM TÍNH GIAO CẮT THÔNG MINH ---
def calculate_ioa(hand_box, target_box):
    x1_min, y1_min, x1_max, y1_max = hand_box
    x2_min, y2_min, x2_max, y2_max = target_box
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
        return 0.0
        
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    hand_area = (x1_max - x1_min) * (y1_max - y1_min)
    target_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    if hand_area == 0 or target_area == 0: return 0
    # Lấy tỷ lệ lớn nhất giữa (Phần giao / Tay) hoặc (Phần giao / Vật thể)
    return max(inter_area / hand_area, inter_area / target_area)

print("Đang xử lý video bằng Thuật toán Pure Logic... Vui lòng đợi!")

current_state = "TRANG THAI: DANG CHO"
previous_state = "TRANG THAI: DANG CHO"
cycle_start_frame = 0
cycle_count = 0
sop_data = [] 
frame_count = 0

# Biến nhớ (Cache) cho các vật thể tĩnh
cached_tray = None
cached_mold = None

# Bộ đệm trạng thái (Smoothing)
STATE_BUFFER_SIZE = 5
state_history = deque(maxlen=STATE_BUFFER_SIZE)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame_count += 1
    results = model(frame, verbose=False)[0]
    annotated_frame = results.plot()

    hands, boards, trays, molds = [], [], [], []
    for box in results.boxes:
        coords = box.xyxy[0].tolist() 
        class_name = model.names[int(box.cls[0])] 
        if class_name == 'board': boards.append(coords) 
        elif class_name == 'hand': hands.append(coords)    
        elif class_name == 'mold': molds.append(coords) 
        elif class_name == 'tray': trays.append(coords)

    # --- CẬP NHẬT TRÍ NHỚ ĐỘNG (CACHE) ---
    if len(trays) > 0: cached_tray = trays[0]
    if len(molds) > 0: cached_mold = molds[0]

    # Nếu AI bị mù (che khuất), lôi trí nhớ ra dùng
    effective_trays = trays if len(trays) > 0 else ([cached_tray] if cached_tray else [])
    effective_molds = molds if len(molds) > 0 else ([cached_mold] if cached_mold else [])

    # Vẽ khung bộ nhớ (Màu xám) để bạn thấy code đang tự nội suy vật thể bị che
    if len(molds) == 0 and cached_mold:
        cv2.rectangle(annotated_frame, (int(cached_mold[0]), int(cached_mold[1])), 
                     (int(cached_mold[2]), int(cached_mold[3])), (150, 150, 150), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, "Mold (Cached)", (int(cached_mold[0]), int(cached_mold[1])-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # --- TÍNH TOÁN LOGIC GIAO CẮT ---
    raw_state = "TRANG THAI: DANG CHO"
    max_ioa_tray = 0.0
    max_ioa_board = 0.0
    max_ioa_mold = 0.0
    
    for hand_box in hands:
        # So với Khay (Dùng danh sách effective)
        for tray_box in effective_trays:
            ioa = calculate_ioa(hand_box, tray_box)
            if ioa > max_ioa_tray: max_ioa_tray = ioa
            
        # So với Mạch (Mạch có thể di chuyển nên không dùng cache)
        for board_box in boards:
            ioa = calculate_ioa(hand_box, board_box)
            if ioa > max_ioa_board: max_ioa_board = ioa
            
        # So với Khuôn (Dùng danh sách effective)
        for mold_box in effective_molds:
            ioa = calculate_ioa(hand_box, mold_box)
            if ioa > max_ioa_mold: max_ioa_mold = ioa

    # SO KÈO: Bên nào tỷ lệ đè cao nhất và vượt ngưỡng (> 15%) thì thắng
    threshold = 0.15
    if max_ioa_mold >= threshold and max_ioa_mold >= max_ioa_tray and max_ioa_mold >= max_ioa_board:
        raw_state = "TRANG THAI: DANG DAP KHUON"
    elif max_ioa_tray >= threshold and max_ioa_tray >= max_ioa_board and max_ioa_tray >= max_ioa_mold:
        raw_state = "TRANG THAI: DANG LAY KIEN"
    elif max_ioa_board >= threshold and max_ioa_board >= max_ioa_tray and max_ioa_board >= max_ioa_mold:
        raw_state = "TRANG THAI: DANG CAM KIEN"

    # --- BỘ LỌC CHỐNG NHÁY (DEBOUNCING) ---
    state_history.append(raw_state)
    
    # Chỉ đổi trạng thái nếu trạng thái mới chiếm ĐA SỐ (>= 3/5 frame gần nhất)
    # Điều này giúp trạng thái chuyển mượt mà nhưng vẫn rất nhạy bén
    if state_history.count(raw_state) >= 3:
        status = raw_state
    else:
        status = current_state

    # Đổ màu Text
    if status == "TRANG THAI: DANG LAY KIEN": color = (0, 0, 255) 
    elif status == "TRANG THAI: DANG CAM KIEN": color = (255, 0, 0) 
    elif status == "TRANG THAI: DANG DAP KHUON": color = (0, 255, 255) 
    else: color = (0, 255, 0) 

    current_state = status 

    # --- LOGIC ĐẾM CYCLE TIME ---
    if current_state == "TRANG THAI: DANG LAY KIEN" and previous_state != "TRANG THAI: DANG LAY KIEN":
        cycle_start_frame = frame_count

    # Chu trình chốt khi Dập Khuôn xong và rút tay ra (Về chờ hoặc lấy kiện mới)
    elif previous_state == "TRANG THAI: DANG DAP KHUON" and (current_state == "TRANG THAI: DANG CHO" or current_state == "TRANG THAI: DANG LAY KIEN"):
        if cycle_start_frame > 0: 
            cycle_count += 1
            cycle_end_frame = frame_count
            cycle_time_seconds = round((cycle_end_frame - cycle_start_frame) / fps, 2)
            sop_data.append({
                "Chu trinh": cycle_count, "Frame bat dau": cycle_start_frame,
                "Frame ket thuc": cycle_end_frame, "Cycle Time (Giay)": cycle_time_seconds
            })
            cycle_start_frame = 0 
            # Đề phòng trường hợp chuyển thẳng sang lấy kiện
            if current_state == "TRANG THAI: DANG LAY KIEN":
                cycle_start_frame = frame_count

    previous_state = current_state

    # Ghi log ra màn hình video
    cv2.putText(annotated_frame, status, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
    out.write(annotated_frame)

cap.release()
out.release()

csv_file = "SOP_Cycle_Time_Report.csv"
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=["Chu trinh", "Frame bat dau", "Frame ket thuc", "Cycle Time (Giay)"])
    writer.writeheader()
    writer.writerows(sop_data)

print(f"📊 XONG! Đã xuất báo cáo SOP tại file: {csv_file}")