import os
import glob
import random
import cv2
import numpy as np
from collections import defaultdict

DATASET_DIR = 'datasets/obj_train_data'
SMALL_CLASSES = {6, 7, 8}  # 6: shielding, 7: PCIe cable, 8: gasket
TRAY_CLASS = 1
HAND_CLASS = 0
KEEP_N_IN_TRAY = 3  # Số lượng box ngẫu nhiên giữ lại trên mỗi tray cho 1 class (ở đây chọn 3)

def get_center(coords):
    xs = [coords[i] for i in range(0, 8, 2)]
    ys = [coords[i] for i in range(1, 8, 2)]
    return sum(xs)/4.0, sum(ys)/4.0

def point_in_poly(x, y, poly_coords):
    # Dùng OpenCV để kiểm tra xem tọa độ tâm x,y có nằm trong đa giác OBB không
    pts = np.array([[poly_coords[i], poly_coords[i+1]] for i in range(0, 8, 2)], np.float32)
    return cv2.pointPolygonTest(pts, (x, y), False) >= 0

def boxes_intersect_axis_aligned(coords1, coords2):
    # Kiểm tra nhanh xem 2 hộp bao trục có chạm/đè lên nhau không (dùng cho vùng hand)
    xs1 = [coords1[i] for i in range(0, 8, 2)]
    ys1 = [coords1[i] for i in range(1, 8, 2)]
    xs2 = [coords2[i] for i in range(0, 8, 2)]
    ys2 = [coords2[i] for i in range(1, 8, 2)]
    
    xmin1, xmax1 = min(xs1), max(xs1)
    ymin1, ymax1 = min(ys1), max(ys1)
    xmin2, xmax2 = min(xs2), max(xs2)
    ymin2, ymax2 = min(ys2), max(ys2)
    
    # Mở rộng nhẹ biên độ 5% để bắt dính tay 
    margin = 0.05
    xmax1 += margin; xmin1 -= margin
    ymax1 += margin; ymin1 -= margin
    
    return not (xmax1 < xmin2 or xmin1 > xmax2 or ymax1 < ymin2 or ymin1 > ymax2)

def process_label_file(filepath):
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        
    if not lines:
        return 0, 0
    
    parsed = []
    for line in lines:
        parts = line.split()
        cls_id = int(parts[0])
        coords = [float(x) for x in parts[1:9]]
        parsed.append({'class': cls_id, 'coords': coords, 'line': line, 'keep': False})
        
    # Tập hợp các vùng quan trọng
    trays = [p['coords'] for p in parsed if p['class'] == TRAY_CLASS]
    hands = [p['coords'] for p in parsed if p['class'] == HAND_CLASS]
    
    # Tập hợp các box rác cần lọc
    seen_first_of_class = set()
    
    # 1. Luôn ưu tiên giữ lại các classes TO/QUAN TRỌNG (hand, tray, board, jig, liner, tweezers)
    for p in parsed:
        cls = p['class']
        if cls not in SMALL_CLASSES:
            p['keep'] = True
            
        else: # Nếu là class bé (shielding, gasket...)
            keep = False
            
            # Ưu tiên số 1: ID xuất hiện đầu tiên của dòng họ nó (theo yêu cầu user)
            if cls not in seen_first_of_class:
                seen_first_of_class.add(cls)
                keep = True
                
            # Ưu tiên số 2: Đang được tay (hand) cầm nắm hoặc quẹt ngang qua
            if not keep:
                for hand_coords in hands:
                    if boxes_intersect_axis_aligned(p['coords'], hand_coords):
                        keep = True
                        break
            
            # Ưu tiên số 3: Nằm NGOÀI khay chứa (chắc chắn đang nằm trên vỉ mạch/board)
            if not keep:
                cx, cy = get_center(p['coords'])
                in_tray = False
                for tray_coords in trays:
                    if point_in_poly(cx, cy, tray_coords):
                        in_tray = True
                        break
                
                if not in_tray:
                    keep = True
                    
            p['keep'] = keep

    # Giờ đối phó với đội ngũ siêu tốn không gian nằm gọn trong tray
    tray_candidates = defaultdict(list)
    for i, p in enumerate(parsed):
        if p['class'] in SMALL_CLASSES and not p['keep']:
            # Nó chắc chắn nằm trong tray vì đã bị filter màng trên
            tray_candidates[p['class']].append(i)
            
    # Giữ lại ngẫu nhiên 3 em "may mắn" đại diện cho tray, loại bỏ toàn bộ lũ còn lại
    for cls, indices in tray_candidates.items():
        if len(indices) <= KEEP_N_IN_TRAY:
            for idx in indices:
                parsed[idx]['keep'] = True
        else:
            lucky_winners = random.sample(indices, KEEP_N_IN_TRAY)
            for idx in lucky_winners:
                parsed[idx]['keep'] = True
                
    # Ghi đè file
    kept_lines = [p['line'] for p in parsed if p['keep']]
    
    with open(filepath, 'w') as f:
        f.write("\n".join(kept_lines) + "\n")
        
    return len(lines), len(kept_lines)

if __name__ == '__main__':
    label_files = glob.glob(os.path.join(DATASET_DIR, '*.txt'))
    
    total_original = 0
    total_kept = 0
    
    for lf in label_files:
        orig, kept = process_label_file(lf)
        total_original += orig
        total_kept += kept
        
    print(f"\n✅ ĐÃ DỌN DẸP XONG TẬP LABEL!")
    print(f"➜ Tổng số Box ban đầu : {total_original}")
    print(f"➜ Tổng số Box giữ lại : {total_kept}")
    print(f"🪓 Đã loại bỏ hoàn toàn : {total_original - total_kept} boxes thừa trong Khay chứa.")
