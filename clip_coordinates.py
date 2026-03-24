"""
Script clip tọa độ OBB label về khoảng [0, 1] và tạo ảnh so sánh trước/sau.

Chức năng:
1. Vẽ ảnh mẫu với OBB boxes TRƯỚC khi clip
2. Clip tất cả tọa độ trong obj_train_data về [0, 1]
3. Vẽ ảnh mẫu với OBB boxes SAU khi clip
4. Lưu 2 ảnh so sánh
"""

import os
import glob
import cv2
import numpy as np
from copy import deepcopy

# ======== CẤU HÌNH ========
DATASET_DIR = 'datasets/obj_train_data'
OUTPUT_DIR = 'datasets'
# File mẫu có tọa độ lệch nhiều nhất (1.166871)
SAMPLE_IMAGE = 'xb10_11_000350.png'
SAMPLE_LABEL = 'xb10_11_000350.txt'

CLASS_NAMES = ['hand', 'tray', 'board', 'jig', 'liner',
               'tweezers', 'shielding', 'PCIe cable', 'gasket']

# Bảng màu cho từng class (BGR)
CLASS_COLORS = [
    (0, 255, 0),      # 0: hand - xanh lá
    (255, 0, 0),      # 1: tray - xanh dương
    (0, 0, 255),      # 2: board - đỏ
    (255, 255, 0),    # 3: jig - cyan
    (0, 255, 255),    # 4: liner - vàng
    (255, 0, 255),    # 5: PCIe cable - hồng
    (128, 255, 128),  # 6: tweezers - xanh lá nhạt
    (255, 128, 0),    # 7: shielding - cam
    (128, 0, 255),    # 8: gasket - tím
]


def read_labels(label_path):
    """Đọc file label OBB, trả về list các annotation."""
    annotations = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 9:
                continue
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            annotations.append({'class_id': class_id, 'coords': coords})
    return annotations


def clip_annotations(annotations):
    """Clip tất cả tọa độ về [0, 1]."""
    clipped = deepcopy(annotations)
    for ann in clipped:
        ann['coords'] = [max(0.0, min(1.0, c)) for c in ann['coords']]
    return clipped


def write_labels(label_path, annotations):
    """Ghi lại file label sau khi clip."""
    with open(label_path, 'w') as f:
        for ann in annotations:
            coords_str = ' '.join(f'{c:.6f}' for c in ann['coords'])
            f.write(f"{ann['class_id']} {coords_str}\n")


def draw_obb_boxes(image, annotations, img_h, img_w, title=""):
    """Vẽ OBB boxes lên ảnh, highlight box nào bị lệch."""
    canvas = image.copy()
    
    for ann in annotations:
        cls_id = ann['class_id']
        coords = ann['coords']
        color = CLASS_COLORS[cls_id]
        
        # Kiểm tra có tọa độ nào ngoài [0,1] không
        is_out_of_bounds = any(c < 0 or c > 1 for c in coords)
        
        # Chuyển tọa độ normalized → pixel
        points = []
        for i in range(0, 8, 2):
            px = int(coords[i] * img_w)
            py = int(coords[i + 1] * img_h)
            points.append([px, py])
        points = np.array(points, dtype=np.int32)
        
        # Vẽ polygon
        thickness = 3 if is_out_of_bounds else 2
        if is_out_of_bounds:
            # Vẽ viền đỏ dày cho box bị lệch
            cv2.polylines(canvas, [points], isClosed=True, color=(0, 0, 255), thickness=4)
        cv2.polylines(canvas, [points], isClosed=True, color=color, thickness=thickness)
        
        # Vẽ label text
        label = CLASS_NAMES[cls_id]
        text_x = min(p[0] for p in points)
        text_y = min(p[1] for p in points) - 8
        text_y = max(20, text_y)  # Đảm bảo text không bị cắt ở trên
        
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (text_x, text_y - th - 4), (text_x + tw + 4, text_y + 2), color, -1)
        cv2.putText(canvas, label, (text_x + 2, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Vẽ title
    if title:
        cv2.rectangle(canvas, (0, 0), (len(title) * 18 + 20, 45), (0, 0, 0), -1)
        cv2.putText(canvas, title, (10, 32), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    
    return canvas


def create_comparison_image():
    """Tạo ảnh so sánh trước/sau clip."""
    img_path = os.path.join(DATASET_DIR, SAMPLE_IMAGE)
    lbl_path = os.path.join(DATASET_DIR, SAMPLE_LABEL)
    
    image = cv2.imread(img_path)
    if image is None:
        print(f"LỖI: Không đọc được ảnh {img_path}")
        return
    
    img_h, img_w = image.shape[:2]
    print(f"Ảnh mẫu: {SAMPLE_IMAGE} ({img_w}x{img_h})")
    
    # Đọc label gốc
    original_anns = read_labels(lbl_path)
    clipped_anns = clip_annotations(original_anns)
    
    # Đếm annotation bị ảnh hưởng
    out_count = sum(1 for ann in original_anns if any(c < 0 or c > 1 for c in ann['coords']))
    print(f"  Tổng annotations: {len(original_anns)}, bị lệch: {out_count}")
    
    # Vẽ TRƯỚC clip
    before_img = draw_obb_boxes(image, original_anns, img_h, img_w, 
                                f"BEFORE CLIP ({out_count} boxes out of bounds)")
    
    # Vẽ SAU clip
    after_img = draw_obb_boxes(image, clipped_anns, img_h, img_w, 
                               "AFTER CLIP (all coords in [0,1])")
    
    # Ghép 2 ảnh ngang nhau
    # Resize cho vừa nếu ảnh quá lớn
    max_w = 960
    if img_w > max_w:
        scale = max_w / img_w
        new_w = max_w
        new_h = int(img_h * scale)
        before_img = cv2.resize(before_img, (new_w, new_h))
        after_img = cv2.resize(after_img, (new_w, new_h))
    
    # Tạo đường phân cách
    separator = np.ones((before_img.shape[0], 4, 3), dtype=np.uint8) * 255
    
    comparison = np.hstack([before_img, separator, after_img])
    
    # Lưu
    before_path = os.path.join(OUTPUT_DIR, 'before_clip.png')
    after_path = os.path.join(OUTPUT_DIR, 'after_clip.png')
    comparison_path = os.path.join(OUTPUT_DIR, 'clip_comparison.png')
    
    cv2.imwrite(before_path, before_img)
    cv2.imwrite(after_path, after_img)
    cv2.imwrite(comparison_path, comparison)
    
    print(f"\n📸 Đã lưu ảnh:")
    print(f"  - Trước clip: {before_path}")
    print(f"  - Sau clip:   {after_path}")
    print(f"  - So sánh:    {comparison_path}")


def clip_all_labels():
    """Clip tọa độ trong toàn bộ dataset."""
    label_files = glob.glob(os.path.join(DATASET_DIR, '*.txt'))
    
    total_files = 0
    fixed_files = 0
    fixed_lines = 0
    
    for lbl_path in sorted(label_files):
        total_files += 1
        annotations = read_labels(lbl_path)
        
        # Kiểm tra có annotation nào bị lệch không
        has_issue = False
        lines_fixed_in_file = 0
        for ann in annotations:
            if any(c < 0 or c > 1 for c in ann['coords']):
                has_issue = True
                lines_fixed_in_file += 1
        
        if has_issue:
            clipped = clip_annotations(annotations)
            write_labels(lbl_path, clipped)
            fixed_files += 1
            fixed_lines += lines_fixed_in_file
    
    print(f"\n✅ Clip hoàn tất!")
    print(f"  - Tổng file label:   {total_files}")
    print(f"  - File đã sửa:      {fixed_files}")
    print(f"  - Dòng đã clip:     {fixed_lines}")


def verify_after_clip():
    """Xác minh không còn tọa độ ngoài [0,1] sau clip."""
    label_files = glob.glob(os.path.join(DATASET_DIR, '*.txt'))
    issues = 0
    
    for lbl_path in label_files:
        annotations = read_labels(lbl_path)
        for ann in annotations:
            if any(c < 0 or c > 1 for c in ann['coords']):
                issues += 1
                print(f"  CÒN LỖI: {lbl_path}")
    
    if issues == 0:
        print("✅ Xác minh thành công: Không còn tọa độ ngoài [0, 1]!")
    else:
        print(f"❌ Còn {issues} file có vấn đề!")


if __name__ == '__main__':
    print("=" * 60)
    print("🔧 CLIP COORDINATES TOOL - YOLO OBB")
    print("=" * 60)
    
    # Bước 1: Tạo ảnh so sánh TRƯỚC khi clip
    print("\n📷 Bước 1: Tạo ảnh mẫu TRƯỚC clip...")
    create_comparison_image()
    
    # Bước 2: Clip tất cả label
    print("\n🔧 Bước 2: Clip tọa độ toàn bộ dataset...")
    clip_all_labels()
    
    # Bước 3: Xác minh
    print("\n🔍 Bước 3: Xác minh sau clip...")
    verify_after_clip()
    
    print("\n" + "=" * 60)
    print("🎉 HOÀN TẤT! Chạy lại split_data.py để cập nhật yolo_format.")
    print("=" * 60)
