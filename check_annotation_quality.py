"""
Script phân tích chất lượng annotation trong dataset OBB.
Kiểm tra: kích thước box, tỷ lệ box, overlap, phân bố class theo video,
box bất thường, consistency giữa các frame liên tiếp...
"""

import os
import glob
import numpy as np
from collections import defaultdict, Counter
import math

DATASET_DIR = 'datasets/obj_train_data'
CLASS_NAMES = ['hand', 'tray', 'board', 'jig', 'liner',
               'tweezers', 'shielding', 'PCIe cable', 'gasket']

def read_labels(label_path):
    annotations = []
    with open(label_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) != 9:
                annotations.append({'error': f'Sai số cột: {len(parts)} (cần 9)', 'line': line_num})
                continue
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            # Tính width, height từ 4 đỉnh OBB
            xs = [coords[i] for i in range(0, 8, 2)]
            ys = [coords[i] for i in range(1, 8, 2)]
            # OBB dimensions: khoảng cách giữa các cạnh
            d01 = math.sqrt((coords[2]-coords[0])**2 + (coords[3]-coords[1])**2)
            d12 = math.sqrt((coords[4]-coords[2])**2 + (coords[5]-coords[3])**2)
            w = max(d01, d12)
            h = min(d01, d12)
            cx = sum(xs) / 4
            cy = sum(ys) / 4
            area = w * h
            
            annotations.append({
                'class_id': class_id,
                'coords': coords,
                'cx': cx, 'cy': cy,
                'w': w, 'h': h,
                'area': area,
                'aspect_ratio': w / h if h > 0 else 999,
                'xs': xs, 'ys': ys,
                'line': line_num
            })
    return annotations


def check_box_sizes(all_data):
    """Kiểm tra kích thước box bất thường."""
    print("\n" + "="*70)
    print("📏 KIỂM TRA KÍCH THƯỚC BOX")
    print("="*70)
    
    class_sizes = defaultdict(list)
    tiny_boxes = []
    huge_boxes = []
    
    for fname, anns in all_data.items():
        for ann in anns:
            if 'error' in ann:
                continue
            cls_id = ann['class_id']
            area = ann['area']
            class_sizes[cls_id].append((fname, ann))
            
            if area < 0.0005:  # Box quá nhỏ (< 0.05% diện tích ảnh)
                tiny_boxes.append((fname, ann))
            if area > 0.5:  # Box quá lớn (> 50% diện tích ảnh)
                huge_boxes.append((fname, ann))
    
    print("\n📊 Thống kê kích thước box theo class:")
    print(f"{'Class':<15} {'Count':>6} {'Area min':>10} {'Area avg':>10} {'Area max':>10} {'AR avg':>8} {'AR max':>8}")
    print("-" * 75)
    for cls_id in sorted(class_sizes.keys()):
        items = class_sizes[cls_id]
        areas = [ann['area'] for _, ann in items]
        ars = [ann['aspect_ratio'] for _, ann in items]
        print(f"{CLASS_NAMES[cls_id]:<15} {len(items):>6} {min(areas):>10.6f} {np.mean(areas):>10.6f} {max(areas):>10.6f} {np.mean(ars):>8.2f} {max(ars):>8.2f}")
    
    if tiny_boxes:
        print(f"\n⚠️ BOX QUÁ NHỎ (area < 0.05%): {len(tiny_boxes)} boxes")
        for fname, ann in tiny_boxes[:10]:
            print(f"  {os.path.basename(fname)}: {CLASS_NAMES[ann['class_id']]} area={ann['area']:.6f} w={ann['w']:.4f} h={ann['h']:.4f}")
    
    if huge_boxes:
        print(f"\n⚠️ BOX QUÁ LỚN (area > 50%): {len(huge_boxes)} boxes")
        for fname, ann in huge_boxes[:10]:
            print(f"  {os.path.basename(fname)}: {CLASS_NAMES[ann['class_id']]} area={ann['area']:.6f}")


def check_aspect_ratios(all_data):
    """Kiểm tra tỷ lệ aspect ratio bất thường."""
    print("\n" + "="*70)
    print("📐 KIỂM TRA ASPECT RATIO BẤT THƯỜNG (>20:1)")
    print("="*70)
    
    extreme_ar = []
    for fname, anns in all_data.items():
        for ann in anns:
            if 'error' in ann:
                continue
            if ann['aspect_ratio'] > 20:
                extreme_ar.append((fname, ann))
    
    if extreme_ar:
        print(f"\n⚠️ Tìm thấy {len(extreme_ar)} boxes có aspect ratio > 20:1")
        for fname, ann in extreme_ar[:15]:
            print(f"  {os.path.basename(fname)}: {CLASS_NAMES[ann['class_id']]} AR={ann['aspect_ratio']:.1f} w={ann['w']:.4f} h={ann['h']:.4f}")
    else:
        print("✅ Không có box nào có aspect ratio quá cực đoan.")


def check_overlapping_same_class(all_data):
    """Kiểm tra overlap giữa các box cùng class (có thể bị gán trùng)."""
    print("\n" + "="*70)
    print("🔄 KIỂM TRA BOX BỊ GÁN TRÙNG (cùng class, gần nhau)")
    print("="*70)
    
    duplicates = []
    for fname, anns in all_data.items():
        valid_anns = [a for a in anns if 'error' not in a]
        # Nhóm theo class
        by_class = defaultdict(list)
        for ann in valid_anns:
            by_class[ann['class_id']].append(ann)
        
        for cls_id, cls_anns in by_class.items():
            # Bỏ qua shielding, gasket - đây là class có nhiều instance hợp lệ
            if cls_id in [6, 8]:  # shielding, gasket
                continue
            for i in range(len(cls_anns)):
                for j in range(i+1, len(cls_anns)):
                    a, b = cls_anns[i], cls_anns[j]
                    # Kiểm tra tâm gần nhau (< 5% khoảng cách)
                    dist = math.sqrt((a['cx']-b['cx'])**2 + (a['cy']-b['cy'])**2)
                    if dist < 0.05:
                        duplicates.append((fname, cls_id, a, b, dist))
    
    if duplicates:
        print(f"\n⚠️ Tìm thấy {len(duplicates)} cặp box nghi ngờ BỊ GÁN TRÙNG:")
        for fname, cls_id, a, b, dist in duplicates[:15]:
            print(f"  {os.path.basename(fname)}: 2x {CLASS_NAMES[cls_id]}, "
                  f"tâm1=({a['cx']:.3f},{a['cy']:.3f}), tâm2=({b['cx']:.3f},{b['cy']:.3f}), "
                  f"dist={dist:.4f}")
    else:
        print("✅ Không phát hiện box bị gán trùng.")


def check_class_consistency(all_data):
    """Kiểm tra sự nhất quán của class giữa các frame."""
    print("\n" + "="*70)
    print("🎬 PHÂN BỐ CLASS THEO VIDEO")
    print("="*70)
    
    # Nhóm theo video
    video_classes = defaultdict(lambda: defaultdict(int))
    video_frame_count = defaultdict(int)
    
    for fname, anns in all_data.items():
        base = os.path.basename(fname).replace('.txt', '')
        # Tách tên video từ tên file (vd: xb10_1_000000 → xb10_1)
        parts = base.rsplit('_', 1)
        video_name = parts[0] if len(parts) > 1 else base
        video_frame_count[video_name] += 1
        
        for ann in anns:
            if 'error' not in ann:
                video_classes[video_name][ann['class_id']] += 1
    
    print(f"\n{'Video':<15} {'Frames':>7}", end="")
    for name in CLASS_NAMES:
        print(f" {name[:5]:>6}", end="")
    print()
    print("-" * (15 + 7 + 6 * len(CLASS_NAMES) + len(CLASS_NAMES)))
    
    for video in sorted(video_classes.keys()):
        print(f"{video:<15} {video_frame_count[video]:>7}", end="")
        for cls_id in range(len(CLASS_NAMES)):
            count = video_classes[video].get(cls_id, 0)
            col = f"{count:>6}" if count > 0 else "     -"
            print(f" {col}", end="")
        print()


def check_missing_classes_per_frame(all_data):
    """Kiểm tra frame nào thiếu class quan trọng."""
    print("\n" + "="*70)
    print("❓ KIỂM TRA FRAME THIẾU CLASS QUAN TRỌNG")
    print("="*70)
    
    # Các class thường luôn xuất hiện (tray, board, jig thường cố định)
    static_classes = {1: 'tray', 2: 'board', 3: 'jig'}
    
    # Nhóm frame theo video
    video_frames = defaultdict(dict)
    for fname, anns in all_data.items():
        base = os.path.basename(fname).replace('.txt', '')
        parts = base.rsplit('_', 1)
        video_name = parts[0]
        frame_num = int(parts[1]) if len(parts) > 1 else 0
        
        classes_in_frame = set()
        for ann in anns:
            if 'error' not in ann:
                classes_in_frame.add(ann['class_id'])
        video_frames[video_name][frame_num] = classes_in_frame
    
    for video in sorted(video_frames.keys()):
        frames = video_frames[video]
        sorted_frames = sorted(frames.keys())
        
        # Tìm class nào xuất hiện trong >80% frame → xem là "expected"
        class_freq = Counter()
        for f in sorted_frames:
            for cls in frames[f]:
                class_freq[cls] += 1
        
        expected = {cls for cls, cnt in class_freq.items() if cnt > 0.8 * len(sorted_frames)}
        
        missing_report = []
        for f in sorted_frames:
            missing = expected - frames[f]
            if missing:
                names = [CLASS_NAMES[c] for c in missing]
                missing_report.append((f, names))
        
        if missing_report:
            print(f"\n📹 {video}: {len(missing_report)} frame(s) thiếu class thường có:")
            for f, names in missing_report[:8]:
                print(f"  Frame {f:06d}: thiếu {', '.join(names)}")
            if len(missing_report) > 8:
                print(f"  ... và {len(missing_report) - 8} frame khác")


def check_degenerate_boxes(all_data):
    """Kiểm tra box bị suy biến (diện tích ~0, các đỉnh trùng nhau)."""
    print("\n" + "="*70)
    print("💀 KIỂM TRA BOX SUY BIẾN (đỉnh trùng, diện tích ≈ 0)")
    print("="*70)
    
    degenerate = []
    for fname, anns in all_data.items():
        for ann in anns:
            if 'error' in ann:
                continue
            # Kiểm tra đỉnh trùng
            coords = ann['coords']
            pts = [(coords[i], coords[i+1]) for i in range(0, 8, 2)]
            
            # Check nếu 2 đỉnh liên tiếp trùng nhau
            for i in range(4):
                j = (i + 1) % 4
                dist = math.sqrt((pts[i][0]-pts[j][0])**2 + (pts[i][1]-pts[j][1])**2)
                if dist < 0.001:
                    degenerate.append((fname, ann, 'đỉnh trùng'))
                    break
            
            # Check diện tích quá nhỏ
            if ann['area'] < 0.00001:
                degenerate.append((fname, ann, 'diện tích ≈ 0'))
    
    if degenerate:
        print(f"\n⚠️ Tìm thấy {len(degenerate)} boxes suy biến:")
        for fname, ann, reason in degenerate[:10]:
            print(f"  {os.path.basename(fname)} line {ann['line']}: {CLASS_NAMES[ann['class_id']]} - {reason}")
    else:
        print("✅ Không có box suy biến.")


def check_box_outside_center(all_data):
    """Kiểm tra box có tâm nằm quá sát biên (có thể bị cắt)."""
    print("\n" + "="*70)
    print("📍 KIỂM TRA BOX CÓ TÂM QUÁ SÁT BIÊN (có thể bị cắt)")
    print("="*70)
    
    edge_boxes = []
    for fname, anns in all_data.items():
        for ann in anns:
            if 'error' in ann:
                continue
            margin = 0.02  # 2% biên
            if ann['cx'] < margin or ann['cx'] > 1-margin or ann['cy'] < margin or ann['cy'] > 1-margin:
                edge_boxes.append((fname, ann))
    
    print(f"\n📊 {len(edge_boxes)} boxes có tâm sát biên (<2%)")
    # Nhóm theo class
    by_class = Counter()
    for fname, ann in edge_boxes:
        by_class[ann['class_id']] += 1
    
    for cls_id, cnt in by_class.most_common():
        print(f"  {CLASS_NAMES[cls_id]}: {cnt} boxes")


def check_rotation_consistency(all_data):
    """Kiểm tra góc xoay của box có nhất quán không."""
    print("\n" + "="*70)
    print("🔄 KIỂM TRA GÓC XOAY OBB")
    print("="*70)
    
    class_rotations = defaultdict(list)
    for fname, anns in all_data.items():
        for ann in anns:
            if 'error' in ann:
                continue
            coords = ann['coords']
            # Tính góc xoay từ cạnh đầu tiên
            dx = coords[2] - coords[0]
            dy = coords[3] - coords[1]
            angle = math.degrees(math.atan2(dy, dx))
            class_rotations[ann['class_id']].append((fname, angle, ann))
    
    print(f"\n{'Class':<15} {'Chủ yếu':>20} {'Axis-aligned (0°/90°)':>25}")
    print("-" * 65)
    for cls_id in sorted(class_rotations.keys()):
        angles = [a for _, a, _ in class_rotations[cls_id]]
        angles_abs = [abs(a) % 180 for a in angles]
        
        # Kiểm tra bao nhiêu % box gần axis-aligned (0° hoặc 90°)
        axis_aligned = sum(1 for a in angles_abs if a < 5 or abs(a-90) < 5 or abs(a-180) < 5)
        pct = axis_aligned / len(angles) * 100
        
        mean_angle = np.mean(angles_abs)
        std_angle = np.std(angles_abs)
        
        print(f"{CLASS_NAMES[cls_id]:<15} mean={mean_angle:>6.1f}° ± {std_angle:>5.1f}°   {pct:>6.1f}% axis-aligned")


def check_label_format_errors(all_data):
    """Kiểm tra lỗi format trong label files."""
    print("\n" + "="*70)
    print("📄 KIỂM TRA LỖI FORMAT")
    print("="*70)
    
    errors = []
    empty_files = []
    
    for fname, anns in all_data.items():
        if len(anns) == 0:
            empty_files.append(fname)
        for ann in anns:
            if 'error' in ann:
                errors.append((fname, ann))
    
    if errors:
        print(f"\n❌ {len(errors)} dòng bị lỗi format:")
        for fname, ann in errors[:10]:
            print(f"  {os.path.basename(fname)} line {ann['line']}: {ann['error']}")
    else:
        print("✅ Tất cả label đều đúng format.")
    
    if empty_files:
        print(f"\n📭 {len(empty_files)} file label rỗng (không có object):")
        for f in empty_files[:5]:
            print(f"  {os.path.basename(f)}")


def check_class_id_validity(all_data):
    """Kiểm tra class ID có hợp lệ không."""
    print("\n" + "="*70)
    print("🏷️ KIỂM TRA CLASS ID")
    print("="*70)
    
    invalid = []
    for fname, anns in all_data.items():
        for ann in anns:
            if 'error' in ann:
                continue
            if ann['class_id'] < 0 or ann['class_id'] >= len(CLASS_NAMES):
                invalid.append((fname, ann))
    
    if invalid:
        print(f"\n❌ {len(invalid)} boxes có class_id không hợp lệ!")
        for fname, ann in invalid[:10]:
            print(f"  {os.path.basename(fname)}: class_id={ann['class_id']}")
    else:
        print("✅ Tất cả class ID đều hợp lệ (0-8).")


def check_multi_instance_anomaly(all_data):
    """Kiểm tra class nào thường chỉ có 1 instance nhưng đôi khi có nhiều."""
    print("\n" + "="*70)
    print("🔢 KIỂM TRA SỐ LƯỢNG INSTANCE MỖI CLASS TRONG TỪNG FRAME")
    print("="*70)
    
    class_counts_per_frame = defaultdict(list)
    
    for fname, anns in all_data.items():
        valid = [a for a in anns if 'error' not in a]
        counter = Counter(a['class_id'] for a in valid)
        for cls_id in range(len(CLASS_NAMES)):
            class_counts_per_frame[cls_id].append(counter.get(cls_id, 0))
    
    print(f"\n{'Class':<15} {'Min':>5} {'Avg':>6} {'Max':>5} {'Mode':>5} {'Std':>6}  Ghi chú")
    print("-" * 80)
    for cls_id in range(len(CLASS_NAMES)):
        counts = class_counts_per_frame[cls_id]
        non_zero = [c for c in counts if c > 0]
        if not non_zero:
            continue
        mode_val = Counter(non_zero).most_common(1)[0][0]
        max_val = max(non_zero)
        avg_val = np.mean(non_zero)
        std_val = np.std(non_zero)
        
        note = ""
        if max_val > 3 * mode_val and mode_val <= 2:
            note = f"⚠️ Max ({max_val}) cao bất thường so với mode ({mode_val})"
        
        print(f"{CLASS_NAMES[cls_id]:<15} {min(non_zero):>5} {avg_val:>6.1f} {max_val:>5} {mode_val:>5} {std_val:>6.2f}  {note}")


if __name__ == '__main__':
    print("="*70)
    print("🔍 PHÂN TÍCH CHẤT LƯỢNG ANNOTATION - YOLO OBB")
    print("="*70)
    
    # Load tất cả label
    label_files = sorted(glob.glob(os.path.join(DATASET_DIR, '*.txt')))
    print(f"\nĐọc {len(label_files)} file label...")
    
    all_data = {}
    for lf in label_files:
        all_data[lf] = read_labels(lf)
    
    total_anns = sum(len(v) for v in all_data.values())
    print(f"Tổng annotations: {total_anns}")
    
    # Chạy tất cả kiểm tra
    check_label_format_errors(all_data)
    check_class_id_validity(all_data)
    check_box_sizes(all_data)
    check_aspect_ratios(all_data)
    check_degenerate_boxes(all_data)
    check_overlapping_same_class(all_data)
    check_box_outside_center(all_data)
    check_multi_instance_anomaly(all_data)
    check_class_consistency(all_data)
    check_missing_classes_per_frame(all_data)
    check_rotation_consistency(all_data)
    
    print("\n" + "="*70)
    print("🎉 PHÂN TÍCH HOÀN TẤT!")
    print("="*70)
