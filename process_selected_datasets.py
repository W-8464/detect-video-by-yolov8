import os
import shutil
import random
from pathlib import Path

def process():
    base_src = Path("datasets")
    yolo_dst = Path("datasets/yolo_format")
    
    # Tự động tìm tất cả các thư mục có dạng xb10_... và chứa folder images
    src_dirs = [
        d.name for d in base_src.iterdir() 
        if d.is_dir() and d.name.startswith("xb10_") and (d / "images").exists() and not d.name.endswith("_pose")
    ]
    src_dirs.sort()
    
    print(f"🔍 Đã tìm thấy {len(src_dirs)} thư mục dữ liệu: {src_dirs}")
    print(f"🧹 Đang dọn dẹp {yolo_dst}...")
    
    if yolo_dst.exists():
        shutil.rmtree(yolo_dst)
    
    # Chuẩn bị thư mục đích
    for split in ["train", "val"]:
        (yolo_dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_dst / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    all_pairs = []
    
    for sdir_name in src_dirs:
        sdir = base_src / sdir_name
        
        # Thử tìm trong images/train hoặc images trực tiếp
        img_dir = sdir / "images" / "train"
        lbl_dir = sdir / "labels" / "train"
        
        if not img_dir.exists():
            img_dir = sdir / "images"
            lbl_dir = sdir / "labels"
            
        if not img_dir.exists():
            continue
            
        count = 0
        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                if lbl_path.exists():
                    all_pairs.append((img_path, lbl_path, sdir_name))
                    count += 1
        if count > 0:
            print(f"✅ {sdir_name}: {count} cặp")
    
    print(f"\n📊 Tổng cộng: {len(all_pairs)} cặp image/label")
    
    if not all_pairs:
        print("❌ Không tìm thấy dữ liệu để xử lý.")
        return

    random.seed(42)
    random.shuffle(all_pairs)
    
    split_ratio = 0.85
    split_idx = int(len(all_pairs) * split_ratio)
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]
    
    def copy_pairs(pairs, split):
        for img_src, lbl_src, sname in pairs:
            # Thêm tiền tố tên thư mục nguồn để tránh trùng tên file frame_000...
            new_stem = f"{sname}_{img_src.stem}"
            img_dst = yolo_dst / "images" / split / f"{new_stem}{img_src.suffix}"
            lbl_dst = yolo_dst / "labels" / split / f"{new_stem}.txt"
            
            shutil.copy2(img_src, img_dst)
            shutil.copy2(lbl_src, lbl_dst)
            
    print(f"🚚 Đang copy {len(train_pairs)} vào train...")
    copy_pairs(train_pairs, "train")
    print(f"🚚 Đang copy {len(val_pairs)} vào val...")
    copy_pairs(val_pairs, "val")
    
    print("\n✅ Đã gộp TẤT CẢ dữ liệu thành công vào yolo_format!")

if __name__ == "__main__":
    process()
