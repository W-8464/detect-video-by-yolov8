import os
import random
import shutil

def setup_yolo_directories(base_dir):
    yolo_dir = os.path.join(base_dir, 'yolo_format')
    
    if os.path.exists(yolo_dir):
        print("Đang dọn dẹp dữ liệu cũ trong yolo_format...")
        shutil.rmtree(yolo_dir)
    
    os.makedirs(os.path.join(yolo_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(yolo_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(yolo_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(yolo_dir, 'labels', 'val'), exist_ok=True)
    
    return yolo_dir

def split_data(base_dir, split_ratio=0.8):
    # CHỈ SỬA Ở ĐÂY: Trỏ vào thư mục obj_train_data giải nén từ CVAT
    source_dir = os.path.join(base_dir, 'obj_train_data') 
    
    if not os.path.exists(source_dir):
        print(f"LỖI: Không tìm thấy thư mục {source_dir}. Hãy copy thư mục obj_train_data từ CVAT vào đây!")
        return

    yolo_dir = setup_yolo_directories(base_dir)
    
    all_files = os.listdir(source_dir)
    images = [f for f in all_files if f.endswith(('.jpg', '.png'))]
    
    random.seed(42) 
    random.shuffle(images)
    
    train_size = int(len(images) * split_ratio)
    train_images = images[:train_size]
    val_images = images[train_size:]
    
    def copy_files(image_list, dest_folder):
        for img in image_list:
            base_name = os.path.splitext(img)[0]
            txt_file = base_name + '.txt'
            
            img_src = os.path.join(source_dir, img)
            txt_src = os.path.join(source_dir, txt_file)
            
            # CVAT luôn sinh ra file .txt cho mọi frame (dù không có vật thể nào)
            # Nên nó luôn thỏa mãn điều kiện này, rất tốt để AI học ảnh nền trống!
            if os.path.exists(txt_src):
                shutil.copy(img_src, os.path.join(yolo_dir, 'images', dest_folder, img))
                shutil.copy(txt_src, os.path.join(yolo_dir, 'labels', dest_folder, txt_file))

    print(f"Bắt đầu chia dữ liệu: {len(train_images)} Train / {len(val_images)} Val")
    copy_files(train_images, 'train')
    copy_files(val_images, 'val')
    print("✅ Đã chia xong! Dữ liệu CVAT đã được convert sang format chuẩn YOLO.")

if __name__ == "__main__":
    project_dir = 'datasets' 
    split_data(project_dir)