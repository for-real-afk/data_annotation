import os

yolo_dir = "yolo_dataset"
for split in ["train", "val"]:
    img_dir = os.path.join(yolo_dir, "images", split)
    lbl_dir = os.path.join(yolo_dir, "labels", split)
    
    img_count = len([f for f in os.listdir(img_dir) if f.endswith('.png')]) if os.path.exists(img_dir) else 0
    lbl_count = len([f for f in os.listdir(lbl_dir) if f.endswith('.txt')]) if os.path.exists(lbl_dir) else 0
    print(f"{split} split: {img_count} images, {lbl_count} labels")
