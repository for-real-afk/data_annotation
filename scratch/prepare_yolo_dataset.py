import os
import shutil
import random

def prepare_dataset():
    accepted_dir = "dataset\\accepted"
    yolo_dir = "yolo_dataset"
    
    # Target directories
    img_train_dir = os.path.join(yolo_dir, "images", "train")
    img_val_dir = os.path.join(yolo_dir, "images", "val")
    lbl_train_dir = os.path.join(yolo_dir, "labels", "train")
    lbl_val_dir = os.path.join(yolo_dir, "labels", "val")
    
    # Recreate clean directories
    if os.path.exists(yolo_dir):
        shutil.rmtree(yolo_dir)
        
    for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
        os.makedirs(d, exist_ok=True)
        
    # Get all unique plans (plans are those with .png files in dataset/accepted)
    plans = [os.path.splitext(f)[0] for f in os.listdir(accepted_dir) if f.endswith(".png")]
    print(f"Total accepted plans: {len(plans)}")
    
    # Shuffle and split (85% train, 15% val)
    random.seed(42)
    random.shuffle(plans)
    
    split_idx = int(len(plans) * 0.85)
    train_plans = plans[:split_idx]
    val_plans = plans[split_idx:]
    
    print(f"Split: {len(train_plans)} training plans, {len(val_plans)} validation plans")
    
    # Copy helper function
    def copy_plan_files(plan, img_dest, lbl_dest):
        src_img = os.path.join(accepted_dir, f"{plan}.png")
        src_lbl = os.path.join(accepted_dir, f"{plan}.txt")
        
        dst_img = os.path.join(img_dest, f"{plan}.png")
        dst_lbl = os.path.join(lbl_dest, f"{plan}.txt")
        
        if os.path.exists(src_img):
            shutil.copy(src_img, dst_img)
        if os.path.exists(src_lbl):
            shutil.copy(src_lbl, dst_lbl)
            
    # Copy training plans
    for plan in train_plans:
        copy_plan_files(plan, img_train_dir, lbl_train_dir)
        
    # Copy validation plans
    for plan in val_plans:
        copy_plan_files(plan, img_val_dir, lbl_val_dir)
        
    # Generate dataset.yaml
    yaml_content = f"""path: /content/yolo_dataset
train: images/train
val: images/val

names:
  0: door
"""
    yaml_path = os.path.join(yolo_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
        
    print(f"YOLO dataset prepared successfully at {yolo_dir}/")
    print(f"Created dataset configuration file: {yaml_path}")

if __name__ == "__main__":
    prepare_dataset()
