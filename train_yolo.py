import os
import argparse
import torch
from ultralytics import YOLO

def train_yolo():
    parser = argparse.ArgumentParser(description="YOLOv8 Door Segmentation Training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for training")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    args = parser.parse_args()
    
    # 1. Define dataset config path
    dataset_yaml = os.path.abspath("yolo_dataset/dataset.yaml")
    if not os.path.exists(dataset_yaml):
        print(f"Error: Dataset config {dataset_yaml} not found. Run prepare_yolo_dataset.py first!")
        return
        
    print(f"Loading pre-trained YOLOv8n-seg baseline weights...")
    model = YOLO("yolov8n-seg.pt")
    
    # 2. Determine training device
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} (CUDA available: {torch.cuda.is_available()})")
    
    # 3. Launch training
    print(f"Starting training for {args.epochs} epochs with batch size {args.batch}...")
    results = model.train(
        data=dataset_yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        project="yolo_training",
        name="yolov8_door_seg",
        exist_ok=True,
        workers=2 if device == "cpu" else 4
    )
    
    # 4. Display training results summary
    best_weights = os.path.join("yolo_training", "yolov8_door_seg", "weights", "best.pt")
    print("\n" + "="*50)
    print("Training Completed Successfully!")
    if os.path.exists(best_weights):
        print(f"Best trained model weights saved to: {os.path.abspath(best_weights)}")
        print("You can load this model to perform door segmentations on new images!")
    print("="*50)

if __name__ == "__main__":
    train_yolo()
