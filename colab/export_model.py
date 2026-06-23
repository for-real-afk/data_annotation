import os
import argparse
import pickle
import shutil
from ultralytics import YOLO

def export_pipeline(weights_path, export_dir, img_size):
    print(f"Starting export pipeline for: {weights_path}")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Source weights file not found: {weights_path}")
        
    os.makedirs(export_dir, exist_ok=True)
    
    # 1. Load YOLO model
    print("Loading YOLO model weights...")
    model = YOLO(weights_path)
    
    # Get weights directory to locate generated exports
    weights_dir = os.path.dirname(os.path.abspath(weights_path))
    weights_basename = os.path.splitext(os.path.basename(weights_path))[0]
    
    # 2. Export to ONNX
    print("Exporting model to ONNX format...")
    onnx_path_src = model.export(format="onnx", imgsz=img_size, opset=12)
    print(f"ONNX export completed: {onnx_path_src}")
    
    # 3. Export to TorchScript
    print("Exporting model to TorchScript format...")
    ts_path_src = model.export(format="torchscript", imgsz=img_size)
    print(f"TorchScript export completed: {ts_path_src}")
    
    # 4. Define target export paths
    best_pt_dst = os.path.join(export_dir, "best.pt")
    best_onnx_dst = os.path.join(export_dir, "best.onnx")
    best_ts_dst = os.path.join(export_dir, "best.torchscript")
    pkl_dst = os.path.join(export_dir, "bom_model.pkl")
    
    # Copy files to export directory
    print(f"Copying files to export folder: {export_dir}")
    shutil.copy2(weights_path, best_pt_dst)
    
    # YOLO exports sometimes return paths, verify and copy
    if onnx_path_src and os.path.exists(onnx_path_src):
        shutil.copy2(onnx_path_src, best_onnx_dst)
    else:
        # Fallback locate
        onnx_fallback = os.path.join(weights_dir, f"{weights_basename}.onnx")
        if os.path.exists(onnx_fallback):
            shutil.copy2(onnx_fallback, best_onnx_dst)
            
    if ts_path_src and os.path.exists(ts_path_src):
        shutil.copy2(ts_path_src, best_ts_dst)
    else:
        # Fallback locate
        ts_fallback = os.path.join(weights_dir, f"{weights_basename}.torchscript")
        if os.path.exists(ts_fallback):
            shutil.copy2(ts_fallback, best_ts_dst)
            
    # 5. Create and save Metadata PKL
    metadata = {
        "classes": {
            0: "door",
            1: "window"
        },
        "img_size": img_size,
        "model_type": "YOLO11s-Seg",
        "epochs": 100,
        "dataset": "SVG Synthetic Floorplans",
        "weights": "best.pt"
    }
    
    with open(pkl_dst, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Metadata dictionary successfully saved to: {pkl_dst}")
    
    print("="*60)
    print("Export Stage Completed Successfully!")
    print(f"Artifacts available in: {os.path.abspath(export_dir)}")
    print(f" - PyTorch: {best_pt_dst}")
    print(f" - ONNX: {best_onnx_dst}")
    print(f" - TorchScript: {best_ts_dst}")
    print(f" - Metadata PKL: {pkl_dst}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Model Export Pipeline")
    parser.add_argument("--weights", type=str, default="bom_project/yolo11s_seg/weights/best.pt", help="Path to best.pt")
    parser.add_argument("--export_dir", type=str, default="exports", help="Directory to save exported artifacts")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image size used during training")
    args = parser.parse_args()
    
    try:
        export_pipeline(args.weights, args.export_dir, args.imgsz)
    except Exception as e:
        print(f"Export execution failed: {e}")
        import sys
        sys.exit(1)
