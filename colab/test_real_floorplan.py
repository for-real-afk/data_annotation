import os
import argparse
import cv2
import json
from ultralytics import YOLO

def run_local_inference(weights_path, image_path, output_dir):
    print("="*60)
    print("Running Floorplan Segmenter Local Inference...")
    print(f" - Weights: {weights_path}")
    print(f" - Input Image: {image_path}")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image file not found: {image_path}")
        
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load YOLO model (supports .pt, .onnx, and other formats natively via Ultralytics)
    print("Loading model...")
    model = YOLO(weights_path)
    
    # 2. Run inference
    print("Executing model inference...")
    results = model.predict(source=image_path, imgsz=1024, conf=0.25, save=False)
    
    # 3. Process results
    door_count = 0
    window_count = 0
    detections = []
    
    class_map = {0: "door", 1: "window"}
    
    for result in results:
        boxes = result.boxes
        masks = result.masks
        
        # Save plotted image
        plotted_img = result.plot()
        output_image_name = f"prediction_{os.path.basename(image_path)}"
        output_image_path = os.path.join(output_dir, output_image_name)
        cv2.imwrite(output_image_path, plotted_img)
        print(f"Overlay image successfully saved to: {output_image_path}")
        
        if boxes is not None:
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0].item())
                cls_name = class_map.get(cls_id, f"unknown_{cls_id}")
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()
                
                # Class tracking
                if cls_id == 0:
                    door_count += 1
                elif cls_id == 1:
                    window_count += 1
                    
                det_info = {
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": conf,
                    "bbox": xyxy
                }
                
                # Try to extract polygon coordinates if mask is present
                if masks is not None and len(masks.xy) > i:
                    polygon_coords = masks.xy[i].tolist()
                    det_info["polygon"] = polygon_coords
                    
                detections.append(det_info)
                
    # 4. Print Summary
    print("\n" + "="*30 + " Inference Summary " + "="*30)
    print(f"Total Doors Detected: {door_count}")
    print(f"Total Windows Detected: {window_count}")
    print(f"Total Detections: {len(detections)}")
    for j, det in enumerate(detections[:10]):
        print(f" [{j+1}] Class: {det['class_name']} | Conf: {det['confidence']:.4f} | BBox: {[round(c, 2) for c in det['bbox']]}")
    if len(detections) > 10:
        print(f" ... and {len(detections) - 10} more detections.")
    print("="*79 + "\n")
    
    # 5. Output detections report locally as JSON
    report_path = os.path.join(output_dir, "inference_report.json")
    report_data = {
        "source_image": os.path.basename(image_path),
        "door_count": door_count,
        "window_count": window_count,
        "detections": detections
    }
    
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"Detections JSON report saved to: {report_path}")
    print("Inference completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Architectural Floorplan Local Inference Script")
    parser.add_argument("--weights", type=str, required=True, help="Path to best.pt or best.onnx model")
    parser.add_argument("--image", type=str, required=True, help="Path to input floorplan image (PNG/JPG)")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save output overlays and report")
    args = parser.parse_args()
    
    try:
        run_local_inference(args.weights, args.image, args.output_dir)
    except Exception as e:
        print(f"Local inference execution failed: {e}")
        import sys
        sys.exit(1)
