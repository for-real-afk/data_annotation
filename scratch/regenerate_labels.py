import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import cv2
from segmentation_generator import generate_yolo_labels

def regenerate():
    accepted_dir = "dataset\\accepted"
    if not os.path.exists(accepted_dir):
        print(f"Error: {accepted_dir} does not exist.")
        return
        
    json_files = [f for f in os.listdir(accepted_dir) if f.endswith(".json")]
    print(f"Regenerating YOLO labels for {len(json_files)} plans...")
    
    count = 0
    for idx, jf in enumerate(json_files):
        plan_name = os.path.splitext(jf)[0]
        json_path = os.path.join(accepted_dir, jf)
        png_path = os.path.join(accepted_dir, f"{plan_name}.png")
        txt_path = os.path.join(accepted_dir, f"{plan_name}.txt")
        
        # Load JSON and PNG
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
                
            img = cv2.imread(png_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            
            # Extract doors list (dict format)
            doors = data.get("doors", [])
            # Filter to aligned doors only
            aligned_doors = [d for d in doors if d.get("drift_status") == "aligned"]
            
            # Generate labels
            _, yolo_seg = generate_yolo_labels(aligned_doors, w, h)
            
            # Save back to txt
            with open(txt_path, "w") as f:
                f.write("\n".join(yolo_seg) + "\n")
                
            count += 1
        except Exception as e:
            print(f"Error processing {plan_name}: {e}")
            
        if (idx + 1) % 100 == 0 or (idx + 1) == len(json_files):
            print(f"Regenerated labels for {idx + 1}/{len(json_files)} plans...")
            
    print(f"Successfully regenerated {count} label files.")

if __name__ == "__main__":
    regenerate()
