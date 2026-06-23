import os
import sys
import shutil

def main():
    data_dir = r"C:\Users\deepa\.cache\kagglehub\datasets\qmarva\cubicasa5k\versions\4"
    print(f"Dataset path: {data_dir}")
    
    dataset_dir = "dataset"
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    segments_dir = os.path.join(dataset_dir, "segments")
    accepted_dir = os.path.join(dataset_dir, "accepted")
    
    os.makedirs(accepted_dir, exist_ok=True)
    
    if not os.path.exists(images_dir):
        print(f"Error: {images_dir} does not exist. Run main pipeline first.")
        return
        
    print("Scanning source dataset for original SVG paths...")
    svg_map = {}
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".svg"):
                svg_path = os.path.join(root, file)
                rel_root = os.path.relpath(root, data_dir)
                plan_name = rel_root.replace(os.sep, "_")
                svg_map[plan_name] = svg_path
                
    import json
    # Get only accepted plan names from the current run (which are stored as dicts, not lists)
    print("Identifying accepted plans from the current run...")
    accepted_plans = []
    for f in os.listdir(segments_dir):
        if f.endswith(".json") and f != "metadata.json" and f != "door_types.json" and f != "stats.json":
            plan_name = os.path.splitext(f)[0]
            json_path = os.path.join(segments_dir, f)
            try:
                with open(json_path, "r") as jf:
                    data = json.load(jf)
                if isinstance(data, dict):
                    accepted_plans.append(plan_name)
            except Exception as ex:
                pass
    print(f"Found {len(accepted_plans)} accepted plans.")
    
    copied_count = 0
    for idx, plan in enumerate(accepted_plans):
        # 1. Original SVG
        src_svg = svg_map.get(plan)
        dst_svg = os.path.join(accepted_dir, f"{plan}.svg")
        
        # 2. Rendered/Original Image
        src_png = os.path.join(images_dir, f"{plan}.png")
        dst_png = os.path.join(accepted_dir, f"{plan}.png")
        
        # 3. YOLO Label TXT
        src_txt = os.path.join(labels_dir, f"{plan}.txt")
        dst_txt = os.path.join(accepted_dir, f"{plan}.txt")
        
        # 4. Anatomy Metadata JSON
        src_json = os.path.join(segments_dir, f"{plan}.json")
        dst_json = os.path.join(accepted_dir, f"{plan}.json")
        
        # Copy files if source exists
        try:
            if src_svg and os.path.exists(src_svg):
                shutil.copy(src_svg, dst_svg)
            if os.path.exists(src_png):
                shutil.copy(src_png, dst_png)
            if os.path.exists(src_txt):
                shutil.copy(src_txt, dst_txt)
            if os.path.exists(src_json):
                shutil.copy(src_json, dst_json)
            copied_count += 1
        except Exception as e:
            print(f"Error copying files for plan {plan}: {e}")
            
        if (idx + 1) % 100 == 0 or (idx + 1) == len(accepted_plans):
            print(f"Processed {idx + 1}/{len(accepted_plans)} plans...")
            
    print(f"Finished. Arranged {copied_count} accepted plans in {accepted_dir}.")

if __name__ == "__main__":
    main()
