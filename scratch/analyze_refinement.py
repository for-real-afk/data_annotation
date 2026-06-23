import os
import json
import numpy as np

segments_dir = r"d:\projects\data_annotations\dataset\segments"
if not os.path.exists(segments_dir):
    print(f"Directory {segments_dir} does not exist.")
    sys.exit(0)

files = [f for f in os.listdir(segments_dir) if f.endswith(".json")]
print(f"Analyzing {len(files)} plan JSON files...")

drift_count = 0
large_translation_count = 0
total_doors = 0
drifted_doors = 0

for file in files:
    path = os.path.join(segments_dir, file)
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as ex:
        print(f"Error parsing {file}: {ex}")
        continue
        
    if isinstance(data, dict):
        meta = data.get("alignment_metadata", {})
        doors = data.get("doors", [])
    else:
        meta = {}
        doors = data
        
    total_doors += len(doors)
    
    for d in doors:
        if d.get("drift_status") != "aligned":
            drifted_doors += 1
            
    # Check if any door has negative coordinates
    has_negative_coords = False
    for d in doors:
        bbox = d.get("bbox")
        if bbox and (bbox[0] < 0 or bbox[1] < 0):
            has_negative_coords = True
            
    if has_negative_coords:
        large_translation_count += 1

print(f"Total plans: {len(files)}")
print(f"Plans with negative door coordinates (drift/border issue): {large_translation_count}")
print(f"Total doors: {total_doors}")
print(f"Drifted/misaligned/rejected doors: {drifted_doors}")
