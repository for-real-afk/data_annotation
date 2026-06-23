import os
import re

download_dir = r"C:\Users\deepa\Downloads\good ones"
accepted_dir = r"d:\projects\data_annotations\dataset\accepted"

png_files = [f for f in os.listdir(download_dir) if f.endswith(".png")]
print(f"Total files in Downloads/good ones: {len(png_files)}")

matches = 0
missing = []
for f in png_files:
    # Extract plan name
    # e.g., sample_001_cubicasa5k_cubicasa5k_high_quality_architectural_1850.png
    # -> cubicasa5k_cubicasa5k_high_quality_architectural_1850
    m = re.match(r"sample_\d+_(cubicasa5k_cubicasa5k_.*)\.png", f)
    if m:
        plan_name = m.group(1)
        txt_file = f"{plan_name}.txt"
        txt_path = os.path.join(accepted_dir, txt_file)
        if os.path.exists(txt_path):
            matches += 1
        else:
            missing.append((f, plan_name))
    else:
        # Try raw matching
        plan_name = os.path.splitext(f)[0]
        txt_file = f"{plan_name}.txt"
        txt_path = os.path.join(accepted_dir, txt_file)
        if os.path.exists(txt_path):
            matches += 1
        else:
            missing.append((f, plan_name))

print(f"Matches found in dataset/accepted: {matches} / {len(png_files)}")
if missing:
    print(f"First 10 missing plans:")
    for f, p in missing[:10]:
        print(f"  - File: {f} | Plan: {p}")
