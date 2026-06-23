import os
import shutil
import zipfile
import sys

zip_path = "yolo_dataset.zip"
src_dir = "yolo_dataset"

# 1. Remove existing zip if it exists
if os.path.exists(zip_path):
    print(f"Removing existing {zip_path}...")
    os.remove(zip_path)

# 2. Create archive
print(f"Zipping {src_dir} directory...")
shutil.make_archive("yolo_dataset", "zip", src_dir)
print(f"Archive created at {zip_path}")

# 3. Verify Zip
try:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        bad_file = zip_ref.testzip()
        if bad_file:
            print(f"ZIP is corrupt! First bad file: {bad_file}")
            sys.exit(1)
        else:
            print("ZIP is fully intact and valid!")
            
            # Print first few files to verify structure
            infolist = zip_ref.infolist()
            print(f"Total files in ZIP: {len(infolist)}")
            print("Root entries in ZIP:")
            roots = set()
            for info in infolist[:15]:
                print(f"  {info.filename}")
except Exception as e:
    print(f"Error opening zip: {e}")
    sys.exit(1)
