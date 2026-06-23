import zipfile
import sys

zip_path = "yolo_dataset.zip"
try:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        print(f"Checking zip integrity for {zip_path}...")
        bad_file = zip_ref.testzip()
        if bad_file:
            print(f"ZIP is corrupt! First bad file: {bad_file}")
            sys.exit(1)
        else:
            print("ZIP is fully intact and valid!")
            # Print file count
            infolist = zip_ref.infolist()
            print(f"Total files in ZIP: {len(infolist)}")
except Exception as e:
    print(f"Error opening zip: {e}")
    sys.exit(1)
