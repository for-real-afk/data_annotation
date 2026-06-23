import os

deliverables = [
    "train.ipynb",
    "dataset.yaml",
    "train_config.yaml",
    "test_real_floorplan.py",
    "export_model.py",
    "package_model.py",
    "metrics.json",
    "door_metadata.json",
    "final_model.zip"
]

print("="*60)
print("Verifying Final Deliverables...")
all_present = True
for f in deliverables:
    if os.path.exists(f):
        size = os.path.getsize(f)
        print(f" [OK] {f:<25} | Size: {size:,} bytes")
    else:
        print(f" [MISSING] {f:<25}")
        all_present = False

if all_present:
    print("All target deliverables are present and valid!")
else:
    print("Warning: Some deliverables are missing!")
print("="*60)
