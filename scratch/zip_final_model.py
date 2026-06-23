import os
import shutil

zip_name = "final_model"
src_dir = "exports_temp"

if os.path.exists(zip_name + ".zip"):
    os.remove(zip_name + ".zip")

print(f"Creating {zip_name}.zip from {src_dir}...")
shutil.make_archive(zip_name, "zip", src_dir)
print(f"Successfully created {zip_name}.zip")

# Clean up exports_temp
if os.path.exists(src_dir):
    print(f"Cleaning up temporary directory {src_dir}...")
    shutil.rmtree(src_dir)
    print("Cleanup completed.")
