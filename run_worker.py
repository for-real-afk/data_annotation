import os
import sys
import shutil
import subprocess

def run_pipeline():
    pycache_path = os.path.join("d:\\projects\\data_annotations", "__pycache__")
    if os.path.exists(pycache_path):
        print("Clearing pycache...")
        try:
            shutil.rmtree(pycache_path)
            print("Successfully cleared pycache.")
        except Exception as e:
            print(f"Warning: failed to clear pycache: {e}")
            
    # Launch main pipeline
    # Let's run with 6 workers (or auto-detected CPU count in main.py)
    cmd = [
        r"d:\projects\env\Scripts\python.exe",
        "main.py",
        "--output_dir", "dataset",
        "--workers", "8",
        "--qa_limit", "100"
    ]
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    run_pipeline()
