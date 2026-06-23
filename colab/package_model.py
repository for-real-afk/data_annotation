import os
import argparse
import shutil
import zipfile

def package_artifacts(export_dir, training_run_dir, dataset_dir, zip_dest_dir, drive_backup_dir):
    print("="*60)
    print("Starting packaging stage...")
    
    os.makedirs(zip_dest_dir, exist_ok=True)
    
    # 1. Copy metrics and plots from training run directory to exports
    print(f"Copying metrics and curves from: {training_run_dir} to: {export_dir}")
    metrics_files = [
        "results.csv",
        "PR_curve.png",
        "F1_curve.png",
        "confusion_matrix.png",
        "args.yaml"
    ]
    
    for f_name in metrics_files:
        src = os.path.join(training_run_dir, f_name)
        dst = os.path.join(export_dir, f_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f" - Copied {f_name}")
        else:
            print(f" - Warning: {f_name} not found in training run dir")

    # 2. Package exports/ to bom_training_bundle.zip
    bundle_zip_path = os.path.join(zip_dest_dir, "bom_training_bundle.zip")
    print(f"Creating training bundle: {bundle_zip_path}")
    if os.path.exists(bundle_zip_path):
        os.remove(bundle_zip_path)
        
    shutil.make_archive(
        base_name=os.path.join(zip_dest_dir, "bom_training_bundle"),
        format="zip",
        root_dir=export_dir
    )
    print(f"Successfully generated {bundle_zip_path}")
    
    # 3. Package dataset to dataset_export.zip
    dataset_zip_path = os.path.join(zip_dest_dir, "dataset_export.zip")
    print(f"Creating dataset backup: {dataset_zip_path}")
    if os.path.exists(dataset_zip_path):
        os.remove(dataset_zip_path)
        
    if os.path.exists(dataset_dir):
        # We want the ZIP root to contain images/, labels/ and dataset.yaml
        shutil.make_archive(
            base_name=os.path.join(zip_dest_dir, "dataset_export"),
            format="zip",
            root_dir=dataset_dir
        )
        print(f"Successfully generated {dataset_zip_path}")
    else:
        print(f"Error: Dataset directory {dataset_dir} not found. Cannot create dataset_export.zip")
        
    # 4. Copy to Google Drive if available
    if drive_backup_dir:
        print(f"Google Drive target backup directory: {drive_backup_dir}")
        try:
            os.makedirs(drive_backup_dir, exist_ok=True)
            
            # Files to back up to Drive
            drive_files = [
                (os.path.join(export_dir, "best.pt"), "best.pt"),
                (os.path.join(export_dir, "best.onnx"), "best.onnx"),
                (os.path.join(export_dir, "best.torchscript"), "best.torchscript"),
                (os.path.join(export_dir, "bom_model.pkl"), "bom_model.pkl"),
                (bundle_zip_path, "bom_training_bundle.zip"),
                (dataset_zip_path, "dataset_export.zip")
            ]
            
            for src_file, dest_name in drive_files:
                if os.path.exists(src_file):
                    dst = os.path.join(drive_backup_dir, dest_name)
                    shutil.copy2(src_file, dst)
                    print(f" - Backed up to Drive: {dest_name}")
                else:
                    print(f" - Info: {src_file} not found, skipping Drive copy")
            print("Google Drive backup completed successfully!")
        except Exception as e:
            print(f"Warning: Failed to back up files to Google Drive: {e}")
            print("Artifacts remain saved locally in /content/")
            
    print("Packaging Stage Completed Successfully!")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Training Artifact Packaging Script")
    parser.add_argument("--export_dir", type=str, default="exports", help="Directory of exported model artifacts")
    parser.add_argument("--run_dir", type=str, default="bom_project/yolo11s_seg", help="Training run output directory")
    parser.add_argument("--dataset_dir", type=str, default="yolo_dataset", help="YOLO dataset directory to zip")
    parser.add_argument("--zip_dest_dir", type=str, default=".", help="Directory to save generated ZIP files")
    parser.add_argument("--drive_backup_dir", type=str, default="/content/drive/MyDrive/BOM_Project", help="Google Drive backup path")
    args = parser.parse_args()
    
    try:
        package_artifacts(
            args.export_dir,
            args.run_dir,
            args.dataset_dir,
            args.zip_dest_dir,
            args.drive_backup_dir
        )
    except Exception as e:
        print(f"Packaging failed: {e}")
        import sys
        sys.exit(1)
