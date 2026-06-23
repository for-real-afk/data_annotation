# Floorplan Door Anatomy Extraction & YOLO Training Pipeline

A production-grade, end-to-end computer vision and data engineering pipeline for architectural floorplan understanding. This project maps SVG CAD geometries to raster floorplans, extracts high-fidelity door/window segmentations, and trains a robust YOLO instance segmentation model on Google Colab for automated quantity takeoff and Bill of Materials (BOM) estimation.

---

## 📂 Repository Structure

The codebase is organized into clean, modular subdirectories:

```text
├── colab/                    # YOLO Colab Training, Config & Inference
│   ├── train.ipynb           # Master Jupyter Notebook for Colab T4 training
│   ├── dataset.yaml          # YOLO dataset split classes configuration
│   ├── train_config.yaml     # Custom hyperparameters & CAD augmentations
│   ├── export_model.py       # Exporter (best.pt -> ONNX, TorchScript, PKL)
│   ├── package_model.py      # Packager (curves, logs, dataset and model zips)
│   ├── test_real_floorplan.py# Command-line utility for local inference
│   ├── metrics.json          # Metrics schema template
│   └── door_metadata.json    # Target metadata schema template
│
├── src/                      # Core SVG Parsing & Registration Pipeline
│   ├── transform_resolver.py # Nested SVG parent transform solver
│   ├── coordinate_mapper.py  # SVG-to-raster coordinate alignment system
│   ├── wall_extractor.py     # Structural SVG wall extractor with RANSAC
│   ├── image_wall_detector.py# Morphological image wall detector
│   ├── svg_image_registrar.py# Nelder-Mead scale, translation, rotation optimizer
│   ├── door_extractor.py     # High-level door node classifier
│   ├── door_anatomy_extractor.py # Main door anatomy vector solver
│   ├── hinge_detector.py     # Door hinge detector
│   ├── leaf_detector.py      # Door leaf segment & swing solver
│   ├── arc_detector.py       # Hinge swing arc sampler
│   ├── orientation_calculator.py # Door orientation angle solver
│   ├── opening_direction_calculator.py # Opening direction (CW/CCW) calculator
│   ├── polygon_generator.py  # Assembles closed door wedge segments
│   ├── drift_detector.py     # Annotation drift and OOB filter
│   ├── alignment_validator.py# Visual alignment check generator
│   ├── segmentation_generator.py # YOLO label generator
│   ├── qa_visualizer.py      # Visual verification exporter
│   ├── dataset_stats.py      # Global stats aggregator
│   └── svg_parser.py         # SVG viewBox and layout helper
│
├── main.py                   # Root execution CLI orchestrator
├── verify_labels.py          # Visual label overlay checks script
├── debug_worker.py           # Submodule debugger
├── run_worker.py             # Parallel multi-core execution runner
├── .gitignore                # Clean Git exclusions (datasets, zip, pt)
└── README.md                 # Project Documentation (this file)
```

---

## 🛠️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd data_annotations
   ```

2. **Set up Virtual Environment**:
   ```bash
   python -m venv env
   # Activate on Windows:
   env\Scripts\activate
   # Activate on macOS/Linux:
   source env/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   # Requirements include: opencv-python, numpy, beautifulsoup4, lxml, ultralytics, cairosvg, svgpathtools, pandas, pyyaml
   ```

---

## 🚀 Usage Guide

### 1. Run the Local Processing Pipeline
Extract doors, align coordinates, run the optimization engine, and export visual annotations:
```bash
python main.py --output_dir dataset --workers 8 --qa_limit 100
```
This command:
- Processes raw floorplan images and SVGs.
- Registers SVG walls to Canny wall lines using multi-start Nelder-Mead optimization.
- Extracts hinges, leaves, and swing curves, generating YOLO segment label coordinates.
- Rejects any layouts with alignment scores `< 0.85`.

### 2. Run Google Colab Training
To train the model on a Google Colab T4 GPU:
1. Upload `colab/train.ipynb` and `colab/` folder contents to Google Colab.
2. Ensure runtime type is set to **GPU**.
3. Run the cells. The notebook will automatically:
   - Load the Kaggle CubiCasa5K dataset.
   - Render SVGs directly to PNG at `imgsz=1024` for high resolution.
   - Split dataset to 80% Train, 10% Val, 10% Test.
   - Run validation checks and log `dataset_report.json`.
   - Train `yolo11s-seg.pt` for 100 epochs (auto-resuming from Google Drive if a crash occurs).
   - Export weights to ONNX, TorchScript, and `bom_model.pkl`.
   - Package all models, validation logs, metrics, and data into `bom_training_bundle.zip` and `dataset_export.zip` and back them up directly to your Drive `/content/drive/MyDrive/BOM_Project/`.

### 3. Local Real-World Inference Check
Once training completes, use the local inference utility to check custom floorplan images:
```bash
python colab/test_real_floorplan.py \
    --weights colab/best.pt \
    --image floorplan.png \
    --output_dir predictions
```
**Outputs**:
- Printed door and window counts in the console.
- `prediction_floorplan.png` showing predicted door and window masks.
- `inference_report.json` detailing detections and polygons.

---

## ⚙️ Core Architecture Details

- ** Nelder-Mead Optimizer**: stage 1 coarse grid-search over translation followed by stage 2 Nelder-Mead refinement on aspect ratio scale and rotation. Features a clipping distance loss map to prevent outlier drift.
- **Door Geometry Solver**: Parses lines and Bézier/Arc curves inside structural doors to automatically locate hinges (reconstruction vertex), door leaf lines (opening vector), and swing curves (opening radius).
- **Synthetic Renderer**: To bypass manual registration during Colab training, `train.ipynb` renders the SVG vectors directly to pixel-perfect PNG files, eliminating alignment shifts entirely.
