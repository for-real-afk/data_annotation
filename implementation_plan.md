# Implementation Plan - Door Anatomy Extraction & Registration Pipeline

We will complete the data annotations project by implementing the high-fidelity CAD/BIM-style door anatomy model with coordinate normalization, nested SVG transformation resolution, and automatic SVG-to-image registration.

## User Review Required

> [!IMPORTANT]
> The new SVG-to-Image registration pipeline is a critical blocker. We will reject any plan image pairs with an alignment confidence score of `< 0.85`, and flag plans with scores between `0.85` and `0.95` for manual review. This ensures high dataset quality but might reduce the total size of the generated dataset depending on how many floorplans fail registration.

## Proposed Changes

We will introduce several new modules and modify `main.py`, `qa_visualizer.py`, and `svg_parser.py` in `d:\projects\data_annotations`.

### Coordinates and Transforms

#### [NEW] [transform_resolver.py](file:///d:/projects/data_annotations/transform_resolver.py)
* Parse individual SVG elements' `transform` attributes (supporting `matrix`, `translate`, `scale`, `rotate`, `skewX`, `skewY`).
* Build $3 \times 3$ affine transformation matrices.
* Recursively resolve parent transformations from leaf element to root svg element by pre-multiplying.

#### [NEW] [coordinate_mapper.py](file:///d:/projects/data_annotations/coordinate_mapper.py)
* Encapsulates the combined transformation matrix $M_{\text{total}} = M_{\text{refine}} \times M_{\text{coarse}}$.
* Exposes `svg_to_image(x, y)` and `image_to_svg(x, y)` coordinate mappings.

### Wall Registration Engine

#### [NEW] [wall_extractor.py](file:///d:/projects/data_annotations/wall_extractor.py)
* Walk the SVG tree and extract all structural wall elements (using classes `Wall External`, `Wall Internal`, `Wall`).
* Apply nested parent transforms via `transform_resolver.py` to get global SVG coordinates.
* Render the walls onto a binary mask (`svg_wall_mask`) using coarse scaling/translation.

#### [NEW] [image_wall_detector.py](file:///d:/projects/data_annotations/image_wall_detector.py)
* Isolate walls from the raster PNG image using adaptive thresholding, simple thresholding, and morphological opening/closing (`image_wall_mask`).

#### [NEW] [svg_image_registrar.py](file:///d:/projects/data_annotations/svg_image_registrar.py)
* Registers the `svg_wall_mask` and `image_wall_mask` to estimate the refinement matrix $M_{\text{refine}}$ using `cv2.findTransformECC` at a downsampled resolution for speed and robustness.
* Fallback to ORB feature keypoint matching and RANSAC (`cv2.estimateAffinePartial2D`) if ECC fails.
* Compute an `alignment_score` based on the overlap between dilated warped SVG walls and image walls.

### Door Anatomy Extraction

#### [NEW] [door_anatomy_extractor.py](file:///d:/projects/data_annotations/door_anatomy_extractor.py)
* Orchestrates door anatomy extraction in global SVG space using:
  * `hinge_detector.py` to identify hinge coordinates.
  * `leaf_detector.py` to locate the door leaf line segment.
  * `arc_detector.py` to parse/sample swing curve points.
  * `orientation_calculator.py` and `opening_direction_calculator.py` to compute orientation/direction.
* Uses `polygon_generator.py` to construct a closed wedge polygon.
* Maps all points using `coordinate_mapper.py` to PNG coordinates.

#### [NEW] [polygon_generator.py](file:///d:/projects/data_annotations/polygon_generator.py)
* Assembles hinge, leaf, and arc coordinates into a closed wedge slice polygon.

#### [NEW] [drift_detector.py](file:///d:/projects/data_annotations/drift_detector.py)
* Detects annotations falling outside building envelope, in margins, or floating in empty space.

#### [NEW] [alignment_validator.py](file:///d:/projects/data_annotations/alignment_validator.py)
* Generates validation overlays: Layer 1 (Red SVG Walls), Layer 2 (Blue Image Walls), Layer 3 (Green Aligned Door Geometry).

### Orchestration & Exporters

#### [MODIFY] [svg_parser.py](file:///d:/projects/data_annotations/svg_parser.py)
* Retain utility functions but update layout scaling functions.

#### [MODIFY] [qa_visualizer.py](file:///d:/projects/data_annotations/qa_visualizer.py)
* Upgrade to draw door hinge (Red Dot), leaf (Blue Line), swing arc (Green Curve), and mask (Yellow Polygon).

#### [MODIFY] [main.py](file:///d:/projects/data_annotations/main.py)
* Integrates all registration and anatomy stages.
* Saves `alignment_metadata.json` and alignment validations.
* Filters out low-confidence plans (`score < 0.85`).
* Aggregates stats and door classifications.

## Verification Plan

### Automated Tests
* Run processing on `local_test_data`:
  `python main.py --data_dir local_test_data --output_dir local_dataset_output --qa_limit 10`
* Verify generation of outputs:
  * `local_dataset_output/images/` and `local_dataset_output/labels/`
  * `local_dataset_output/crops/` with metadata index
  * `local_dataset_output/qa/` and `local_dataset_output/qa_alignment/`
  * `local_dataset_output/stats.json` and `local_dataset_output/door_types.json`
* Run `verify_labels.py` on generated labels to ensure YOLO annotations overlap visual symbols.

### Manual Verification
* Visual inspect generated validation masks under `local_dataset_output/qa_alignment/` to verify Red (SVG) and Blue (Image) walls align.
* Check QA files under `local_dataset_output/qa/` to verify CAD door elements are aligned perfectly.
