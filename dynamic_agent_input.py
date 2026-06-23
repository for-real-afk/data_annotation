# Implementation Plan - Door Anatomy Extraction & Alignment Refinement

This plan details the refactoring of our floorplan door annotation pipeline from a bounding-box-centric approach to a high-fidelity CAD/BIM-style door anatomy model, with a **mandatory Coordinate Normalization Layer** to resolve nested SVG transformations, viewBox scaling, and image-space alignments.

---

## Proposed Architecture

We will modularize the codebase in the `data_annotations` directory into 13 specialized files and a main orchestrator:

```
data_annotations/
├── svg_parser.py                    # viewBox parsing, scale/offset transform calibration
├── transform_resolver.py            # [NEW] Nested parent transformation matrix resolver
├── coordinate_mapper.py             # [NEW] SVG-to-Image / Image-to-SVG converter
├── door_anatomy_extractor.py        # High-level anatomy orchestrator
├── hinge_detector.py                # shared point geometric threshold detector
├── leaf_detector.py                 # Straight door leaf segment finder
├── arc_detector.py                  # Elliptical arc and Bezier curve parser
├── orientation_calculator.py        # leaf vector arctan2 calculation & 90-degree normalization
├── opening_direction_calculator.py  # Path sweep_flag & sampled vector cross-product sweep calculator
├── polygon_generator.py             # Wedge-shaped pie slice polygon assembler
├── segmentation_exporter.py         # YOLO detection (bbox) and segmentation (polygon) formatter
├── qa_visualizer.py                 # Upgraded visual verification overlay generator
├── geometry_validator.py            # [NEW] Raw/Transformed/Final debug overlay validator
└── main.py                          # Multiprocessing directory processor & stats compiler
```

---

## Technical Details by Stage

### Coordinate Normalization Layer

To prevent shifting, offsets, or stretched geometries on rendered floorplan images, we implement a strict sequence of coordinate resolution before extracting any anatomy:

```
SVG Element (polygon, path, line, rect)
 ↓
Resolve Parent & Local Transforms (translate, scale, rotate, matrix, skewX/Y)
 ↓
Transform Coordinates to Global SVG Space
 ↓
Map Global SVG Coordinates to PNG Pixel Coordinates (viewBox, scale, dx, dy)
 ↓
Extract Door Anatomy Components (Hinge, Leaf, Arc)
```

### Stage X: Transformation Resolver (`transform_resolver.py`)
- Parse the `transform` attribute string of any SVG node using regular expressions.
- Build $3 \times 3$ affine transformation matrices representing `translate`, `scale`, `rotate`, `matrix`, `skewX`, and `skewY`.
- Recursively resolve parent transformations from leaf to root by pre-multiplying:
  $$M_{\text{global}} = M_{\text{ancestor}} \times M_{\text{current}}$$
- Provide `apply_transform(point, matrix)` to convert local coordinates into the global SVG space.

### Stage X+2: SVG-to-Image Mappings (`coordinate_mapper.py`)
- Takes viewBox dimensions and PNG dimensions.
- Combines structural alignment metrics (scale, dx, dy) to expose:
  - `svg_to_image(x, y)`: Maps global SVG coordinate to image pixel coordinate.
  - `image_to_svg(x, y)`: Maps image pixel coordinate back to SVG coordinate space.

### Stage 1: Door Anatomy Extraction (`door_anatomy_extractor.py`)
For each SVG door group, we extract:
- Hinge coordinate `[x, y]`
- Leaf segment `[[x1, y1], [x2, y2]]`
- Arc points `[[x1, y1], [x2, y2], ...]`

### Stage 2 & 3: Hinge & Door Leaf Detection (`hinge_detector.py`, `leaf_detector.py`)
- We search the parsed segments for curve elements (`QuadraticBezier`, `CubicBezier`, `Arc`) representing the swing arc.
- Let the endpoints of the curve be $A_{\text{start}}$ and $A_{\text{end}}$.
- We search for a straight `Line` segment (representing the door leaf) that connects to $A_{\text{end}}$ (the open position of the leaf).
- The leaf segment endpoints are $L_{\text{start}}$ and $L_{\text{end}}$, where $L_{\text{end}} \approx A_{\text{end}}$.
- The other endpoint, $L_{\text{start}}$, is identified as the **Hinge ($H$)**.
- If no connecting line is found, we fall back to finding the threshold bounding box corners. The corner closest to the expected center of rotation is classified as the Hinge, and the leaf is modeled as the segment from the hinge to $A_{\text{end}}$.

### Stage 4 & 6: Swing Arc & Opening Direction (`arc_detector.py`, `opening_direction_calculator.py`)
- The swing arc is extracted by sampling points along the curve segment.
- For `Arc` segments, we inspect the SVG `sweep_flag` directly (`arc_seg.sweep`). If `True` (1), it is clockwise (`CW`); if `False` (0), it is counter-clockwise (`CCW`).
- For Bezier curves (`QuadraticBezier`, `CubicBezier`), we compute the sum of signed angular differences of sampled points relative to the hinge $H$:
  $$\theta_i = \text{atan2}(P_{i, y} - H_y, P_{i, x} - H_x)$$
  $$\text{sweep} = \sum_{i} \text{normalize\_angle}(\theta_{i+1} - \theta_i)$$
  - $\text{sweep} > 0 \implies \text{CW}$
  - $\text{sweep} < 0 \implies \text{CCW}$

### Stage 5: Leaf-based Orientation (`orientation_calculator.py`)
We compute the leaf orientation using the open leaf segment vector (from Hinge $H$ to Leaf End $L_{\text{end}}$):
$$\theta = \text{atan2}(L_{\text{end}, y} - H_y, L_{\text{end}, x} - H_x) \times \frac{180}{\pi}$$
- We normalize the angle to $[0^\circ, 360^\circ)$.
- We round it to the nearest $90^\circ$ cardinal angle: $0^\circ, 90^\circ, 180^\circ, 270^\circ$.

### Stage 7: Tight Polygon Assembly (`polygon_generator.py`)
We build the tight segment polygon by creating a closed wedge shape:
- Start at Hinge $H$.
- Follow the door leaf line to Leaf End $L_{\text{end}}$.
- Follow the sampled swing arc curve points from $L_{\text{end}}$ to Arc Start $A_{\text{start}}$.
- Return back to Hinge $H$ (forming a closed wedge slice).

### Stage 8: Exporter (`segmentation_exporter.py`)
Generates annotations in two formats:
- **Labels Detection**: YOLO BBox format (`class_id x_center y_center width height`).
- **Labels Segmentation**: YOLO Polygon format (`class_id x1 y1 x2 y2 ...`) using the tight polygon coordinates.

### Stage 9 & X+3: Upgraded QA Verification & Alignment Debugger (`qa_visualizer.py`, `geometry_validator.py`)
Draws high-visibility graphics over plan PNGs:
- **Red Dot / Red overlay**: Raw Local Geometry (local space).
- **Green Line/Curve / Green overlay**: Transformed Global SVG Geometry.
- **Blue Line/Curve / Blue overlay**: Final mapped segmentation wedge.
- Writes validation debug overlays to `qa_alignment/` for visual verification of coordinate alignment correctness.

### Stage X+4 — Automatic SVG ↔ Image Registration (Critical)

#### Problem Statement
Current observations from QA:
* Door Detection: OK
* Door Classification: OK
* Door Geometry: OK
* Door Overlay Alignment: FAILED

Examples observed:
* Door correctly found, door polygon reasonable, door box close to actual location BUT overlay shifted, overlay floating outside walls, overlay appears in margins, overlay appears inside text regions, overlay partially aligned.
* This indicates that extraction is largely correct, but SVG geometry and raster image coordinates are not perfectly aligned.
* This issue must be solved before generating training labels.

#### Root Cause Hypothesis
The system currently assumes SVG Coordinate Space = Rendered Image Coordinate Space. This assumption is invalid due to:
1. **ViewBox Scaling Mismatch**: SVG coordinates differ from PNG coordinates and lack proper scaling.
2. **Renderer Padding**: Rendered PNGs have margins/paddings not in the SVG.
3. **Image Cropping**: Renderers may trim whitespace or center the floorplan.
4. **Nested SVG Transformations**: Ignored `transform` attributes result in wrong positioning.
5. **Separate Raster Generation Pipeline**: Annotations and images from different pipelines.

#### New Architecture
```
SVG -> Transform Resolution -> Coordinate Normalization -> SVG ↔ Image Registration -> Door Extraction -> Aligned Annotations
```

#### Stage X+4.1 — Structural Wall Alignment
Extract wall structures first.
* **SVG Side**: Use classes `Wall External`, `Wall Internal`, `Wall` to generate `svg_wall_mask`.
* **Raster Side**: Use OpenCV (`threshold`, `adaptiveThreshold`, `Canny`) to isolate `image_wall_mask`.

#### Stage X+4.2 — Global Registration
Estimate transformation between `svg_wall_mask` and `image_wall_mask`. Compute `scale_x`, `scale_y`, `translation_x`, `translation_y`, and `rotation` using `cv2.findTransformECC()`, `cv2.estimateAffinePartial2D()`, or `cv2.findHomography()`.

#### Stage X+4.3 — Alignment Matrix
Generate `alignment_matrix` containing scale, translation, rotation and store it in `alignment_metadata.json` for every floorplan.

#### Stage X+4.4 — Apply Registration
Before exporting labels, transform all geometries: `aligned_point = alignment_matrix @ svg_point`.

#### Stage X+4.5 — Registration Validation
Generate validation visualization:
* Layer 1: Red = SVG Walls
* Layer 2: Blue = Image Walls
* Layer 3: Green = Aligned Door Geometry

#### Stage X+4.6 — Alignment Confidence Score
Compute `alignment_score` in range [0.0, 1.0] for every plan.
* Accept: `>= 0.95`
* Review manually: `0.85 – 0.95`
* Reject: `< 0.85`

#### Stage X+4.7 — Automatic Drift Detection
Detect annotations falling outside building envelope, in margins, watermarks, footer text, or not overlapping walls/openings and flag them as `{"status": "misaligned"}`.

### New QA Requirements
A door annotation is valid ONLY if:
* Door Extracted Correctly + Door Geometry Correct + Door Orientation Correct + Door Opening Direction Correct + Door Polygon Correct + Door Properly Registered To Raster Image.
* Registration failure invalidates the annotation.

### New Metrics
* **Registration Accuracy**: >98% (Aligned SVG geometry overlaps floorplan structures).
* **Door Overlay Accuracy**: >98% (Door polygon overlaps actual door symbol in image).
* **Misalignment Rate**: <1% (Door annotations shifted from symbols).

### Additional Modules
Add:
* `transform_resolver.py`
* `coordinate_mapper.py`
* `wall_extractor.py`
* `image_wall_detector.py`
* `svg_image_registrar.py`
* `alignment_validator.py`
* `drift_detector.py`

### Updated Final Success Condition
Pipeline is complete when:
* Door Detection: >99%
* Orientation Accuracy: >95%
* Opening Direction: >95%
* Polygon Accuracy: >95%
* Registration Accuracy: >98%
* Door Overlay Accuracy: >98%

### Stage 10: Metadata dataset (`main.py`)
Compiles a global `door_metadata.json` containing all extracted door anatomical dictionaries, along with individual JSON files per plan under `dataset/metadata/`.

---

## Proposed Changes

We will create/rewrite the following files in `data_annotations`:

### [NEW] [transform_resolver.py](file:///d:/projects/data_annotations/transform_resolver.py)
Implements SVG transform attribute parsing and parent matrix multiplication.

### [NEW] [coordinate_mapper.py](file:///d:/projects/data_annotations/coordinate_mapper.py)
Maps coordinates between global SVG ViewBox space and image pixel space.

### [NEW] [geometry_validator.py](file:///d:/projects/data_annotations/geometry_validator.py)
Generates alignment validation overlays: Red (Raw), Green (Transformed), and Blue (Final).

### [MODIFY] [svg_parser.py](file:///d:/projects/data_annotations/svg_parser.py)
Updates geometry collection functions to resolve transforms before compiling bounds.

### [NEW] [door_anatomy_extractor.py](file:///d:/projects/data_annotations/door_anatomy_extractor.py)
Orchestrates Stages 1-10 for each door node element.

### [NEW] [hinge_detector.py](file:///d:/projects/data_annotations/hinge_detector.py)
Geometrically locates the hinge coordinates from parsed segments.

### [NEW] [leaf_detector.py](file:///d:/projects/data_annotations/leaf_detector.py)
Locates the door leaf line segment (start, end, length).

### [NEW] [arc_detector.py](file:///d:/projects/data_annotations/arc_detector.py)
Parses curves into sequential sampled coordinates.

### [NEW] [orientation_calculator.py](file:///d:/projects/data_annotations/orientation_calculator.py)
Computes and normalizes leaf-based rotation degrees.

### [NEW] [opening_direction_calculator.py](file:///d:/projects/data_annotations/opening_direction_calculator.py)
Computes opening direction (CW/CCW) directly from path sweep properties or angles.

### [NEW] [polygon_generator.py](file:///d:/projects/data_annotations/polygon_generator.py)
Assembles Hinge, Leaf, and Arc points into a continuous wedge polygon.

### [NEW] [segmentation_exporter.py](file:///d:/projects/data_annotations/segmentation_exporter.py)
Formats YOLO detection and segmentation annotation strings.

### [MODIFY] [qa_visualizer.py](file:///d:/projects/data_annotations/qa_visualizer.py)
Upgrades rendering to draw Red Dots (hinges), Blue Lines (leaves), Green Curves (arcs), and Yellow Polygons (masks).

### [MODIFY] [main.py](file:///d:/projects/data_annotations/main.py)
Configures directory structure, runs parallel processing using 4 workers, renames crops sequentially, and outputs JSON metadata.

---

## Verification Plan

### Automated Tests
- Run `python main.py --limit 100 --output_dir dataset` to run on a random subset of 100 plans.
- Verify that `dataset/labels_detection/` and `dataset/labels_segmentation/` are fully populated.
- Verify that `dataset/metadata/` contains the JSON files and `door_metadata.json` is generated.
- Verify that `dataset/stats/stats.json` and `dataset/stats/door_types.json` exist.

### Manual QA Check
- Inspect 10 overlays in `qa_alignment/` to ensure the Raw Geometry (Red) matches the Transformed Geometry (Green) after normalization.
- Inspect 10 overlays in `dataset/qa/` to ensure the hinge (Red Dot), door leaf (Blue Line), swing arc (Green Curve), and mask (Yellow Polygon) align perfectly with the rendered symbols.
