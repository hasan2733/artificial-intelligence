# YOLO — Full Architecture & Intuition Notes
> Personal revision notes built from your questions. Read top to bottom for a complete mental model.

---

## 1. What is YOLO and Why is it Special?

**YOLO = You Only Look Once**

Before YOLO, detectors like R-CNN worked in two stages:
1. First find candidate regions
2. Then classify each region

This was slow. YOLO does both in a **single forward pass** through a neural network. One look at the image → output all objects with their locations.

---

## 2. The Big Picture Pipeline

```
Input Image (640×640×3)
        ↓
  CNN Backbone
  (feature extraction)
        ↓
  Feature Maps (multi-scale)
        ↓
  Detection Head
  (predict boxes + classes)
        ↓
  NMS (Non-Max Suppression)
        ↓
  Final Detected Objects
```

---

## 3. Step 1 — CNN Backbone (Feature Extraction)

The full image goes into the CNN backbone (e.g., CSPDarknet in YOLOv8).

The CNN applies **filters/kernels** across the image to extract features.

### What is a Kernel/Filter?
A kernel is a small matrix (e.g., 3×3) that slides over the image and detects patterns.

```
Image patch:        Kernel (edge detector):
1  2  3             -1  0  1
4  5  6      ×      -1  0  1
7  8  9             -1  0  1
```
Multiply element-wise, sum → one output value.

Each filter detects something specific:
- Early layers → edges, corners, textures
- Deeper layers → eyes, wheels, faces, animal shapes

### What is a Feature Map?
After passing through CNN layers, you get a **feature map** like:

```
80 × 80 × 256
```

- `80 × 80` = spatial locations (WHERE in the image)
- `256` = number of feature channels (WHAT is detected at each location)

Each spatial location has a **256-dimensional vector** describing what features are present there.

### Why does the spatial size shrink?
CNN uses **stride** (step size). With stride = 8:

```
640 / 8 = 80
```

So a 640×640 image becomes an 80×80 feature map. Each cell in the feature map represents an **8×8 pixel region** of the original image.

> **Key insight:** You don't manually divide the image into a grid. The CNN produces a feature map — that feature map IS the grid.

---

## 4. Step 2 — Multi-Scale Detection

Objects can be big or small. YOLO uses feature maps at **multiple resolutions**:

| Feature Map | Detects |
|-------------|---------|
| 80 × 80     | Small objects |
| 40 × 40     | Medium objects |
| 20 × 20     | Large objects |

This is called the **FPN (Feature Pyramid Network)** or "neck" in YOLO.

```
Image → Backbone → Neck (Feature Fusion) → 3 Detection Heads
                                              ↓     ↓     ↓
                                           80×80  40×40  20×20
```

---

## 5. Step 3 — Detection Head (Predictions per Cell)

Each cell in the 80×80 feature map predicts:

```
x       ← x center of bounding box (offset within cell)
y       ← y center of bounding box (offset within cell)
w       ← width of bounding box
h       ← height of bounding box
conf    ← confidence that an object exists here
class   ← probability for each class (e.g., dog=0.90, cat=0.05)
```

For 80 COCO classes, each cell predicts **5 + 80 = 85 values**.

Full output tensor for one scale:
```
80 × 80 × 85   (for anchor-based YOLO)
```

---

## 6. How Object Center is Detected

YOLO does NOT geometrically calculate the center. It **learns to predict it** from training data.

### Grid Cell Assignment (Math)

Given a normalized annotation:
```
x = 0.52,  y = 0.63
```

Step 1 — Scale to grid space (grid size S = 80):
```
x_grid = 0.52 × 80 = 41.6
y_grid = 0.63 × 80 = 50.4
```

Step 2 — Find the responsible cell (floor):
```
i = floor(41.6) = 41
j = floor(50.4) = 50
→ Cell (41, 50) is responsible
```

Step 3 — The cell stores the **offset** (fractional part):
```
x_offset = 0.6
y_offset = 0.4
```

Step 4 — Final coordinate reconstruction:
```
b_x = sigmoid(t_x) + c_x
b_y = sigmoid(t_y) + c_y
```
Where `c_x, c_y` = top-left of the cell. Sigmoid keeps values in [0, 1].

### Visualization
```
80×80 grid:

[ ][ ][ ][ ][ ][ ]
[ ][ ][ ][ ][ ][ ]
[ ][ ][ ][ ][ ][ ]
[ ][ ][ ][ ][ ][ ]
[ ][ ][ ][ ][ ][ ]
[ ][ ][ ][ ][ ][ ]
[ ][ ][ ][ ][ ][ ]
...........
            [X]        ← cell (41, 50) is responsible for this object
...........
```

Only ONE cell is responsible per object. All other cells predict "no object."

---

## 7. Step 4 — NMS (Non-Max Suppression)

For one dog, YOLO might predict 10–20 overlapping boxes:
```
Box A: confidence = 0.95
Box B: confidence = 0.91
Box C: confidence = 0.87
```

NMS removes duplicates and keeps only the most confident box.

**How NMS works:**
1. Sort boxes by confidence
2. Keep the highest confidence box
3. Remove all other boxes with high IoU (overlap) with the kept box
4. Repeat

**IoU = Intersection over Union:**
```
IoU = Area of Overlap / Area of Union
```
If IoU > threshold (e.g., 0.5) → boxes cover the same object → remove the weaker one.

---

## 8. Final Output Format

For each detected object, YOLO outputs:

```python
{
  "class": "dog",
  "confidence": 0.97,
  "box": [x1, y1, x2, y2]   # top-left and bottom-right corners
}
```

### Bounding Box Formats in Ultralytics YOLOv8

```python
results = model("image.jpg")
boxes = results[0].boxes

boxes.xyxy    # [x1, y1, x2, y2] — corner format
boxes.xywh    # [x_center, y_center, width, height]
boxes.xywhn   # normalized version (0–1)
boxes.conf    # confidence scores
boxes.cls     # class IDs
```

### Getting Center from Corner Coordinates
```
x_center = (x1 + x2) / 2
y_center = (y1 + y2) / 2
```

---

## 9. Training — What Labels Do You Provide?

### You only provide object annotations, NOT grid vectors.

For one image with a cat:
```
cat.txt:
0  0.52  0.41  0.30  0.25
```

Format: `class_id  x_center  y_center  width  height`
All values are **normalized (0 to 1)**.

For an image with multiple objects:
```
0  0.45  0.30  0.20  0.25   ← cat
1  0.72  0.65  0.18  0.22   ← dog
```

### What YOLO does internally:
1. Reads your label file (N objects × 5 values)
2. Finds which grid cell each object center falls in
3. Assigns that cell as "positive" (responsible)
4. Marks all other cells as "negative" (no object)
5. Builds the full target tensor automatically

> You give: `(N, 5)` shaped labels
> YOLO builds: `(80×80×85)` shaped target tensor internally

---

## 10. Loss Function

YOLO uses multiple losses, combined:

```
Total Loss = Box Loss + Class Loss + Confidence Loss
```

### Box Loss (Center)
```
L_center = (x - x̂)² + (y - ŷ)²
```

### Width/Height Loss
```
L_size = (w - ŵ)² + (h - ĥ)²
```

### Confidence Loss
- Positive cell (object exists): `conf` should be close to 1
- Negative cell (no object): `conf` should be close to 0

### Class Loss
Cross-entropy between predicted class probabilities and ground truth class.

### Backpropagation
```
Prediction:  center = (300, 220)
Ground truth: center = (320, 240)
→ Loss is high → backprop → weights updated → next prediction is closer
```

The model learns: "what pattern in the feature map corresponds to what bounding box."

---

## 11. Dataset Structure for Training

```
dataset/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── val/
│       └── img3.jpg
└── labels/
    ├── train/
    │   ├── img1.txt
    │   └── img2.txt
    └── val/
        └── img3.txt
data.yaml
```

`data.yaml`:
```yaml
train: images/train
val: images/val
nc: 2
names: ['cat', 'dog']
```

---

## 12. Running YOLO in Google Colab

### Install
```python
pip install ultralytics
```

### Inference (Pretrained)
```python
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model = YOLO("yolov8n.pt")
results = model("image.jpg", imgsz=640, conf=0.25)

# Correct color display (BGR → RGB)
img = results[0].plot()
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 8))
plt.imshow(img_rgb)
plt.axis("off")
plt.show()
```

> `results[0].plot()` returns BGR (OpenCV format). `matplotlib` needs RGB. Always convert with `cv2.cvtColor`.

### Multiple Images
```python
# From folder
results = model("images/", imgsz=640)

# From list
results = model(["img1.jpg", "img2.jpg", "img3.jpg"])

# From Google Drive
results = model("/content/drive/MyDrive/test_images/")
```

### Custom Training
```python
model = YOLO("yolov8n.pt")
model.train(
    data="data.yaml",
    epochs=20,
    imgsz=640
)
```

### Image Size Guide
| imgsz | Speed | Memory | Best for |
|-------|-------|--------|----------|
| 320   | Fast  | Low    | Practice |
| 640   | Medium| Medium | Standard |
| 1280  | Slow  | High   | Small objects |

---

## 13. Training Metrics to Watch

| Metric | Meaning |
|--------|---------|
| Box Loss | How accurate are the bounding boxes |
| Class Loss | How accurate are the class predictions |
| Confidence Loss | How well does the model know when objects are present |
| mAP | Mean Average Precision — overall detection quality |
| Precision | Of all predicted boxes, how many were correct |
| Recall | Of all actual objects, how many were found |

---

## 14. Annotation Tools

You cannot train without bounding box annotations. Use these tools to draw boxes:

| Tool | Notes |
|------|-------|
| **LabelImg** | Simple desktop app, exports YOLO format |
| **CVAT** | Web-based, supports many formats |
| **Roboflow** | Web-based, also handles augmentation and export |

These tools let you draw boxes manually → they compute and save the normalized `(x, y, w, h)` for you.

---

## 15. Suggested Learning Path

```
1. CNN recap (conv, pooling, activation)
2. Bounding box, IoU concept
3. Non-Max Suppression (NMS)
4. YOLO architecture (this document)
5. YOLOv8 inference on pretrained model
6. Custom dataset annotation (LabelImg / Roboflow)
7. Custom training in Colab
8. Evaluate: mAP, Precision, Recall
```

---

## 16. Quick Reference — Key Formulas

| Concept | Formula |
|---------|---------|
| Grid cell index | `i = floor(x × S)` |
| Center offset | `b_x = sigmoid(t_x) + c_x` |
| Center from corners | `x_center = (x1 + x2) / 2` |
| IoU | `Intersection Area / Union Area` |
| Center loss | `(x - x̂)² + (y - ŷ)²` |
| Feature map size | `image_size / stride` |

---

## 17. Common Confusion Points — Cleared

| Confusion | Truth |
|-----------|-------|
| "I need to give 80×80 labels" | ❌ No. You give N object labels. YOLO maps them to 80×80 internally. |
| "Grid is made before CNN" | ❌ No. CNN runs first, feature map = grid. |
| "YOLO calculates center geometrically" | ❌ No. It learns to predict the center from patterns. |
| "results[0].plot() is RGB" | ❌ No. It's BGR. Convert with `cv2.cvtColor`. |
| "Only class label needed for training" | ❌ No. Bounding box is mandatory. Without it, there's no location to learn. |

---

*End of Notes — Good luck with your YOLO journey!*
