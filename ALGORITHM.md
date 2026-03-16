# Algorithm Deep Dive: Autonomous Landing Vision System

## Table of Contents
1. [Terrain Classification](#terrain-classification)
2. [Roughness Analysis](#roughness-analysis)
3. [Heatmap Generation](#heatmap-generation)
4. [Coordinate Transformation](#coordinate-transformation)
5. [Mathematical Formulations](#mathematical-formulations)

---

## 1. Terrain Classification

### 1.1 OpenCV Method (Default)

#### Color-Based Segmentation (HSV Space)

**Why HSV?**
- Separates color (Hue) from intensity (Value)
- More robust to lighting changes than RGB
- Better for outdoor environments

**Grass Detection:**
```python
H_grass ∈ [35°, 85°]    # Green hue range
S_grass ∈ [40, 255]     # Moderate to high saturation
V_grass ∈ [40, 255]     # Avoid very dark pixels

grass_mask = inRange(HSV, [35, 40, 40], [85, 255, 255])
grass_ratio = count(grass_mask) / total_pixels
```

**Water Detection:**
```python
H_water ∈ [90°, 130°]   # Blue/cyan hue range
S_water ∈ [30, 255]     # Can have low saturation (reflections)
V_water ∈ [30, 255]     # Wide brightness range

water_mask = inRange(HSV, [90, 30, 30], [130, 255, 255])
water_ratio = count(water_mask) / total_pixels
```

**Concrete Detection:**
```python
S_concrete ∈ [0, 50]    # Low saturation (gray/achromatic)
V_concrete ∈ [60, 220]  # Medium brightness (not too dark/bright)
H_concrete = Any        # Hue irrelevant for achromatic colors

# Extract saturation and value channels
s_channel = HSV[:, :, 1]
v_channel = HSV[:, :, 2]

concrete_mask = inRange(s_channel, 0, 50) AND 
                inRange(v_channel, 60, 220)
concrete_ratio = count(concrete_mask) / total_pixels
```

#### Texture Analysis (Laplacian Variance)

Grass typically has higher texture variance due to blade structure:

```python
gray = cvtColor(frame, GRAY)
laplacian = Laplacian(gray, CV_64F)
texture_variance = var(laplacian)

# Decision thresholds
if texture_variance > 100:
    # High texture → likely grass
elif texture_variance < 50:
    # Low texture → might be water
```

**Why Laplacian?**
- Detects edges and rapid intensity changes
- Grass has many micro-edges (blade tips, shadows)
- Water surfaces are smooth (low Laplacian response)

#### Concrete Detection (Color Uniformity)

Concrete is achromatic (gray), making saturation the key indicator:

```python
# Concrete mask based on low saturation
s_channel = HSV[:, :, 1]
v_channel = HSV[:, :, 2]

concrete_mask = (s_channel < 50) AND (60 < v_channel < 220)
concrete_ratio = count(concrete_mask) / total_pixels

# Color uniformity check
color_std = mean([std(B), std(G), std(R)])
color_uniformity = 1 - min(color_std / 50, 1.0)

# Texture check (concrete is very smooth)
if concrete_ratio > 0.3 and texture_variance < 80 and color_uniformity > 0.7:
    # Confident concrete detection
```

**Why Low Saturation?**
- Gray colors have R ≈ G ≈ B
- In HSV, equal RGB values → low saturation
- Differentiates from colored grass/water

**Texture Comparison:**
```
Concrete:  σ_texture < 80   (very smooth, uniform)
Grass:     σ_texture > 100  (varied, textured)
Water:     σ_texture 40-60  (smooth with ripples)
```

#### Reflection Detection

Water exhibits specular reflections (bright spots):

```python
V_channel = HSV[:, :, 2]  # Brightness channel
bright_mask = threshold(V_channel, 200, 255)

bright_ratio = count(bright_mask) / total_pixels

# Check for clustered bright spots (water reflections)
components = connectedComponents(bright_mask)
avg_area = mean(component_areas)

if avg_area > 50 and bright_ratio > threshold:
    # Large, sparse bright regions → water reflections
```

#### Decision Logic

```python
def classify(frame):
    grass_score = grass_ratio + texture_variance/1000
    water_score = water_ratio + reflection_score * 0.5
    concrete_score = concrete_ratio + color_uniformity * 0.3
    
    # Prioritize by distinctiveness
    if concrete_ratio > 0.3 and texture_variance < 80 and color_uniformity > 0.7:
        return CONCRETE, concrete_score
    elif water_score > 0.45:
        return WATER, water_score
    elif grass_score > 0.75:
        return GRASS, grass_score
    else:
        # Compare coverage ratios
        if concrete_ratio > max(grass_ratio, water_ratio):
            return CONCRETE, concrete_ratio
        elif grass_ratio > water_ratio:
            return GRASS, grass_ratio
        else:
            return WATER, water_ratio
```

**Why This Order?**
1. **Concrete first**: Most distinctive (achromatic + very smooth)
2. **Water second**: Highly distinctive (reflections + blue hue)
3. **Grass third**: Common but textured
4. **Coverage fallback**: If no strong indicators, choose highest coverage

### 1.2 TensorFlow Lite Method (Optional)

**Model Architecture:**
- Base: MobileNetV2 or EfficientNet-Lite0
- Input: 224×224×3 RGB
- Output: 2 classes (Grass, Water) + confidence

**Preprocessing:**
```python
# Resize and normalize
input_img = cv2.resize(frame, (224, 224))
input_img = (input_img / 127.5) - 1.0  # Normalize to [-1, 1]

# Inference
interpreter.set_tensor(input_index, input_img)
interpreter.invoke()
output = interpreter.get_tensor(output_index)

# Softmax probabilities
probs = softmax(output)
class_id = argmax(probs)
confidence = probs[class_id]
```

**Performance Trade-off:**
| Method | Accuracy | Speed (Pi 4) | Robustness |
|--------|----------|--------------|------------|
| OpenCV | 85-90% | 30ms | Medium |
| TFLite | 95%+ | 80ms | High |

---

## 2. Roughness Analysis

### 2.1 Grid-Based Scoring

Divide frame into N×N grid (default: 32×32):

```python
cell_height = frame.height / N
cell_width = frame.width / N

for i in range(N):
    for j in range(N):
        cell = frame[i*h : (i+1)*h, j*w : (j+1)*w]
        score[i, j] = analyze_cell(cell)
```

### 2.2 Texture Safety Score

**Metric:** Standard deviation of pixel intensities

```
σ_texture = √(Σ(I_i - μ)² / n)

safety_texture = 1 - min(σ_texture / 60, 1.0)
```

**Interpretation:**
- Low σ (< 20): Smooth, uniform surface → Safe (score ≈ 0.9)
- Medium σ (20-40): Normal grass texture → Moderate (score ≈ 0.6)
- High σ (> 50): Rough, varied surface → Unsafe (score ≈ 0.2)

### 2.3 Edge Safety Score

**Metric:** Edge density from Canny edge detection

```
edges = Canny(cell, threshold1=50, threshold2=150)
edge_density = count_nonzero(edges) / cell_area

safety_edge = 1 - min(edge_density * 5.0, 1.0)
```

**Interpretation:**
- Few edges: Flat surface, no obstacles → Safe
- Many edges: Sharp transitions, potential obstacles → Unsafe

**Why multiply by 5?**
- Typical safe grass: edge_density ≈ 0.02-0.05
- Obstacles/rocks: edge_density > 0.1
- Scaling factor ensures unsafe zones score < 0.5

### 2.4 Color Uniformity Score

**Metric:** Color variance across BGR channels

```
σ_color = [σ_B, σ_G, σ_R]
σ_avg = mean(σ_color)

safety_color = 1 - min(σ_avg / 50, 1.0)
```

**Interpretation:**
- Uniform color: Consistent grass type → Safe
- Varied color: Mixed terrain, potential hazards → Less safe

### 2.5 Shadow Detection Score

**Metric:** Proportion of dark pixels (potential holes/depressions)

```
dark_threshold = 50  # Low intensity
dark_ratio = count(cell < dark_threshold) / cell_area

safety_shadow = 1 - min(dark_ratio * 3.0, 1.0)
```

**Interpretation:**
- Few dark pixels: Even lighting, flat terrain → Safe
- Many dark pixels: Shadows from obstacles or depressions → Unsafe

### 2.6 Combined Safety Score

Weighted average of all metrics:

```
S_cell = w₁·S_texture + w₂·S_edge + w₃·S_color + w₄·S_shadow

Default weights:
w₁ = 0.30  # Texture weight
w₂ = 0.35  # Edge weight (highest - most reliable)
w₃ = 0.20  # Color weight
w₄ = 0.15  # Shadow weight

Σwᵢ = 1.0
```

**Why these weights?**
- Edges are most reliable indicators of obstacles
- Texture variance correlates well with surface roughness
- Color and shadows provide supplementary information

---

## 3. Heatmap Generation

### 3.1 Gaussian Smoothing

Apply Gaussian blur to reduce noise and local minima:

```
G(x, y) = (1 / 2πσ²) · exp(-(x² + y²) / 2σ²)

S_smooth = S_raw ⊗ G
```

**Kernel size:** 5×5 (default)
**Sigma:** Auto-calculated by OpenCV

**Purpose:**
- Eliminate isolated high-score cells (unreliable)
- Create smooth transitions between zones
- Reduce sensitivity to local variations

### 3.2 Morphological Opening

Remove small safe spots that are too isolated:

```
kernel = ones(3×3)
S_opened = opening(S_smooth, kernel)
         = dilate(erode(S_smooth, kernel), kernel)
```

**Purpose:**
- Eliminate noise specks
- Require minimum safe zone size for landing
- Conservative approach to safety

### 3.3 Landing Zone Selection

**Step 1:** Find maximum safety score
```
S_max = max(S_opened)
candidates = where(S_opened == S_max)
```

**Step 2:** Select center-most candidate (if multiple)
```
center = [height/2, width/2]

for each candidate (i, j):
    distance[i, j] = √((i - center[0])² + (j - center[1])²)

best = argmin(distance)
```

**Why prefer center?**
- Reduces required maneuver angle
- More stable flight path
- Better sensor coverage

**Step 3:** Convert to pixel coordinates
```
grid_to_pixel_scale_x = frame_width / grid_width
grid_to_pixel_scale_y = frame_height / grid_height

pixel_x = (grid_x + 0.5) * scale_x
pixel_y = (grid_y + 0.5) * scale_y
```

Note: +0.5 targets the center of the grid cell

**Step 4:** Normalize coordinates
```
x_normalized = pixel_x / frame_width   # ∈ [0, 1]
y_normalized = pixel_y / frame_height  # ∈ [0, 1]
```

---

## 4. Coordinate Transformation

### 4.1 Camera Model

Assuming pinhole camera model:

```
                  y_pixel
                     ↑
                     |
    (-1, 1) ←────────┼────────→ (1, 1)
                     |
         ←───────────┼───────────→ x_pixel
                     |
    (-1, -1) ←───────┼────────→ (1, -1)
                     |
```

**Normalized Device Coordinates (NDC):**
```
x_ndc = 2 * (x_normalized - 0.5)  # ∈ [-1, 1]
y_ndc = 2 * (y_normalized - 0.5)  # ∈ [-1, 1]
```

### 4.2 Angular Offset Calculation

Convert from NDC to angular offset from camera center:

```
FOV_horizontal = 62.2°  # Raspberry Pi Camera V2
FOV_vertical = 48.8°    # Raspberry Pi Camera V2

angle_x = x_ndc * (FOV_h / 2)  # degrees
angle_y = y_ndc * (FOV_v / 2)  # degrees
```

**Example:**
- Target at (0.75, 0.5) → x_ndc = 0.5, y_ndc = 0.0
- angle_x = 0.5 × 31.1° = 15.55° right
- angle_y = 0.0° (centered vertically)

### 4.3 Distance Estimation

Given drone altitude h and vertical angle θ_y:

```
         drone (altitude h)
           |
           |h
           |
           |θ_y
           |___
          /    \
         /      \
        /        \
    ground    target

distance = h / cos(θ_y)
```

**Derivation:**
```
cos(θ_y) = h / distance
distance = h / cos(θ_y)
```

**Edge cases:**
- If |θ_y| > 45°: Target too far off-center, increase altitude
- If h < 1m: Switch to high-precision mode (smaller FOV)

### 4.4 MAVLink Message Format

```python
LANDING_TARGET {
    time_boot_ms: 0,              # Not used
    target_num: 0,                # Single target
    frame: MAV_FRAME_BODY_NED,    # Body-fixed frame
    angle_x: radians(angle_x),    # Radians
    angle_y: radians(angle_y),    # Radians
    distance: distance,           # Meters
    size_x: 0.0,                  # Not used
    size_y: 0.0                   # Not used
}
```

**Frame Convention:**
- X: Forward (nose direction)
- Y: Right
- Z: Down
- Origin: Center of vehicle

---

## 5. Mathematical Formulations

### 5.1 Complete Pipeline Equation

Given input frame I, the landing coordinate (x, y) is:

```
T(I) = argmax_{(x,y)} [ G(M(R(C(I)))) ]

where:
C(I) = Classification function
R(I) = Roughness analysis (if C(I) = GRASS)
M(R) = Morphological operations
G(M) = Gaussian smoothing
argmax = Landing zone selection
```

### 5.2 Safety Score Aggregation

For grid cell (i, j):

```
S(i,j) = Σ_{k=1}^{4} w_k · f_k(I_cell)

f₁ = 1 - min(σ_texture / 60, 1)       # Texture
f₂ = 1 - min(ρ_edges · 5, 1)          # Edges
f₃ = 1 - min(σ_color / 50, 1)         # Color
f₄ = 1 - min(ρ_dark · 3, 1)           # Shadows

Σw_k = 1, w_k > 0
```

### 5.3 Confidence Estimation

Classification confidence:

```
P(GRASS) = (ρ_grass + σ_texture/1000) / 2
P(WATER) = (ρ_water + ρ_reflections/2) / 2

confidence = max(P(GRASS), P(WATER))
```

Landing zone confidence:

```
confidence = S_max · (1 - d_center/d_max)

where:
S_max = Maximum safety score
d_center = Distance from frame center
d_max = Maximum possible distance (corner)
```

### 5.4 Performance Metrics

**Precision:** Fraction of safe predictions that are actually safe
```
Precision = TP / (TP + FP)
```

**Recall:** Fraction of actually safe zones that are detected
```
Recall = TP / (TP + FN)
```

**F1 Score:** Harmonic mean of precision and recall
```
F1 = 2 · (Precision · Recall) / (Precision + Recall)
```

**Processing Time Budget (60 FPS = 16.7ms):**
```
T_total = T_capture + T_classify + T_analyze + T_heatmap + T_mavlink

Target:
T_classify < 30ms
T_analyze < 50ms
T_heatmap < 20ms
T_mavlink < 5ms
----------------
T_total < 105ms → ~9.5 FPS
```

---

## 6. Error Analysis & Mitigation

### 6.1 Common Failure Modes

1. **False Water Detection**
   - Cause: Very wet grass, shadow-heavy grass
   - Mitigation: Increase texture variance threshold
   
2. **Missed Obstacles**
   - Cause: Low contrast, similar color to grass
   - Mitigation: Lower edge detection thresholds
   
3. **Center Bias**
   - Cause: Always selecting center regardless of safety
   - Mitigation: Add minimum safety threshold

### 6.2 Robustness Improvements

**Multi-scale Analysis:**
```python
scales = [32, 16, 64]
heatmaps = [analyze(frame, scale) for scale in scales]
final_heatmap = weighted_average(heatmaps)
```

**Temporal Filtering:**
```python
# Moving average over last N frames
heatmap_filtered = α · heatmap_current + (1-α) · heatmap_previous
```

**Confidence Gating:**
```python
if landing_zone.safety_score < 0.6:
    # Request hover or altitude increase
    send_hover_command()
```

---

## References

1. OpenCV Documentation: https://docs.opencv.org/
2. MAVLink Protocol: https://mavlink.io/en/
3. Computer Vision Metrics: Forsyth & Ponce, "Computer Vision: A Modern Approach"
4. Texture Analysis: Haralick et al., "Textural Features for Image Classification"
