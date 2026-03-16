# Performance Tuning Guide

## Overview

This guide helps you optimize the landing vision system for different Raspberry Pi models and use cases.

## Hardware-Specific Configurations

### Raspberry Pi 4 (4GB/8GB) - Recommended

**Default Configuration (Balanced):**
```ini
[camera]
width = 640
height = 480
fps = 30

[roughness]
grid_size = 32

[performance]
target_fps = 10
```

**Expected Performance:** ~10 FPS, 100ms latency

**High-Speed Configuration (Sacrifice accuracy for speed):**
```ini
[camera]
width = 320
height = 240
fps = 30

[roughness]
grid_size = 16

[heatmap]
smoothing_kernel = 3

[performance]
target_fps = 20
```

**Expected Performance:** ~20 FPS, 50ms latency

**High-Accuracy Configuration (Sacrifice speed for accuracy):**
```ini
[camera]
width = 1280
height = 720
fps = 15

[classification]
use_tflite = true

[roughness]
grid_size = 64

[performance]
target_fps = 5
```

**Expected Performance:** ~5 FPS, 200ms latency

### Raspberry Pi 3B+ (1GB)

**Recommended Configuration:**
```ini
[camera]
width = 320
height = 240
fps = 15

[roughness]
grid_size = 16
smoothing_kernel = 3

[heatmap]
smoothing_kernel = 3

[performance]
target_fps = 5
```

**Expected Performance:** ~5 FPS, 180ms latency

**Tips:**
- Disable visualization: Run headless
- Reduce grid size to 16×16
- Use 320×240 resolution
- Disable TFLite (too slow)

### Raspberry Pi Zero 2W (512MB)

**Minimal Configuration:**
```ini
[camera]
width = 320
height = 240
fps = 10

[roughness]
grid_size = 8

[heatmap]
smoothing_kernel = 3

[performance]
target_fps = 2
```

**Expected Performance:** ~2 FPS, 500ms latency

**Warning:** Not recommended for real-time landing. Consider upgrading to Pi 4.

## Optimization Techniques

### 1. Resolution Trade-offs

| Resolution | Processing Time | Field of View | Accuracy |
|------------|----------------|---------------|----------|
| 320×240 | 25ms | Full | Low |
| 640×480 | 100ms | Full | Medium |
| 1280×720 | 300ms | Full | High |
| 1920×1080 | 600ms | Full | Very High |

**Recommendation:** 640×480 is the sweet spot for most applications.

### 2. Grid Size Impact

| Grid Size | Memory | Processing Time | Spatial Resolution |
|-----------|--------|-----------------|-------------------|
| 8×8 | 256 B | 10ms | 80×60 px/cell @640×480 |
| 16×16 | 1 KB | 25ms | 40×30 px/cell |
| 32×32 | 4 KB | 50ms | 20×15 px/cell (recommended) |
| 64×64 | 16 KB | 150ms | 10×7.5 px/cell |

**Formula:**
```
cell_size = frame_size / grid_size
processing_time ≈ grid_size² × 1.5ms
```

### 3. Algorithm Optimizations

#### Disable Unused Features

```python
# In landing_vision_system.py
class RoughnessAnalyzer:
    def analyze(self, frame):
        # Comment out expensive operations
        # shadow_score = self._detect_shadows(cell_gray)  # Save 15%
        
        combined_score = (
            texture_score * 0.35 +  # Increased weight
            edge_score * 0.40 +     # Increased weight
            color_score * 0.25      # Increased weight
            # shadow_score * 0.0   # Disabled
        )
```

#### Reduce Smoothing

```python
# Less smoothing = faster but noisier
self.smoothing_kernel = 3  # Instead of 5
```

#### Skip Morphological Operations

```python
# In HeatmapGenerator.generate()
# opened_map = cv2.morphologyEx(smoothed_map, cv2.MORPH_OPEN, kernel)
opened_map = smoothed_map  # Skip morphology
```

### 4. System-Level Optimizations

#### Overclock Raspberry Pi 4

Add to `/boot/config.txt`:
```
# CPU overclock
arm_freq=2000          # From 1500 MHz
over_voltage=6         # Required for 2000 MHz

# GPU overclock  
gpu_freq=600           # From 500 MHz
```

**Warning:** Requires active cooling (heatsink + fan).

#### Disable Desktop Environment

```bash
# Use Raspberry Pi OS Lite
sudo systemctl set-default multi-user.target

# Or disable desktop temporarily
sudo systemctl stop lightdm
```

**Performance Gain:** 15-20% CPU freed up

#### Process Priority

Run with high priority:
```bash
sudo nice -n -10 python3 landing_vision_system.py
```

#### CPU Governor

Set performance mode:
```bash
echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### 5. Memory Optimization

#### Reduce Camera Buffer

```python
# In LandingVisionPipeline.__init__()
self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer
```

#### Use In-Place Operations

```python
# Bad (creates copy)
smoothed = cv2.GaussianBlur(img, (5, 5), 0)

# Good (in-place when possible)
cv2.GaussianBlur(img, (5, 5), 0, dst=img)
```

#### Pre-allocate Arrays

```python
# Pre-allocate heatmap array
self.heatmap_buffer = np.zeros((32, 32), dtype=np.float32)

# Reuse in generate()
np.copyto(self.heatmap_buffer, new_heatmap)
```

## Benchmarking

### Built-in Performance Metrics

The system tracks processing time automatically:

```python
pipeline.run(duration=60)  # Run for 60 seconds

# On exit, prints:
# Average processing time: 105.3ms (9.5 FPS)
```

### Manual Benchmarking

```python
import time

classifier = TerrainClassifier()
frame = cv2.imread('test.jpg')

start = time.time()
for _ in range(100):
    _ = classifier.classify(frame)
elapsed = (time.time() - start) / 100

print(f"Classification time: {elapsed*1000:.1f}ms")
```

### Profiling with cProfile

```bash
python3 -m cProfile -o profile.stats landing_vision_system.py --duration 10

# Analyze results
python3 -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"
```

## Real-Time Requirements

### Latency Budget for 10 FPS (100ms)

```
┌─────────────────────┬──────────┐
│ Component           │ Time (ms)│
├─────────────────────┼──────────┤
│ Camera Capture      │ 10       │
│ Classification      │ 30       │
│ Roughness Analysis  │ 40       │
│ Heatmap Generation  │ 15       │
│ MAVLink Transmission│ 5        │
├─────────────────────┼──────────┤
│ Total               │ 100      │
└─────────────────────┴──────────┘
```

### Acceptable Latency by Altitude

| Altitude (m) | Max Latency | Target FPS |
|--------------|-------------|------------|
| > 10m | 500ms | 2+ |
| 5-10m | 200ms | 5+ |
| 2-5m | 100ms | 10+ |
| < 2m | 50ms | 20+ |

**Recommendation:** Reduce altitude if processing cannot keep up.

## Troubleshooting Performance Issues

### Symptom: Low FPS (< 5 FPS)

**Check:**
1. CPU temperature: `vcgencmd measure_temp`
2. CPU throttling: `vcgencmd get_throttled`
3. Memory usage: `free -h`
4. Background processes: `htop`

**Solutions:**
1. Add cooling (heatsink + fan)
2. Kill unnecessary processes
3. Reduce resolution
4. Disable visualization

### Symptom: Variable Frame Rate

**Causes:**
- Thermal throttling
- Swapping (low RAM)
- Other processes competing for CPU

**Solutions:**
```bash
# Disable swap
sudo swapoff -a

# Check for other processes
ps aux | grep python

# Monitor CPU frequency
watch -n 1 cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
```

### Symptom: High Latency Spikes

**Causes:**
- Garbage collection
- I/O operations
- Network activity

**Solutions:**
```python
# Disable garbage collection during critical sections
import gc
gc.disable()
result = process_frame()
gc.enable()
```

## Configuration Presets

### Preset 1: Speed (Real-time landing)
```bash
python3 landing_vision_system.py \
    --width 320 --height 240 \
    --mavlink
```

### Preset 2: Balanced (Recommended)
```bash
python3 landing_vision_system.py \
    --width 640 --height 480 \
    --mavlink
```

### Preset 3: Accuracy (Survey missions)
```bash
python3 landing_vision_system.py \
    --width 1280 --height 720 \
    --tflite --mavlink
```

### Preset 4: Debug (Development)
```bash
python3 landing_vision_system.py \
    --width 640 --height 480 \
    --duration 30
```

## Expected Performance by Configuration

```
┌─────────────┬────────────┬─────────┬──────────┬──────────┐
│ Config      │ Resolution │ Grid    │ FPS (Pi4)│ FPS (Pi3)│
├─────────────┼────────────┼─────────┼──────────┼──────────┤
│ Speed       │ 320×240    │ 16×16   │ 20       │ 8        │
│ Balanced    │ 640×480    │ 32×32   │ 10       │ 5        │
│ Accuracy    │ 1280×720   │ 64×64   │ 5        │ 2        │
│ High-Acc    │ 1920×1080  │ 64×64   │ 2        │ 0.5      │
└─────────────┴────────────┴─────────┴──────────┴──────────┘
```

## Power Consumption

| Configuration | Power Draw | Battery Life (10Ah) |
|---------------|------------|---------------------|
| Idle | 2.5W | ~20 hours |
| Speed | 5W | ~10 hours |
| Balanced | 7W | ~7 hours |
| Accuracy | 9W | ~5.5 hours |

**Note:** Power draw includes Pi + Camera, not including motors/FC.

## Summary

**For most users:** Use Balanced configuration (640×480, 32×32 grid)
**For speed:** Use Speed configuration (320×240, 16×16 grid)
**For accuracy:** Use Accuracy configuration (1280×720, 64×64 grid, TFLite)

Always test your configuration in safe conditions before deploying in production.
