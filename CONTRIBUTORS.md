# Contributors

## Dhairya Vijay
*GitHub: [@DhairyaVijay](https://github.com/DhairyaVijay)*

**Contributions:**
Lead developer for system architecture, spatial safety analysis, and flight controller communication.

**Key Work:**
* **Grid-Based Safety Analysis:** Built the `RoughnessAnalyzer` using edge detection and filtering across a 32x32 grid to score localized terrain safety.
* **Heatmap & Target Selection:** Developed the `HeatmapGenerator` using Gaussian smoothing to pinpoint the safest physical touchdown coordinates.
* **Flight Controller Integration:** Interfaced with the SpeedyBee F4 V3 via MAVLink, translating 2D camera pixels into 3D angular flight commands.
* **Stream Management:** Engineered the ESP32-CAM video pipeline, implementing strict buffer controls to eliminate 2-3 seconds of stream latency.
---

## Ritu Choudhary
*GitHub: [@rituchoudhary-1907](https://github.com/rituchoudhary-1907)*

**Contributions:Terrain Classification Pipeline:**

**Key Work:**
* **Color Segmentation:** Tuned HSV thresholds for grass/green, water/blue, concrete/gray detection under varying light
* **Texture Analysis:** Used Laplacian variance to differentiate textured grass vs smooth water/concrete
* **Reflection Logic:** Detected water via specular highlights using Value-channel clustering
* **Decision Engine:** Built threshold-based classifier combining 3 metrics without ML overhead
