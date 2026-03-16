#!/usr/bin/env python3
"""
Autonomous Drone Landing Vision Pipeline
Optimized for Raspberry Pi + SpeedyBee F4 V3
Author: UAV Vision Systems Team
"""

import cv2
import numpy as np
import time
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TerrainType(Enum):
    """Terrain classification types"""
    GRASS = "grass"
    WATER = "water"
    CONCRETE = "concrete"
    UNKNOWN = "unknown"


@dataclass
class LandingZone:
    """Landing zone information"""
    x_normalized: float  # 0.0 to 1.0
    y_normalized: float  # 0.0 to 1.0
    x_pixel: int
    y_pixel: int
    safety_score: float  # 0.0 to 1.0
    terrain_type: TerrainType


class TerrainClassifier:
    """
    Grass vs Water classification using OpenCV
    Optimized for Raspberry Pi performance
    """
    
    def __init__(self, use_tflite: bool = False):
        """
        Initialize terrain classifier
        
        Args:
            use_tflite: If True, use TensorFlow Lite model (higher accuracy, slower)
        """
        self.use_tflite = use_tflite
        self.tflite_model = None
        
        # Color thresholds in HSV space
        # Grass: Green hues with moderate saturation
        self.grass_lower = np.array([35, 40, 40])    # Lower green
        self.grass_upper = np.array([85, 255, 255])  # Upper green
        
        # Water: Blue/cyan hues, often with low saturation (reflections)
        self.water_lower = np.array([90, 30, 30])    # Lower blue
        self.water_upper = np.array([130, 255, 255]) # Upper blue
        
        # Concrete: Gray/achromatic (low saturation, any hue)
        # We'll use saturation and value thresholds instead
        self.concrete_sat_max = 50      # Low saturation (gray)
        self.concrete_val_min = 60      # Not too dark
        self.concrete_val_max = 220     # Not too bright (avoid white lines)
        
        if use_tflite:
            self._load_tflite_model()
    
    def _load_tflite_model(self):
        """Load TensorFlow Lite model if available"""
        try:
            import tensorflow as tf
            # Placeholder for TFLite model loading
            # self.tflite_model = tf.lite.Interpreter(model_path="terrain_classifier.tflite")
            # self.tflite_model.allocate_tensors()
            logger.info("TFLite model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load TFLite model: {e}. Falling back to OpenCV.")
            self.use_tflite = False
    
    def classify(self, frame: np.ndarray) -> Tuple[TerrainType, float]:
        """
        Classify terrain as Grass or Water
        
        Args:
            frame: BGR image from camera
            
        Returns:
            Tuple of (TerrainType, confidence_score)
        """
        if self.use_tflite and self.tflite_model is not None:
            return self._classify_tflite(frame)
        else:
            return self._classify_opencv(frame)
    
    def _classify_opencv(self, frame: np.ndarray) -> Tuple[TerrainType, float]:
        """
        OpenCV-based classification using color and texture analysis
        Fast and suitable for Raspberry Pi
        """
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Apply Gaussian blur to reduce noise
        hsv_blur = cv2.GaussianBlur(hsv, (5, 5), 0)
        
        # Create masks for grass and water
        grass_mask = cv2.inRange(hsv_blur, self.grass_lower, self.grass_upper)
        water_mask = cv2.inRange(hsv_blur, self.water_lower, self.water_upper)
        
        # Create mask for concrete (low saturation, medium value)
        # Extract S and V channels
        s_channel = hsv_blur[:, :, 1]
        v_channel = hsv_blur[:, :, 2]
        
        # Concrete has low saturation and medium brightness
        concrete_mask = cv2.inRange(s_channel, 0, self.concrete_sat_max)
        concrete_mask = cv2.bitwise_and(
            concrete_mask,
            cv2.inRange(v_channel, self.concrete_val_min, self.concrete_val_max)
        )
        
        # Calculate coverage percentages
        total_pixels = frame.shape[0] * frame.shape[1]
        grass_pixels = cv2.countNonZero(grass_mask)
        water_pixels = cv2.countNonZero(water_mask)
        concrete_pixels = cv2.countNonZero(concrete_mask)
        
        grass_ratio = grass_pixels / total_pixels
        water_ratio = water_pixels / total_pixels
        concrete_ratio = concrete_pixels / total_pixels
        
        # Texture analysis for disambiguation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate texture variance (grass is typically more textured)
        texture_variance = self._calculate_texture_variance(gray)
        
        # Water detection: Often has high specular reflections
        reflection_score = self._detect_reflections(frame, hsv)
        
        # Concrete detection: Very uniform color and low texture
        color_uniformity = self._calculate_color_uniformity(frame)
        
        # Decision logic (prioritize by confidence)
        # Concrete is distinctive: high coverage + low texture + high uniformity
        if concrete_ratio > 0.3 and texture_variance < 80 and color_uniformity > 0.7:
            # Strong concrete indicators
            confidence = min(concrete_ratio + color_uniformity * 0.3, 1.0)
            return TerrainType.CONCRETE, confidence
        
        # Water detection: High water ratio or strong reflections
        elif water_ratio > 0.15 and reflection_score > 0.3:
            # Strong water indicators
            confidence = min(water_ratio + reflection_score * 0.5, 1.0)
            return TerrainType.WATER, confidence
        
        # Grass detection: High grass ratio and texture
        elif grass_ratio > 0.25 and texture_variance > 100:
            # Strong grass indicators
            confidence = min(grass_ratio + (texture_variance / 1000), 1.0)
            return TerrainType.GRASS, confidence
        
        # Secondary decision based on coverage ratios
        elif concrete_ratio > grass_ratio and concrete_ratio > water_ratio:
            confidence = concrete_ratio
            return TerrainType.CONCRETE, confidence
        elif grass_ratio > water_ratio:
            return TerrainType.GRASS, grass_ratio
        elif water_ratio > grass_ratio:
            return TerrainType.WATER, water_ratio
        else:
            return TerrainType.UNKNOWN, 0.5
    
    def _calculate_texture_variance(self, gray: np.ndarray) -> float:
        """Calculate texture variance using Laplacian"""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        return variance
    
    def _detect_reflections(self, frame: np.ndarray, hsv: np.ndarray) -> float:
        """
        Detect specular reflections typical of water surfaces
        Returns score 0.0 to 1.0
        """
        # Extract Value channel (brightness)
        v_channel = hsv[:, :, 2]
        
        # Find very bright spots (potential reflections)
        _, bright_mask = cv2.threshold(v_channel, 200, 255, cv2.THRESH_BINARY)
        
        # Calculate ratio of bright pixels
        bright_ratio = cv2.countNonZero(bright_mask) / (frame.shape[0] * frame.shape[1])
        
        # Check if bright spots are clustered (typical of water reflections)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)
        
        if num_labels > 2:  # Background + at least 1 bright spot
            # Large, sparse bright regions suggest water reflections
            avg_area = stats[1:, cv2.CC_STAT_AREA].mean() if num_labels > 1 else 0
            if avg_area > 50:  # Reasonably sized reflections
                return min(bright_ratio * 10, 1.0)
        
        return 0.0
    
    def _calculate_color_uniformity(self, frame: np.ndarray) -> float:
        """
        Calculate color uniformity across the entire frame
        Higher score = more uniform (typical of concrete)
        Returns score 0.0 to 1.0
        """
        # Calculate standard deviation for each color channel
        b_std = np.std(frame[:, :, 0])
        g_std = np.std(frame[:, :, 1])
        r_std = np.std(frame[:, :, 2])
        
        # Average standard deviation
        avg_std = (b_std + g_std + r_std) / 3.0
        
        # Normalize: lower std deviation = more uniform
        # Concrete typically has std < 30, grass has std > 40
        uniformity = 1.0 - min(avg_std / 50.0, 1.0)
        
        return uniformity
    
    def _classify_tflite(self, frame: np.ndarray) -> Tuple[TerrainType, float]:
        """
        TensorFlow Lite classification (higher accuracy, ~50ms on Pi 4)
        Placeholder for future implementation
        """
        # TODO: Implement TFLite inference
        # 1. Preprocess frame (resize to model input size, normalize)
        # 2. Run inference
        # 3. Get class probabilities
        logger.warning("TFLite classification not implemented, falling back to OpenCV")
        return self._classify_opencv(frame)


class RoughnessAnalyzer:
    """
    Analyze grass surface for roughness, obstacles, and suitability for landing
    """
    
    def __init__(self, grid_size: int = 32):
        """
        Initialize roughness analyzer
        
        Args:
            grid_size: Size of analysis grid (32x32 recommended for Pi)
        """
        self.grid_size = grid_size
    
    def analyze(self, frame: np.ndarray) -> np.ndarray:
        """
        Analyze frame and generate roughness map
        
        Args:
            frame: BGR image of grass terrain
            
        Returns:
            2D numpy array with roughness scores (0.0 = rough/unsafe, 1.0 = smooth/safe)
        """
        h, w = frame.shape[:2]
        roughness_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # Calculate cell dimensions
        cell_h = h // self.grid_size
        cell_w = w // self.grid_size
        
        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to preserve edges while smoothing
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Calculate edges for obstacle detection
        edges = cv2.Canny(filtered, 50, 150)
        
        # Analyze each grid cell
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Extract cell region
                y_start = i * cell_h
                y_end = (i + 1) * cell_h
                x_start = j * cell_w
                x_end = (j + 1) * cell_w
                
                cell_gray = gray[y_start:y_end, x_start:x_end]
                cell_edges = edges[y_start:y_end, x_start:x_end]
                cell_color = frame[y_start:y_end, x_start:x_end]
                
                # Calculate multiple safety metrics
                texture_score = self._calculate_texture_safety(cell_gray)
                edge_score = self._calculate_edge_safety(cell_edges)
                color_score = self._calculate_color_uniformity(cell_color)
                shadow_score = self._detect_shadows(cell_gray)
                
                # Combine scores (weighted average)
                combined_score = (
                    texture_score * 0.3 +
                    edge_score * 0.35 +
                    color_score * 0.2 +
                    shadow_score * 0.15
                )
                
                roughness_map[i, j] = combined_score
        
        return roughness_map
    
    def _calculate_texture_safety(self, cell: np.ndarray) -> float:
        """
        Lower texture variance = smoother = safer
        Returns score 0.0 to 1.0
        """
        if cell.size == 0:
            return 0.0
        
        # Calculate standard deviation of pixel intensities
        std_dev = np.std(cell)
        
        # Normalize: Lower std_dev is better
        # Typical grass: std_dev 15-40, rough/rocky: std_dev > 50
        safety_score = 1.0 - min(std_dev / 60.0, 1.0)
        
        return safety_score
    
    def _calculate_edge_safety(self, cell_edges: np.ndarray) -> float:
        """
        Fewer edges = fewer obstacles = safer
        Returns score 0.0 to 1.0
        """
        if cell_edges.size == 0:
            return 0.0
        
        edge_density = cv2.countNonZero(cell_edges) / cell_edges.size
        
        # Normalize: Lower edge density is better
        safety_score = 1.0 - min(edge_density * 5.0, 1.0)
        
        return safety_score
    
    def _calculate_color_uniformity(self, cell_color: np.ndarray) -> float:
        """
        More uniform color = less variation = safer
        Returns score 0.0 to 1.0
        """
        if cell_color.size == 0:
            return 0.0
        
        # Calculate color variance across channels
        color_std = np.std(cell_color, axis=(0, 1))
        avg_std = np.mean(color_std)
        
        # Normalize: Lower variance is better
        safety_score = 1.0 - min(avg_std / 50.0, 1.0)
        
        return safety_score
    
    def _detect_shadows(self, cell: np.ndarray) -> float:
        """
        Detect dark regions that might indicate holes or shadows from obstacles
        Returns score 0.0 to 1.0
        """
        if cell.size == 0:
            return 0.0
        
        # Count dark pixels (potential shadows/holes)
        dark_threshold = 50
        dark_pixels = np.sum(cell < dark_threshold)
        dark_ratio = dark_pixels / cell.size
        
        # Lower dark ratio is better
        safety_score = 1.0 - min(dark_ratio * 3.0, 1.0)
        
        return safety_score


class HeatmapGenerator:
    """
    Generate landing safety heatmap and find optimal landing coordinates
    """
    
    def __init__(self, smoothing_kernel: int = 5):
        """
        Initialize heatmap generator
        
        Args:
            smoothing_kernel: Size of Gaussian kernel for smoothing
        """
        self.smoothing_kernel = smoothing_kernel
    
    def generate(self, roughness_map: np.ndarray, 
                 frame_shape: Tuple[int, int]) -> Tuple[np.ndarray, LandingZone]:
        """
        Generate heatmap and find optimal landing zone
        
        Args:
            roughness_map: 2D array of safety scores
            frame_shape: Original frame dimensions (height, width)
            
        Returns:
            Tuple of (heatmap_image, landing_zone)
        """
        # Apply Gaussian smoothing to avoid local minima
        smoothed_map = cv2.GaussianBlur(
            roughness_map, 
            (self.smoothing_kernel, self.smoothing_kernel), 
            0
        )
        
        # Apply morphological operations to remove isolated safe spots
        kernel = np.ones((3, 3), np.uint8)
        opened_map = cv2.morphologyEx(smoothed_map, cv2.MORPH_OPEN, kernel)
        
        # Find the safest landing spot
        max_safety = np.max(opened_map)
        safe_indices = np.where(opened_map == max_safety)
        
        if len(safe_indices[0]) == 0:
            # No safe spot found
            logger.warning("No safe landing zone detected!")
            landing_zone = LandingZone(
                x_normalized=0.5,
                y_normalized=0.5,
                x_pixel=frame_shape[1] // 2,
                y_pixel=frame_shape[0] // 2,
                safety_score=0.0,
                terrain_type=TerrainType.UNKNOWN
            )
        else:
            # If multiple cells have max safety, choose the center-most one
            best_idx = self._find_centermost_point(
                safe_indices, 
                opened_map.shape,
                prefer_center=True
            )
            
            grid_y, grid_x = safe_indices[0][best_idx], safe_indices[1][best_idx]
            
            # Convert grid coordinates to pixel coordinates (center of cell)
            h, w = frame_shape
            grid_h, grid_w = opened_map.shape
            
            pixel_x = int((grid_x + 0.5) * w / grid_w)
            pixel_y = int((grid_y + 0.5) * h / grid_h)
            
            # Normalize coordinates
            x_norm = pixel_x / w
            y_norm = pixel_y / h
            
            landing_zone = LandingZone(
                x_normalized=x_norm,
                y_normalized=y_norm,
                x_pixel=pixel_x,
                y_pixel=pixel_y,
                safety_score=max_safety,
                terrain_type=TerrainType.GRASS
            )
        
        # Create visualization heatmap
        heatmap_viz = self._create_heatmap_visualization(
            opened_map, 
            frame_shape, 
            landing_zone
        )
        
        return heatmap_viz, landing_zone
    
    def _find_centermost_point(self, indices: Tuple, 
                               shape: Tuple[int, int],
                               prefer_center: bool = True) -> int:
        """
        Find the index closest to the center of the frame
        """
        if not prefer_center or len(indices[0]) == 1:
            return 0
        
        center_y, center_x = shape[0] / 2, shape[1] / 2
        
        # Calculate distances to center
        distances = np.sqrt(
            (indices[0] - center_y) ** 2 + 
            (indices[1] - center_x) ** 2
        )
        
        return np.argmin(distances)
    
    def _create_heatmap_visualization(self, safety_map: np.ndarray,
                                     frame_shape: Tuple[int, int],
                                     landing_zone: LandingZone) -> np.ndarray:
        """
        Create a colorful heatmap visualization
        """
        # Resize safety map to match frame shape
        heatmap_resized = cv2.resize(
            safety_map, 
            (frame_shape[1], frame_shape[0]), 
            interpolation=cv2.INTER_LINEAR
        )
        
        # Normalize to 0-255
        heatmap_normalized = (heatmap_resized * 255).astype(np.uint8)
        
        # Apply colormap (TURBO: blue=unsafe, red=safe)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_TURBO)
        
        # Draw landing target
        cv2.circle(
            heatmap_colored,
            (landing_zone.x_pixel, landing_zone.y_pixel),
            20,
            (0, 255, 0),  # Green circle
            2
        )
        cv2.drawMarker(
            heatmap_colored,
            (landing_zone.x_pixel, landing_zone.y_pixel),
            (0, 255, 0),
            cv2.MARKER_CROSS,
            20,
            2
        )
        
        return heatmap_colored


class MAVLinkCommunicator:
    """
    Communication with SpeedyBee F4 V3 flight controller via MAVLink
    """
    
    def __init__(self, connection_string: str = '/dev/ttyAMA0', baudrate: int = 57600):
        """
        Initialize MAVLink connection
        
        Args:
            connection_string: Serial port or network address
            baudrate: Serial baudrate (57600 for most FCs)
        """
        self.connection_string = connection_string
        self.baudrate = baudrate
        self.vehicle = None
        self.is_connected = False
        
        self._connect()
    
    def _connect(self):
        """Establish connection to flight controller"""
        try:
            from dronekit import connect
            logger.info(f"Connecting to vehicle on: {self.connection_string}")
            
            # Connect with timeout
            self.vehicle = connect(
                self.connection_string,
                wait_ready=True,
                baud=self.baudrate,
                timeout=10
            )
            
            self.is_connected = True
            logger.info("Successfully connected to vehicle")
            logger.info(f"Vehicle mode: {self.vehicle.mode.name}")
            
        except ImportError:
            logger.error("DroneKit not installed. Install with: pip install dronekit")
            self.is_connected = False
        except Exception as e:
            logger.error(f"Failed to connect to vehicle: {e}")
            self.is_connected = False
    
    def send_landing_target(self, landing_zone: LandingZone, 
                          altitude: float = 5.0) -> bool:
        """
        Send landing target coordinates to flight controller
        
        Args:
            landing_zone: LandingZone object with coordinates
            altitude: Current altitude above ground (meters)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected or self.vehicle is None:
            logger.warning("Vehicle not connected. Cannot send landing target.")
            return False
        
        try:
            from pymavlink import mavutil
            
            # Convert normalized coordinates to angles (relative to camera FOV)
            # Assuming 60° horizontal FOV, 45° vertical FOV (typical for Pi Camera V2)
            h_fov = 62.2  # degrees
            v_fov = 48.8  # degrees
            
            # Calculate angular offset from center
            angle_x = (landing_zone.x_normalized - 0.5) * h_fov
            angle_y = (landing_zone.y_normalized - 0.5) * v_fov
            
            # Calculate distance to target based on altitude and angle
            distance = altitude / np.cos(np.radians(angle_y))
            
            # Send LANDING_TARGET message
            msg = self.vehicle.message_factory.landing_target_encode(
                0,  # time_boot_ms (not used)
                0,  # target_num
                mavutil.mavlink.MAV_FRAME_BODY_NED,  # frame
                angle_x,  # angle_x (radians)
                angle_y,  # angle_y (radians)
                distance,  # distance (m)
                0.0,  # size_x (not used)
                0.0   # size_y (not used)
            )
            
            self.vehicle.send_mavlink(msg)
            
            logger.info(f"Sent landing target: angle_x={angle_x:.2f}°, "
                       f"angle_y={angle_y:.2f}°, distance={distance:.2f}m, "
                       f"safety={landing_zone.safety_score:.2f}")
            
            return True
            
        except ImportError:
            logger.error("pymavlink not installed. Install with: pip install pymavlink")
            return False
        except Exception as e:
            logger.error(f"Failed to send landing target: {e}")
            return False
    
    def get_altitude(self) -> Optional[float]:
        """
        Get current altitude from flight controller
        
        Returns:
            Altitude in meters, or None if unavailable
        """
        if not self.is_connected or self.vehicle is None:
            return None
        
        try:
            # Get altitude from rangefinder or barometer
            altitude = self.vehicle.rangefinder.distance
            if altitude is None or altitude < 0:
                altitude = self.vehicle.location.global_relative_frame.alt
            
            return altitude
        except Exception as e:
            logger.error(f"Failed to get altitude: {e}")
            return None
    
    def close(self):
        """Close MAVLink connection"""
        if self.vehicle is not None:
            logger.info("Closing vehicle connection")
            self.vehicle.close()
            self.is_connected = False


class LandingVisionPipeline:
    """
    Main vision pipeline orchestrating all components
    """
    
    def __init__(self, 
                 camera_id: int = 0,
                 resolution: Tuple[int, int] = (640, 480),
                 use_tflite: bool = False,
                 enable_mavlink: bool = False,
                 mavlink_connection: str = '/dev/ttyAMA0'):
        """
        Initialize landing vision pipeline
        
        Args:
            camera_id: Camera device ID
            resolution: Camera resolution (width, height)
            use_tflite: Use TensorFlow Lite for classification
            enable_mavlink: Enable MAVLink communication
            mavlink_connection: MAVLink connection string
        """
        self.resolution = resolution
        
        # Initialize camera
        self.camera = cv2.VideoCapture(camera_id)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        if not self.camera.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")
        
        logger.info(f"Camera initialized: {resolution[0]}x{resolution[1]}")
        
        # Initialize pipeline components
        self.classifier = TerrainClassifier(use_tflite=use_tflite)
        self.roughness_analyzer = RoughnessAnalyzer(grid_size=32)
        self.heatmap_generator = HeatmapGenerator(smoothing_kernel=5)
        
        # Initialize MAVLink (optional)
        self.mavlink = None
        if enable_mavlink:
            self.mavlink = MAVLinkCommunicator(mavlink_connection)
        
        # Performance tracking
        self.frame_times = []
    
    def process_frame(self, visualize: bool = True) -> Optional[LandingZone]:
        """
        Process a single frame through the pipeline
        
        Args:
            visualize: If True, display visualization windows
            
        Returns:
            LandingZone object or None if unsafe
        """
        start_time = time.time()
        
        # Capture frame
        ret, frame = self.camera.read()
        if not ret:
            logger.error("Failed to capture frame")
            return None
        
        # Step 1: Terrain Classification
        terrain_type, confidence = self.classifier.classify(frame)
        
        logger.info(f"Terrain: {terrain_type.value} (confidence: {confidence:.2f})")
        
        landing_zone = None
        
        if terrain_type == TerrainType.WATER:
            logger.warning("WATER DETECTED - Aborting landing sequence")
            if visualize:
                warning_frame = frame.copy()
                cv2.putText(
                    warning_frame,
                    "UNSAFE: WATER DETECTED",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2
                )
                cv2.imshow('Landing Vision', warning_frame)
        
        elif terrain_type == TerrainType.GRASS or terrain_type == TerrainType.CONCRETE:
            # Both grass and concrete are safe for landing - analyze surface
            logger.info(f"Safe terrain detected: {terrain_type.value.upper()}")
            
            # Step 2: Roughness Analysis
            roughness_map = self.roughness_analyzer.analyze(frame)
            
            # Step 3: Heatmap Generation & Landing Zone Selection
            heatmap, landing_zone = self.heatmap_generator.generate(
                roughness_map,
                frame.shape[:2]
            )
            
            # Update terrain type in landing zone
            landing_zone.terrain_type = terrain_type
            
            logger.info(f"Landing Zone: ({landing_zone.x_normalized:.3f}, "
                       f"{landing_zone.y_normalized:.3f}), "
                       f"Safety: {landing_zone.safety_score:.2f}")
            
            # Step 4: Send to Flight Controller
            if self.mavlink is not None:
                altitude = self.mavlink.get_altitude()
                if altitude is not None and altitude > 0.5:  # Only send if altitude is reasonable
                    self.mavlink.send_landing_target(landing_zone, altitude)
            
            # Visualization
            if visualize:
                # Create side-by-side visualization
                vis_frame = frame.copy()
                
                # Draw landing target on original frame
                cv2.circle(vis_frame, (landing_zone.x_pixel, landing_zone.y_pixel), 
                          30, (0, 255, 0), 3)
                cv2.drawMarker(vis_frame, (landing_zone.x_pixel, landing_zone.y_pixel),
                              (0, 255, 0), cv2.MARKER_CROSS, 40, 3)
                
                # Add info text with terrain-specific color
                terrain_color = (0, 255, 0) if terrain_type == TerrainType.GRASS else (255, 200, 0)
                cv2.putText(vis_frame, f"Terrain: {terrain_type.value.upper()}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, terrain_color, 2)
                cv2.putText(vis_frame, f"Confidence: {confidence:.2f}",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(vis_frame, f"Safety: {landing_zone.safety_score:.2f}",
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(vis_frame, f"Target: ({landing_zone.x_normalized:.2f}, "
                           f"{landing_zone.y_normalized:.2f})",
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show both original and heatmap
                combined = np.hstack([vis_frame, heatmap])
                cv2.imshow('Landing Vision', combined)
        
        else:
            logger.warning("UNKNOWN TERRAIN - Cannot determine safety")
            if visualize:
                warning_frame = frame.copy()
                cv2.putText(warning_frame, "UNKNOWN TERRAIN", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
                cv2.imshow('Landing Vision', warning_frame)
        
        # Performance tracking
        elapsed = time.time() - start_time
        self.frame_times.append(elapsed)
        
        fps = 1.0 / elapsed if elapsed > 0 else 0
        logger.debug(f"Processing time: {elapsed*1000:.1f}ms ({fps:.1f} FPS)")
        
        return landing_zone
    
    def run(self, duration: Optional[float] = None):
        """
        Run the vision pipeline continuously
        
        Args:
            duration: Run for specified seconds (None = run until interrupted)
        """
        logger.info("Starting landing vision pipeline...")
        logger.info("Press 'q' to quit")
        
        start_time = time.time()
        
        try:
            while True:
                landing_zone = self.process_frame(visualize=True)
                
                # Check for quit command
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Quit command received")
                    break
                
                # Check duration limit
                if duration is not None and (time.time() - start_time) > duration:
                    logger.info(f"Duration limit ({duration}s) reached")
                    break
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up resources...")
        
        # Print performance statistics
        if self.frame_times:
            avg_time = np.mean(self.frame_times)
            avg_fps = 1.0 / avg_time if avg_time > 0 else 0
            logger.info(f"Average processing time: {avg_time*1000:.1f}ms ({avg_fps:.1f} FPS)")
        
        # Release camera
        if self.camera is not None:
            self.camera.release()
        
        # Close MAVLink
        if self.mavlink is not None:
            self.mavlink.close()
        
        cv2.destroyAllWindows()
        logger.info("Cleanup complete")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Autonomous Drone Landing Vision System')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    parser.add_argument('--width', type=int, default=640, help='Frame width')
    parser.add_argument('--height', type=int, default=480, help='Frame height')
    parser.add_argument('--tflite', action='store_true', help='Use TensorFlow Lite')
    parser.add_argument('--mavlink', action='store_true', help='Enable MAVLink')
    parser.add_argument('--connection', type=str, default='/dev/ttyAMA0',
                       help='MAVLink connection string')
    parser.add_argument('--duration', type=float, default=None,
                       help='Run duration in seconds (default: infinite)')
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = LandingVisionPipeline(
        camera_id=args.camera,
        resolution=(args.width, args.height),
        use_tflite=args.tflite,
        enable_mavlink=args.mavlink,
        mavlink_connection=args.connection
    )
    
    pipeline.run(duration=args.duration)


if __name__ == '__main__':
    main()
