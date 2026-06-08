#!/usr/bin/env python3
"""
Test script for Autonomous Drone Landing Vision System
Tests individual components with synthetic data
"""

import cv2
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from landing_vision_system import (
    TerrainClassifier, 
    RoughnessAnalyzer, 
    HeatmapGenerator,
    TerrainType
)


def create_grass_image(size=(480, 640)):
    """Create synthetic grass image"""
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    
    # Base green color
    img[:, :] = [45, 120, 40]  # BGR
    
    # Add texture (grass blades)
    for _ in range(5000):
        x = np.random.randint(0, size[1])
        y = np.random.randint(0, size[0])
        cv2.circle(img, (x, y), 1, 
                   (30 + np.random.randint(30), 
                    100 + np.random.randint(50), 
                    20 + np.random.randint(40)), -1)
    
    # Add some variation
    noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img


def create_water_image(size=(480, 640)):
    """Create synthetic water image"""
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    
    # Base blue color
    img[:, :] = [150, 100, 40]  # BGR
    
    # Add smooth gradient (water surface)
    for i in range(size[0]):
        intensity = int(20 * np.sin(i / 50))
        img[i, :] = np.clip(img[i, :] + intensity, 0, 255)
    
    # Add specular highlights (reflections)
    for _ in range(5):
        x = np.random.randint(50, size[1] - 50)
        y = np.random.randint(50, size[0] - 50)
        cv2.ellipse(img, (x, y), (30, 20), np.random.randint(0, 180),
                    0, 360, (240, 240, 240), -1)
    
    # Blur for smooth water surface
    img = cv2.GaussianBlur(img, (15, 15), 0)
    
    return img


def create_concrete_image(size=(480, 640)):
    """Create synthetic concrete image
     SHA256-62aacefd53d7b91a3dd9a9bb0b3f34add00a75306628f7cdcb9943017b43e51c
     """
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    
    # Base gray color (achromatic - low saturation)
    base_gray = 120
    img[:, :] = [base_gray, base_gray, base_gray]  # BGR
    
    # Add very subtle texture (concrete is smooth but not perfectly uniform)
    noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Apply slight blur for realistic concrete appearance
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Add occasional crack lines (optional - realistic detail)
    for _ in range(2):
        x1, y1 = np.random.randint(0, size[1]), np.random.randint(0, size[0])
        x2, y2 = np.random.randint(0, size[1]), np.random.randint(0, size[0])
        cv2.line(img, (x1, y1), (x2, y2), (80, 80, 80), 1)
    
    # Add subtle color variations (weathering, stains)
    for _ in range(3):
        x = np.random.randint(50, size[1] - 50)
        y = np.random.randint(50, size[0] - 50)
        radius = np.random.randint(30, 60)
        darkness = np.random.randint(-15, -5)
        overlay = img.copy()
        cv2.circle(overlay, (x, y), radius, 
                   (base_gray + darkness, base_gray + darkness, base_gray + darkness), -1)
        img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    
    return img


def create_rough_grass_image(size=(480, 640)):
    """Create synthetic rough grass with obstacles"""
    img = create_grass_image(size)
    
    # Add obstacles (rocks, rough patches)
    for _ in range(10):
        x = np.random.randint(50, size[1] - 50)
        y = np.random.randint(50, size[0] - 50)
        w = np.random.randint(30, 80)
        h = np.random.randint(30, 80)
        
        # Dark rough patch
        cv2.rectangle(img, (x, y), (x + w, y + h), (30, 50, 20), -1)
        
        # Add edges
        cv2.rectangle(img, (x, y), (x + w, y + h), (10, 20, 5), 2)
    
    return img


def test_terrain_classifier():
    """Test terrain classification"""
    print("\n" + "="*60)
    print("Testing Terrain Classifier")
    print("="*60)
    
    classifier = TerrainClassifier(use_tflite=False)
    
    # Test grass detection
    grass_img = create_grass_image()
    terrain_type, confidence = classifier.classify(grass_img)
    
    print(f"\nTest 1: Grass Image")
    print(f"  Result: {terrain_type.value}")
    print(f"  Confidence: {confidence:.2f}")
    print(f"  Expected: grass")
    print(f"  Status: {'✓ PASS' if terrain_type == TerrainType.GRASS else '✗ FAIL'}")
    
    grass_pass = terrain_type == TerrainType.GRASS
    
    # Test water detection
    water_img = create_water_image()
    terrain_type, confidence = classifier.classify(water_img)
    
    print(f"\nTest 2: Water Image")
    print(f"  Result: {terrain_type.value}")
    print(f"  Confidence: {confidence:.2f}")
    print(f"  Expected: water")
    print(f"  Status: {'✓ PASS' if terrain_type == TerrainType.WATER else '✗ FAIL'}")
    
    water_pass = terrain_type == TerrainType.WATER
    
    # Test concrete detection
    concrete_img = create_concrete_image()
    terrain_type, confidence = classifier.classify(concrete_img)
    
    print(f"\nTest 3: Concrete Image")
    print(f"  Result: {terrain_type.value}")
    print(f"  Confidence: {confidence:.2f}")
    print(f"  Expected: concrete")
    print(f"  Status: {'✓ PASS' if terrain_type == TerrainType.CONCRETE else '✗ FAIL'}")
    
    concrete_pass = terrain_type == TerrainType.CONCRETE
    
    # Visualize all three
    vis = np.hstack([grass_img, water_img, concrete_img])
    cv2.putText(vis, "GRASS", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(vis, "WATER", (grass_img.shape[1] + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(vis, "CONCRETE", (grass_img.shape[1] + water_img.shape[1] + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0), 2)
    
    cv2.imshow('Terrain Classification Test', vis)
    cv2.waitKey(2000)
    
    return grass_pass or water_pass or concrete_pass


def test_roughness_analyzer():
    """Test roughness analysis"""
    print("\n" + "="*60)
    print("Testing Roughness Analyzer")
    print("="*60)
    
    analyzer = RoughnessAnalyzer(grid_size=32)
    
    # Test smooth grass
    smooth_grass = create_grass_image()
    smooth_map = analyzer.analyze(smooth_grass)
    avg_safety_smooth = np.mean(smooth_map)
    
    print(f"\nTest 1: Smooth Grass")
    print(f"  Average Safety Score: {avg_safety_smooth:.2f}")
    print(f"  Expected: > 0.5")
    print(f"  Status: {'✓ PASS' if avg_safety_smooth > 0.5 else '✗ FAIL'}")
    
    # Test rough grass
    rough_grass = create_rough_grass_image()
    rough_map = analyzer.analyze(rough_grass)
    avg_safety_rough = np.mean(rough_map)
    
    print(f"\nTest 2: Rough Grass with Obstacles")
    print(f"  Average Safety Score: {avg_safety_rough:.2f}")
    print(f"  Expected: < {avg_safety_smooth:.2f}")
    print(f"  Status: {'✓ PASS' if avg_safety_rough < avg_safety_smooth else '✗ FAIL'}")
    
    # Visualize
    smooth_viz = (smooth_map * 255).astype(np.uint8)
    smooth_viz = cv2.resize(smooth_viz, (640, 480))
    smooth_viz = cv2.applyColorMap(smooth_viz, cv2.COLORMAP_TURBO)
    
    rough_viz = (rough_map * 255).astype(np.uint8)
    rough_viz = cv2.resize(rough_viz, (640, 480))
    rough_viz = cv2.applyColorMap(rough_viz, cv2.COLORMAP_TURBO)
    
    vis = np.hstack([smooth_viz, rough_viz])
    cv2.putText(vis, "Smooth Grass", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(vis, "Rough Grass", (650, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow('Roughness Analysis Test', vis)
    cv2.waitKey(2000)
    
    return avg_safety_smooth > 0.5 and avg_safety_rough < avg_safety_smooth


def test_heatmap_generator():
    """Test heatmap generation and landing zone selection"""
    print("\n" + "="*60)
    print("Testing Heatmap Generator")
    print("="*60)
    
    generator = HeatmapGenerator(smoothing_kernel=5)
    analyzer = RoughnessAnalyzer(grid_size=32)
    
    # Create test image with clear landing zone
    test_img = create_grass_image()
    
    # Add a clear safe zone in the center
    h, w = test_img.shape[:2]
    center_x, center_y = w // 2, h // 2
    cv2.circle(test_img, (center_x, center_y), 100, (50, 150, 50), -1)
    
    # Analyze and generate heatmap
    roughness_map = analyzer.analyze(test_img)
    heatmap, landing_zone = generator.generate(roughness_map, test_img.shape[:2])
    
    print(f"\nTest 1: Landing Zone Selection")
    print(f"  Selected Position: ({landing_zone.x_normalized:.3f}, {landing_zone.y_normalized:.3f})")
    print(f"  Pixel Position: ({landing_zone.x_pixel}, {landing_zone.y_pixel})")
    print(f"  Safety Score: {landing_zone.safety_score:.2f}")
    
    # Check if landing zone is near center
    distance_from_center = np.sqrt(
        (landing_zone.x_normalized - 0.5)**2 + 
        (landing_zone.y_normalized - 0.5)**2
    )
    
    print(f"  Distance from Center: {distance_from_center:.3f}")
    print(f"  Expected: < 0.3")
    print(f"  Status: {'✓ PASS' if distance_from_center < 0.3 else '✗ FAIL'}")
    
    # Visualize
    vis = np.hstack([test_img, heatmap])
    cv2.putText(vis, f"Target: ({landing_zone.x_normalized:.2f}, {landing_zone.y_normalized:.2f})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis, f"Safety: {landing_zone.safety_score:.2f}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('Heatmap Generation Test', vis)
    cv2.waitKey(2000)
    
    return distance_from_center < 0.3 and landing_zone.safety_score > 0.5


def test_performance():
    """Test processing performance"""
    print("\n" + "="*60)
    print("Testing Processing Performance")
    print("="*60)
    
    import time
    
    classifier = TerrainClassifier(use_tflite=False)
    analyzer = RoughnessAnalyzer(grid_size=32)
    generator = HeatmapGenerator(smoothing_kernel=5)
    
    # Create test image
    test_img = create_grass_image()
    
    # Measure classification time
    start = time.time()
    for _ in range(10):
        _ = classifier.classify(test_img)
    classification_time = (time.time() - start) / 10 * 1000
    
    # Measure roughness analysis time
    start = time.time()
    for _ in range(10):
        roughness_map = analyzer.analyze(test_img)
    analysis_time = (time.time() - start) / 10 * 1000
    
    # Measure heatmap generation time
    start = time.time()
    for _ in range(10):
        _ = generator.generate(roughness_map, test_img.shape[:2])
    heatmap_time = (time.time() - start) / 10 * 1000
    
    total_time = classification_time + analysis_time + heatmap_time
    fps = 1000 / total_time if total_time > 0 else 0
    
    print(f"\nPerformance Metrics (640x480):")
    print(f"  Classification: {classification_time:.1f}ms")
    print(f"  Roughness Analysis: {analysis_time:.1f}ms")
    print(f"  Heatmap Generation: {heatmap_time:.1f}ms")
    print(f"  Total Pipeline: {total_time:.1f}ms")
    print(f"  Theoretical FPS: {fps:.1f}")
    print(f"\n  Target: < 200ms (5+ FPS)")
    print(f"  Status: {'✓ PASS' if total_time < 200 else '⚠ SLOW'}")
    
    return total_time < 200


def run_all_tests():
    """Run all tests"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*10 + "LANDING VISION SYSTEM - TEST SUITE" + " "*13 + "║")
    print("╚" + "="*58 + "╝")
    
    tests = [
        ("Terrain Classifier", test_terrain_classifier),
        ("Roughness Analyzer", test_roughness_analyzer),
        ("Heatmap Generator", test_heatmap_generator),
        ("Performance", test_performance),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} FAILED with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name:.<40} {status}")
    
    print("\n" + "-"*60)
    print(f"  Total: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("\n✓ All tests passed! System ready for deployment.")
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Please review errors above.")
    
    print("\nPress any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
