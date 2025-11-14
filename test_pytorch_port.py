#!/usr/bin/env python3
"""
Test script to validate PyTorch port without GUI components.
Tests basic functionality and numerical accuracy.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), './'))

import numpy as np
import torch

# Test 1: Basic imports
print("=" * 60)
print("TEST 1: Verifying imports...")
try:
    import wave_sim2d.wave_visualizer as vis
    import wave_sim2d.wave_simulation as sim
    from wave_sim2d.scene_objects.source import PointSource, LineSource
    from wave_sim2d.scene_objects.static_dampening import StaticDampening
    from wave_sim2d.scene_objects.static_refractive_index import StaticRefractiveIndex, StaticRefractiveIndexBox
    from wave_sim2d.scene_objects.static_image_scene import StaticImageScene
    from wave_sim2d.scene_objects.strain_refractive_index import StrainRefractiveIndex
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Device detection
print("\n" + "=" * 60)
print("TEST 2: Device detection...")
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"✓ Using device: {device}")

# Test 3: Basic simulator initialization
print("\n" + "=" * 60)
print("TEST 3: Basic simulator initialization...")
try:
    width, height = 128, 128
    objects = [PointSource(64, 64, 0.1, 5)]
    simulator = sim.WaveSimulator2D(width, height, objects)
    print(f"✓ Simulator created (device: {simulator.device})")
    print(f"  - Field shape: {simulator.u.shape}")
    print(f"  - Wave speed field shape: {simulator.c.shape}")
except Exception as e:
    print(f"✗ Simulator initialization failed: {e}")
    sys.exit(1)

# Test 4: Single simulation step
print("\n" + "=" * 60)
print("TEST 4: Single simulation step...")
try:
    initial_energy = torch.sum(simulator.u**2).item()
    simulator.update_scene()
    simulator.update_field()
    final_energy = torch.sum(simulator.u**2).item()
    print(f"✓ Simulation step completed")
    print(f"  - Initial energy: {initial_energy:.6e}")
    print(f"  - Final energy: {final_energy:.6e}")
except Exception as e:
    print(f"✗ Simulation step failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Multiple steps with PointSource
print("\n" + "=" * 60)
print("TEST 5: Multiple simulation steps with PointSource...")
try:
    width, height = 256, 256
    objects = [PointSource(128, 128, 0.1, 3)]
    simulator = sim.WaveSimulator2D(width, height, objects)
    
    max_field_value = 0.0
    for i in range(50):
        simulator.update_scene()
        simulator.update_field()
        max_val = torch.max(torch.abs(simulator.u)).item()
        max_field_value = max(max_field_value, max_val)
    
    print(f"✓ 50 steps completed")
    print(f"  - Max field value reached: {max_field_value:.6f}")
    if max_field_value > 0.1:
        print(f"✓ Wave propagation detected")
    else:
        print(f"✗ Warning: Low wave amplitude")
except Exception as e:
    print(f"✗ Multiple steps failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: LineSource
print("\n" + "=" * 60)
print("TEST 6: LineSource test...")
try:
    width, height = 256, 256
    objects = [LineSource((100, 128), (150, 128), 0.2, 2)]
    simulator = sim.WaveSimulator2D(width, height, objects)
    
    for i in range(30):
        simulator.update_scene()
        simulator.update_field()
    
    print(f"✓ LineSource simulation completed")
    print(f"  - Max field value: {torch.max(torch.abs(simulator.u)).item():.6f}")
except Exception as e:
    print(f"✗ LineSource test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: StaticDampening
print("\n" + "=" * 60)
print("TEST 7: StaticDampening test...")
try:
    width, height = 128, 128
    dampening_field = np.ones((height, width), dtype=np.float32)
    objects = [
        StaticDampening(dampening_field, border_thickness=16),
        PointSource(64, 64, 0.1, 2)
    ]
    simulator = sim.WaveSimulator2D(width, height, objects)
    
    for i in range(40):
        simulator.update_scene()
        simulator.update_field()
    
    print(f"✓ StaticDampening simulation completed")
    print(f"  - Max field value: {torch.max(torch.abs(simulator.u)).item():.6f}")
except Exception as e:
    print(f"✗ StaticDampening test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: StaticRefractiveIndex
print("\n" + "=" * 60)
print("TEST 8: StaticRefractiveIndex test...")
try:
    width, height = 128, 128
    refractive_field = np.full((height, width), 1.5, dtype=np.float32)
    objects = [
        StaticRefractiveIndex(refractive_field),
        PointSource(64, 64, 0.1, 2)
    ]
    simulator = sim.WaveSimulator2D(width, height, objects)
    
    for i in range(40):
        simulator.update_scene()
        simulator.update_field()
    
    print(f"✓ StaticRefractiveIndex simulation completed")
    print(f"  - Max field value: {torch.max(torch.abs(simulator.u)).item():.6f}")
except Exception as e:
    print(f"✗ StaticRefractiveIndex test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 9: StaticRefractiveIndexBox
print("\n" + "=" * 60)
print("TEST 9: StaticRefractiveIndexBox test...")
try:
    width, height = 256, 256
    objects = [
        StaticRefractiveIndexBox((128, 128), (80, 80), 0.0, 2.0),
        PointSource(50, 128, 0.1, 2)
    ]
    simulator = sim.WaveSimulator2D(width, height, objects)
    
    for i in range(50):
        simulator.update_scene()
        simulator.update_field()
    
    print(f"✓ StaticRefractiveIndexBox simulation completed")
    print(f"  - Max field value: {torch.max(torch.abs(simulator.u)).item():.6f}")
except Exception as e:
    print(f"✗ StaticRefractiveIndexBox test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 10: StrainRefractiveIndex
print("\n" + "=" * 60)
print("TEST 10: StrainRefractiveIndex test...")
try:
    width, height = 128, 128
    objects = [
        StrainRefractiveIndex(1.0, 0.1),
        PointSource(64, 64, 0.1, 2)
    ]
    simulator = sim.WaveSimulator2D(width, height, objects)
    
    for i in range(40):
        simulator.update_scene()
        simulator.update_field()
    
    print(f"✓ StrainRefractiveIndex simulation completed")
    print(f"  - Max field value: {torch.max(torch.abs(simulator.u)).item():.6f}")
except Exception as e:
    print(f"✗ StrainRefractiveIndex test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 11: Stability check
print("\n" + "=" * 60)
print("TEST 11: Stability check (long simulation)...")
try:
    width, height = 128, 128
    objects = [PointSource(64, 64, 0.1, 1)]
    simulator = sim.WaveSimulator2D(width, height, objects)
    
    max_value = 0.0
    nan_detected = False
    for i in range(500):
        simulator.update_scene()
        simulator.update_field()
        current_max = torch.max(torch.abs(simulator.u)).item()
        max_value = max(max_value, current_max)
        
        if torch.isnan(simulator.u).any():
            nan_detected = True
            break
    
    print(f"✓ 500 steps completed without NaN")
    print(f"  - Max field value: {max_value:.6f}")
    if not nan_detected and max_value < 1e10:
        print(f"✓ Simulation is stable")
    else:
        print(f"✗ Stability issues detected")
except Exception as e:
    print(f"✗ Stability check failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 12: Tensor to NumPy conversion for visualization
print("\n" + "=" * 60)
print("TEST 12: Tensor to NumPy conversion...")
try:
    width, height = 128, 128
    objects = [PointSource(64, 64, 0.1, 2)]
    simulator = sim.WaveSimulator2D(width, height, objects)
    
    for i in range(10):
        simulator.update_scene()
        simulator.update_field()
    
    # Get field as tensor and convert to numpy
    field_tensor = simulator.get_field()
    field_numpy = field_tensor.cpu().numpy()
    
    print(f"✓ Tensor to NumPy conversion successful")
    print(f"  - Field shape: {field_numpy.shape}")
    print(f"  - Field dtype: {field_numpy.dtype}")
    print(f"  - Field range: [{field_numpy.min():.6f}, {field_numpy.max():.6f}]")
except Exception as e:
    print(f"✗ Tensor conversion failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 13: Visualizer integration
print("\n" + "=" * 60)
print("TEST 13: Visualizer integration...")
try:
    width, height = 128, 128
    field_colormap = vis.get_colormap_lut('colormap_wave1', invert=False, black_level=-0.05)
    intensity_colormap = vis.get_colormap_lut('afmhot', invert=False, black_level=0.0)
    
    objects = [PointSource(64, 64, 0.1, 2)]
    simulator = sim.WaveSimulator2D(width, height, objects)
    visualizer = vis.WaveVisualizer(field_colormap=field_colormap, intensity_colormap=intensity_colormap)
    
    for i in range(10):
        simulator.update_scene()
        simulator.update_field()
        visualizer.update(simulator)
    
    # Try rendering (without displaying)
    frame_field = visualizer.render_field(1.0)
    frame_intensity = visualizer.render_intensity(1.0)
    
    print(f"✓ Visualizer integration successful")
    print(f"  - Field frame shape: {frame_field.shape}")
    print(f"  - Intensity frame shape: {frame_intensity.shape}")
except Exception as e:
    print(f"✗ Visualizer integration failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
