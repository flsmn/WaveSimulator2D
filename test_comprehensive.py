#!/usr/bin/env python3
"""
Comprehensive end-to-end test of PyTorch port.
Validates all components and examples work correctly.
"""
import sys
import os
sys.path.insert(0, '/Users/sebastian/Documents/VSCode/WaveSimulator2D')

import torch
import numpy as np
import traceback

def test_imports():
    """Test 1: Verify all imports work"""
    print("\n" + "=" * 70)
    print("TEST 1: Import Verification")
    print("=" * 70)
    
    try:
        import wave_sim2d.wave_visualizer as vis
        import wave_sim2d.wave_simulation as sim
        from wave_sim2d.scene_objects.source import PointSource, LineSource, ModulatorSmoothSquare
        from wave_sim2d.scene_objects.static_dampening import StaticDampening
        from wave_sim2d.scene_objects.static_refractive_index import (
            StaticRefractiveIndex, StaticRefractiveIndexPolygon, StaticRefractiveIndexBox
        )
        from wave_sim2d.scene_objects.static_image_scene import StaticImageScene
        from wave_sim2d.scene_objects.strain_refractive_index import StrainRefractiveIndex
        
        print("✓ All imports successful")
        print(f"  Device available: {torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')}")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_device_detection():
    """Test 2: Device detection and fallback"""
    print("\n" + "=" * 70)
    print("TEST 2: Device Detection")
    print("=" * 70)
    
    try:
        mps_available = torch.backends.mps.is_available()
        device = torch.device("mps") if mps_available else torch.device("cpu")
        print(f"✓ MPS available: {mps_available}")
        print(f"✓ Using device: {device}")
        
        # Test tensor creation on device
        t = torch.ones(2, 2, device=device)
        print(f"✓ Tensor device: {t.device}")
        return True
    except Exception as e:
        print(f"✗ Device detection failed: {e}")
        return False

def test_simulator_creation():
    """Test 3: Simulator initialization"""
    print("\n" + "=" * 70)
    print("TEST 3: Simulator Creation")
    print("=" * 70)
    
    try:
        import wave_sim2d.wave_simulation as sim
        from wave_sim2d.scene_objects.source import PointSource
        
        width, height = 256, 256
        objects = [PointSource(128, 128, 0.1, 3)]
        simulator = sim.WaveSimulator2D(width, height, objects)
        
        print(f"✓ Simulator created")
        print(f"  - Device: {simulator.device}")
        print(f"  - Field shape: {simulator.u.shape}")
        print(f"  - Field dtype: {simulator.u.dtype}")
        print(f"  - Memory on GPU: {torch.cuda.memory_allocated() if torch.cuda.is_available() else 'N/A'}")
        
        assert simulator.u.dtype == torch.float32, "Field not float32!"
        assert simulator.u.shape == (height, width), "Field shape incorrect!"
        return True
    except Exception as e:
        print(f"✗ Simulator creation failed: {e}")
        traceback.print_exc()
        return False

def test_simulation_steps():
    """Test 4: Simulation stepping"""
    print("\n" + "=" * 70)
    print("TEST 4: Simulation Stepping")
    print("=" * 70)
    
    try:
        import wave_sim2d.wave_simulation as sim
        from wave_sim2d.scene_objects.source import PointSource
        
        width, height = 128, 128
        objects = [PointSource(64, 64, 0.1, 2)]
        simulator = sim.WaveSimulator2D(width, height, objects)
        
        initial_energy = torch.sum(simulator.u**2).item()
        
        for i in range(100):
            simulator.update_scene()
            simulator.update_field()
        
        final_energy = torch.sum(simulator.u**2).item()
        max_field = torch.max(torch.abs(simulator.u)).item()
        
        print(f"✓ 100 simulation steps completed")
        print(f"  - Initial energy: {initial_energy:.6e}")
        print(f"  - Final energy: {final_energy:.6e}")
        print(f"  - Max field value: {max_field:.6f}")
        print(f"  - Time: {simulator.t}")
        
        assert not torch.isnan(simulator.u).any(), "NaN detected!"
        assert not torch.isinf(simulator.u).any(), "Inf detected!"
        assert max_field > 0.1, "No wave propagation detected!"
        return True
    except Exception as e:
        print(f"✗ Simulation stepping failed: {e}")
        traceback.print_exc()
        return False

def test_all_scene_objects():
    """Test 5: All scene object types"""
    print("\n" + "=" * 70)
    print("TEST 5: Scene Objects")
    print("=" * 70)
    
    import wave_sim2d.wave_simulation as sim
    from wave_sim2d.scene_objects.source import PointSource, LineSource
    from wave_sim2d.scene_objects.static_dampening import StaticDampening
    from wave_sim2d.scene_objects.static_refractive_index import (
        StaticRefractiveIndex, StaticRefractiveIndexBox
    )
    from wave_sim2d.scene_objects.strain_refractive_index import StrainRefractiveIndex
    
    tests = [
        ("PointSource", lambda: [
            PointSource(64, 64, 0.1, 2)
        ], 128, 128),
        ("LineSource", lambda: [
            LineSource((50, 64), (100, 64), 0.2, 2)
        ], 128, 128),
        ("StaticDampening", lambda: [
            StaticDampening(np.ones((128, 128), dtype=np.float32), 16),
            PointSource(64, 64, 0.1, 2)
        ], 128, 128),
        ("StaticRefractiveIndex", lambda: [
            StaticRefractiveIndex(np.full((128, 128), 1.5, dtype=np.float32)),
            PointSource(64, 64, 0.1, 2)
        ], 128, 128),
        ("StaticRefractiveIndexBox", lambda: [
            StaticRefractiveIndexBox((128, 128), (60, 60), 0.0, 2.0),
            PointSource(50, 128, 0.1, 2)
        ], 256, 256),
        ("StrainRefractiveIndex", lambda: [
            StrainRefractiveIndex(1.0, 0.1),
            PointSource(64, 64, 0.1, 2)
        ], 128, 128),
    ]
    
    all_passed = True
    for name, objects_fn, w, h in tests:
        try:
            objects = objects_fn()
            simulator = sim.WaveSimulator2D(w, h, objects)
            
            for i in range(30):
                simulator.update_scene()
                simulator.update_field()
            
            max_val = torch.max(torch.abs(simulator.u)).item()
            print(f"  ✓ {name:30s} - max field: {max_val:8.4f}")
            
            assert not torch.isnan(simulator.u).any(), f"NaN in {name}!"
        except Exception as e:
            print(f"  ✗ {name:30s} - {str(e)[:40]}")
            all_passed = False
    
    return all_passed

def test_visualizer():
    """Test 6: Visualization pipeline"""
    print("\n" + "=" * 70)
    print("TEST 6: Visualization Pipeline")
    print("=" * 70)
    
    try:
        import wave_sim2d.wave_visualizer as vis
        import wave_sim2d.wave_simulation as sim
        from wave_sim2d.scene_objects.source import PointSource
        
        # Create colormaps
        field_colormap = vis.get_colormap_lut('colormap_wave1', invert=False, black_level=-0.05)
        intensity_colormap = vis.get_colormap_lut('afmhot', invert=False, black_level=0.0)
        
        print(f"✓ Colormaps created")
        print(f"  - Field colormap: {field_colormap.shape}")
        print(f"  - Intensity colormap: {intensity_colormap.shape}")
        
        # Create simulator
        width, height = 128, 128
        objects = [PointSource(64, 64, 0.1, 2)]
        simulator = sim.WaveSimulator2D(width, height, objects)
        visualizer = vis.WaveVisualizer(field_colormap, intensity_colormap)
        
        # Run and render
        for i in range(20):
            simulator.update_scene()
            simulator.update_field()
            visualizer.update(simulator)
        
        frame_field = visualizer.render_field(1.0)
        frame_intensity = visualizer.render_intensity(1.0)
        
        print(f"✓ Rendering successful")
        print(f"  - Field frame shape: {frame_field.shape}, dtype: {frame_field.dtype}")
        print(f"  - Intensity frame shape: {frame_intensity.shape}, dtype: {frame_intensity.dtype}")
        
        return True
    except Exception as e:
        print(f"✗ Visualization failed: {e}")
        traceback.print_exc()
        return False

def test_numerical_stability():
    """Test 7: Long-term numerical stability"""
    print("\n" + "=" * 70)
    print("TEST 7: Numerical Stability (500 steps)")
    print("=" * 70)
    
    try:
        import wave_sim2d.wave_simulation as sim
        from wave_sim2d.scene_objects.source import PointSource
        
        width, height = 128, 128
        objects = [PointSource(64, 64, 0.1, 1)]
        simulator = sim.WaveSimulator2D(width, height, objects)
        
        max_val = 0.0
        for i in range(500):
            simulator.update_scene()
            simulator.update_field()
            
            current_max = torch.max(torch.abs(simulator.u)).item()
            max_val = max(max_val, current_max)
            
            if torch.isnan(simulator.u).any() or torch.isinf(simulator.u).any():
                print(f"✗ NaN/Inf at step {i}")
                return False
        
        print(f"✓ 500 steps completed without instability")
        print(f"  - Max field value: {max_val:.6f}")
        print(f"  - Final time: {simulator.t}")
        
        assert max_val < 1e6, "Field values growing too large!"
        return True
    except Exception as e:
        print(f"✗ Stability test failed: {e}")
        traceback.print_exc()
        return False

def test_tensor_to_numpy():
    """Test 8: Tensor/NumPy conversions"""
    print("\n" + "=" * 70)
    print("TEST 8: Tensor/NumPy Conversions")
    print("=" * 70)
    
    try:
        import wave_sim2d.wave_simulation as sim
        from wave_sim2d.scene_objects.source import PointSource
        
        width, height = 128, 128
        objects = [PointSource(64, 64, 0.1, 2)]
        simulator = sim.WaveSimulator2D(width, height, objects)
        
        for i in range(10):
            simulator.update_scene()
            simulator.update_field()
        
        # Test tensor conversion
        field_tensor = simulator.get_field()
        field_numpy = field_tensor.cpu().numpy()
        
        print(f"✓ Conversions successful")
        print(f"  - Tensor type: {type(field_tensor)}, device: {field_tensor.device}")
        print(f"  - NumPy type: {type(field_numpy)}, dtype: {field_numpy.dtype}")
        print(f"  - NumPy shape: {field_numpy.shape}")
        print(f"  - Range: [{field_numpy.min():.6f}, {field_numpy.max():.6f}]")
        
        assert isinstance(field_tensor, torch.Tensor), "Not a tensor!"
        assert isinstance(field_numpy, np.ndarray), "Not a NumPy array!"
        return True
    except Exception as e:
        print(f"✗ Conversion test failed: {e}")
        traceback.print_exc()
        return False

def test_example_simulations():
    """Test 9: Run simplified versions of all examples"""
    print("\n" + "=" * 70)
    print("TEST 9: Example Simulations")
    print("=" * 70)
    
    import wave_sim2d.wave_visualizer as vis
    import wave_sim2d.wave_simulation as sim
    from wave_sim2d.scene_objects.source import PointSource, LineSource, ModulatorSmoothSquare
    from wave_sim2d.scene_objects.static_dampening import StaticDampening
    from wave_sim2d.scene_objects.static_refractive_index import StaticRefractiveIndex, StaticRefractiveIndexBox
    
    examples = [
        ("Example0", lambda: [PointSource(200, 256, 0.1, 5)], 512, 512, 50),
        ("Example2", lambda: [
            StaticDampening(np.ones((600, 600), dtype=np.float32), 32),
            StaticRefractiveIndex(np.full((600, 600), 1.5, dtype=np.float32)),
            PointSource(200, 220, 0.2, 8),
            PointSource(200, 380, 0.2, 8, amp_modulator=ModulatorSmoothSquare(0.025, 0.0, 0.5))
        ], 600, 600, 50),
        ("Example3", lambda: [
            StaticDampening(np.ones((600, 600), dtype=np.float32), 64),
            StaticRefractiveIndex(np.full((600, 600), 1.5, dtype=np.float32)),
            PointSource(300, 300, 0.1, 1)
        ], 600, 600, 50),
        ("Example4", lambda: [
            StaticDampening(np.ones((512, 768), dtype=np.float32), 48),
            StaticRefractiveIndexBox((50, 256), (50, int(512*0.8)), 0.0, 100.0),
            StaticRefractiveIndexBox((588, 256), (40, int(512*0.8)), 0.0, 10.0),
            LineSource((77, 116), (77, 396), 0.1003, amplitude=0.3)
        ], 768, 512, 50),
    ]
    
    all_passed = True
    for name, objects_fn, w, h, steps in examples:
        try:
            field_colormap = vis.get_colormap_lut('colormap_wave1', invert=False, black_level=-0.05)
            objects = objects_fn()
            simulator = sim.WaveSimulator2D(w, h, objects)
            
            for i in range(steps):
                simulator.update_scene()
                simulator.update_field()
            
            max_val = torch.max(torch.abs(simulator.u)).item()
            print(f"  ✓ {name:15s} - size {w:3d}×{h:3d}, max: {max_val:8.4f}")
        except Exception as e:
            print(f"  ✗ {name:15s} - {str(e)[:40]}")
            all_passed = False
    
    return all_passed

def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "PyTorch WaveSimulator2D - Comprehensive Test Suite" + " " * 4 + "║")
    print("╚" + "=" * 68 + "╝")
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Device Detection", test_device_detection()))
    results.append(("Simulator Creation", test_simulator_creation()))
    results.append(("Simulation Stepping", test_simulation_steps()))
    results.append(("Scene Objects", test_all_scene_objects()))
    results.append(("Visualization", test_visualizer()))
    results.append(("Numerical Stability", test_numerical_stability()))
    results.append(("Tensor/NumPy Conv", test_tensor_to_numpy()))
    results.append(("Example Simulations", test_example_simulations()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8s} - {name}")
    
    print("=" * 70)
    print(f"Result: {passed}/{total} test groups passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! PyTorch port is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test group(s) failed. See above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
