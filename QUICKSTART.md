# Quick Start Guide - PyTorch Wave Simulator

## Installation (2 minutes)

### 1. Clone and Setup
```bash
cd /path/to/WaveSimulator2D
pip install -r requirements.txt
```

### 2. Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS: {torch.backends.mps.is_available()}')"
```

Expected output on Apple Silicon:
```
PyTorch: 2.5.1
MPS: True
```

## Run Examples (5 minutes)

### Option A: GUI Examples (requires display)

```bash
# Basic wave propagation from point source
python wave_sim2d/examples/example0.py

# Multiple sources with modulation (requires 600×600 window)
python wave_sim2d/examples/example2.py

# Custom moving source object
python wave_sim2d/examples/example3.py

# Optical cavity with mirrors
python wave_sim2d/examples/example4.py
```

### Option B: Headless Testing

```bash
# Run comprehensive validation (no GUI needed)
python test_comprehensive.py
```

Expected output:
```
🎉 ALL TESTS PASSED! PyTorch port is working correctly.
```

## Create Your Own Simulation

### Minimal Example
```python
import torch
import wave_sim2d.wave_simulation as sim
from wave_sim2d.scene_objects.source import PointSource

# Setup
width, height = 512, 512
source = PointSource(x=256, y=256, frequency=0.1, amplitude=5)
simulator = sim.WaveSimulator2D(width, height, [source])

# Run
for step in range(1000):
    simulator.update_scene()
    simulator.update_field()

# Get results
field = simulator.get_field().cpu().numpy()
print(f"Field max: {field.max():.4f}")
```

### With Visualization
```python
import wave_sim2d.wave_visualizer as vis

# Create colormaps
field_cm = vis.get_colormap_lut('colormap_wave1', invert=False, black_level=-0.05)
intensity_cm = vis.get_colormap_lut('afmhot', invert=False)

# Create visualizer
visualizer = vis.WaveVisualizer(field_cm, intensity_cm)

# Simulate and visualize
for step in range(100):
    simulator.update_scene()
    simulator.update_field()
    visualizer.update(simulator)
    
    # Render frames (returns numpy arrays ready for display/saving)
    frame_field = visualizer.render_field(brightness_scale=1.0)
    frame_intensity = visualizer.render_intensity(brightness_scale=1.0)
```

### With Scene Objects
```python
import numpy as np
from wave_sim2d.scene_objects.static_dampening import StaticDampening
from wave_sim2d.scene_objects.static_refractive_index import StaticRefractiveIndex

# Create scene
width, height = 600, 600
objects = [
    # Add damping at borders
    StaticDampening(np.ones((height, width)), border_thickness=32),
    # Add refractive index layer
    StaticRefractiveIndex(np.full((height, width), 1.5)),
    # Add point source
    PointSource(300, 300, frequency=0.2, amplitude=8)
]

simulator = sim.WaveSimulator2D(width, height, objects)

# Simulate...
```

## Device Information

### Check GPU Status
```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"Device: {torch.device('mps' if torch.backends.mps.is_available() else 'cpu')}")
```

### Device Automatic Selection
The simulator automatically selects the best device:
- **Apple Silicon Macs**: Uses GPU (MPS backend)
- **NVIDIA GPUs**: Will use GPU (CUDA backend) if PyTorch CUDA is installed
- **CPU Fallback**: Automatically used if GPU unavailable

No code changes needed - it just works!

## Common Operations

### Access Field Data
```python
# Get as PyTorch tensor (stays on device)
field_tensor = simulator.get_field()

# Convert to NumPy for analysis
field_numpy = field_tensor.cpu().numpy()

# Get specific location
value_at_center = field_numpy[256, 256]
```

### Reset Simulation
```python
simulator.reset_time()
simulator.u.fill_(0)
simulator.u_prev.fill_(0)
```

### Change Parameters
```python
# Global dampening (0-1, lower = more damping)
simulator.global_dampening = 0.95

# Access field sizes
width, height = simulator.c.shape[1], simulator.c.shape[0]
```

## Available Scene Objects

```python
from wave_sim2d.scene_objects.source import (
    PointSource,              # Single point emitter
    LineSource,               # Linear emitter
    ModulatorSmoothSquare,    # Amplitude modulation
)
from wave_sim2d.scene_objects.static_dampening import (
    StaticDampening,          # Absorption layer
)
from wave_sim2d.scene_objects.static_refractive_index import (
    StaticRefractiveIndex,        # Uniform refractive index
    StaticRefractiveIndexPolygon, # Custom polygon
    StaticRefractiveIndexBox,     # Rotated box
)
from wave_sim2d.scene_objects.strain_refractive_index import (
    StrainRefractiveIndex,    # Coupled strain-optical effect
)
```

## Performance Tips

### For Speed
- Use larger grid sizes (512×512+) to saturate GPU
- Minimize `.cpu()` calls (only for visualization/analysis)
- Use float32 (default) rather than float64

### For Stability
- Keep global_dampening close to 1.0
- Use adequate border dampening
- Limit amplitude to < 10 for most cases
- Check Courant number: should satisfy wave speed < 1.0

### For Memory
- Reduce grid size if out of memory
- Use CPU if running many simulations sequentially
- Clear PyTorch cache: `torch.mps.empty_cache()`

## Troubleshooting

### MPS Not Detected
```python
# Update PyTorch if needed
# pip install --upgrade torch
import torch
torch.backends.mps.is_available()  # Should be True
```

### Slow Simulation
```python
# Check device being used
print(simulator.device)

# Use larger grid to better utilize GPU
# 128×128 may be too small for GPU advantage
```

### Output Not Appearing
```python
# Ensure visualization window stays open
import cv2
cv2.waitKey(1)  # Add to update loop for real-time display
```

## Documentation

- **Detailed Migration**: See `PYTORCH_MIGRATION.md`
- **Completion Report**: See `COMPLETION_REPORT.md`
- **Examples**: See `wave_sim2d/examples/`
- **API Reference**: Source code docstrings

## Support

For issues or questions:
1. Check `PYTORCH_MIGRATION.md` for technical details
2. Review examples in `wave_sim2d/examples/`
3. Run `test_comprehensive.py` to validate setup
4. Check Python version: Python 3.8+ required

---

**Last Updated**: November 14, 2025  
**Status**: Production Ready ✅
