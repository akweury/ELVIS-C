# Falling Circles Environment Module

## 🎯 **Purpose**

The `falling_circles_env.py` module provides a **clean, reusable physics simulation engine** that can be used by any script to generate falling circles videos with **guaranteed consistent mechanics**.

## 🔧 **Why Extract the Environment?**

**Before**: Physics logic was embedded in `falling_circles.py` with I/O operations
**After**: Clean separation - environment handles physics, scripts handle I/O

### Benefits:
- ✅ **Consistent Physics**: Same simulation mechanics across all scripts
- ✅ **Reusable Objects**: Create environment once, generate multiple videos
- ✅ **Cleaner Code**: Separation of physics simulation and I/O operations
- ✅ **Easy Testing**: Test physics logic independently
- ✅ **Backward Compatible**: Same API as original `generate_falling_circles_video()`

## 🚀 **Usage**

### Option 1: Compatibility Function (Drop-in Replacement)
```python
from falling_circles_env import generate_falling_circles_video, VideoParams

# Same API as original falling_circles.py
params = VideoParams(hole_diameter=40, wind_strength=2.0, num_circles=8)
frames, metadata = generate_falling_circles_video(params, seed=42)
```

### Option 2: Environment Object (Recommended)
```python
from falling_circles_env import FallingCirclesEnvironment, VideoParams

# Create environment once
params = VideoParams(hole_diameter=40, wind_strength=2.0, num_circles=8)
env = FallingCirclesEnvironment(params)

# Generate multiple videos
frames1, metadata1 = env.generate_video(seed=42)
env.reset()  # Reset for next video
frames2, metadata2 = env.generate_video(seed=123)
```

## 🏗️ **Architecture**

### Core Classes:

**`VideoParams`**: Configuration for simulation parameters
- Same interface as original
- All physics parameters: hole size, wind, gravity, etc.

**`Circle`**: Individual circle with physics properties  
- Position, velocity, size, color
- Active/exited status, stuck counter

**`FallingCirclesEnvironment`**: Main simulation engine
- Handles all physics updates
- Manages circle spawning and collisions
- Renders frames with funnel and circles
- Determines jam types

### Methods:

**`env.generate_video()`**: Generate complete video simulation
**`env.reset()`**: Reset environment for reuse
**`env._update_physics()`**: Single physics time step
**`env._render_frame()`**: Render single frame

## 🎮 **Physics Components**

The environment maintains **identical physics** to the original:

1. **Circle Spawning**: Based on spawn rate and max circles
2. **Gravity & Wind**: Acceleration and velocity updates
3. **Wall Collisions**: Bouncing off side walls
4. **Funnel Physics**: V-shaped slope with sliding forces
5. **Exit Logic**: Complex hole exit conditions
6. **Circle Collisions**: Simple elastic collisions
7. **Jam Detection**: Statistical analysis of exit patterns

## 📊 **API Reference**

### VideoParams
```python
VideoParams(
    num_frames=60,          # Number of simulation frames
    width=224, height=224,  # Video dimensions
    num_circles=15,         # Maximum circles to spawn
    circle_size_min=4,      # Minimum circle radius
    circle_size_max=8,      # Maximum circle radius
    hole_diameter=20,       # Exit hole size
    hole_x_position=0.5,    # Hole position (0.0-1.0)
    wind_strength=0.0,      # Wind force
    wind_direction=1,       # Wind direction (-1 or 1)
    gravity=0.8,            # Gravitational acceleration
    spawn_rate=0.3,         # Spawn probability per frame
    circle_color=None,      # RGB color list
    background_color=None,  # RGB color list
    noise_level=0.0         # Visual noise amount
)
```

### Environment Methods
```python
env = FallingCirclesEnvironment(params)

# Generate video
frames, metadata = env.generate_video(
    seed=42,                    # Random seed
    include_labels=False,       # Show info labels
    actual_jam_type=None        # Override jam type display
)

# Reset for reuse
env.reset()
```

### Return Values
```python
frames: List[np.ndarray]       # Video frames as numpy arrays
metadata: {
    'params': dict,            # Original parameters
    'seed': int,               # Random seed used
    'frames': List[dict],      # Per-frame statistics
    'actual_jam_type': str,    # 'no_jam', 'partial_jam', 'full_jam'
    'exit_statistics': dict    # Final exit counts and ratios
}
```

## 🔧 **Migration Guide**

### For Existing Scripts:

**Minimal Change** (drop-in replacement):
```python
# Old
from falling_circles import generate_falling_circles_video, VideoParams

# New  
from falling_circles_env import generate_falling_circles_video, VideoParams
# Everything else stays the same!
```

**Recommended Upgrade** (better performance):
```python
# Old
for i in range(100):
    frames, meta = generate_falling_circles_video(params, seed=i)

# New
env = FallingCirclesEnvironment(params)
for i in range(100):
    frames, meta = env.generate_video(seed=i)
    env.reset()  # Reuse environment object
```

## 🧪 **Testing**

Run the demo script to verify everything works:
```bash
cd elvis_env/scripts
python demo_environment.py
```

This will test:
- Basic environment usage
- Environment reuse
- Parameter variations  
- Custom physics scenarios
- Video saving

## 🎯 **Scripts Updated**

The following scripts have been updated to use the new environment module:

- ✅ `test_interface.py` - AI interface tests
- ✅ `test_ai_comprehensive.py` - Comprehensive AI tests  
- ✅ `generate_intervention_pairs.py` - Intervention pair generation
- ✅ `generate_active_interventions.py` - Active intervention system
- ✅ `generate_dataset_pipeline.py` - Dataset pipeline
- ✅ AI interface modules (`ai_model_interface.py`, `video_interface.py`)

## 💡 **For New Scripts**

Any new script can now easily use the environment:

```python
from falling_circles_env import FallingCirclesEnvironment, VideoParams

# Your custom parameters
params = VideoParams(
    hole_diameter=my_hole_size,
    wind_strength=my_wind,
    num_circles=my_circle_count
    # ... other parameters
)

# Generate videos with consistent physics
env = FallingCirclesEnvironment(params)
frames, metadata = env.generate_video(seed=42)

# Process frames as needed
for frame in frames:
    # frame is a numpy array, shape (height, width, 3)
    your_processing_function(frame)
```

**The environment guarantees identical physics simulation across all scripts!** 🎯