# ELVIS-C Interface Package

Simple, AI-friendly interface for generating intervention videos in the falling circles environment.

## Quick Start

```python
from elvis_env.interface import InterventionInterface, quick_intervene

# Simple intervention
result = quick_intervene("hole_diameter", 80, seed=42)

# Access the generated videos
baseline_video = result.baseline_frames      # List of numpy arrays
intervention_video = result.intervention_frames  # List of numpy arrays

# Check the effect
print(f"Effect: {result.effect_description}")
print(f"Magnitude: {result.effect_magnitude}")
```

## Main Interfaces

### 1. InterventionInterface (Recommended for AI Models)

The high-level interface designed for AI models:

```python
interface = InterventionInterface(default_seed=42)

# Single parameter intervention
result = interface.intervene(
    intervention_target="hole_diameter",
    intervention_value=80
)

# Multiple parameter intervention
result = interface.multi_intervene(
    interventions={
        "hole_diameter": 60,
        "wind_strength": 4.0,
        "num_circles": 8
    }
)

# Parameter exploration
results = interface.explore_parameter(
    parameter_name="wind_strength",
    values=[1.0, 3.0, 5.0]
)
```

### 2. VideoInterface (Lower-level)

For more direct control over video generation:

```python
from elvis_env.interface import VideoInterface

interface = VideoInterface(default_seed=42)

# Generate single video
frames, metadata = interface.generate_video(
    params={"hole_diameter": 80, "wind_strength": 3.0}
)

# Create intervention pair
result = interface.create_intervention(
    baseline_params={"hole_diameter": 40},
    intervention_params={"hole_diameter": 80}
)
```

## Parameters

Available parameters for intervention:

| Parameter | Type | Range | Default | Effect |
|-----------|------|--------|---------|---------|
| `hole_diameter` | int | 10-200 | 40 | Larger holes allow more circles to exit |
| `wind_strength` | float | 0.0-10.0 | 2.0 | Higher wind pushes circles sideways |
| `num_circles` | int | 1-50 | 5 | More circles can create traffic jams |
| `circle_size_min` | int | 3-30 | 8 | Larger circles have harder time fitting |
| `circle_size_max` | int | 3-30 | 12 | Larger circles have harder time fitting |
| `spawn_rate` | int | 1-20 | 3 | Lower values spawn circles faster |
| `exit_ratio` | float | 0.1-1.0 | 0.8 | Lower values make exiting easier |

## Result Structure

Each intervention returns an `InterventionResult` object:

```python
result = interface.intervene("hole_diameter", 80)

# Access videos (numpy arrays)
baseline_frames = result.baseline_frames
intervention_frames = result.intervention_frames

# Access metadata
baseline_meta = result.baseline_metadata
intervention_meta = result.intervention_metadata

# Access analysis
effect_magnitude = result.effect_magnitude  # 0.0-1.0
effect_description = result.effect_description  # Human readable
```

## Saving Results

```python
# Save as GIFs
interface.video_interface.save_intervention_gifs(
    result, 
    "baseline.gif", 
    "intervention.gif"
)

# Save individual video as GIF
interface.video_interface.save_frames_as_gif(
    result.baseline_frames,
    "my_video.gif"
)
```

## Examples

See `elvis_env/examples/ai_interface_examples.py` for comprehensive examples including:

1. Basic single parameter intervention
2. Multi-parameter intervention  
3. Parameter space exploration
4. Custom baseline scenarios
5. AI model workflow patterns

Run the examples:

```bash
cd elvis_env/examples
python ai_interface_examples.py
```

## For AI Models

The interface is designed to be extremely simple for AI models:

1. **No complex setup required** - just import and use
2. **Automatic parameter validation** - invalid values are constrained to valid ranges
3. **Consistent return format** - always get videos + analysis
4. **Built-in effect analysis** - automatic computation of intervention effects
5. **Flexible input formats** - accept dicts or parameter objects
6. **Reproducible results** - consistent seeding for deterministic outcomes

## Common AI Model Patterns

```python
# Pattern 1: Test single parameter effect
result = quick_intervene("wind_strength", 5.0)
if result.effect_magnitude > 0.1:
    print("Significant effect detected!")

# Pattern 2: Find optimal parameter value
values = [20, 40, 60, 80, 100]
results = compare_parameters("hole_diameter", values)
best_value = values[max(range(len(results)), key=lambda i: results[i].effect_magnitude)]

# Pattern 3: Multi-parameter optimization
interface = InterventionInterface()
result = interface.multi_intervene({
    "hole_diameter": ai_model.predict_optimal_hole_size(),
    "wind_strength": ai_model.predict_optimal_wind(),
    "num_circles": current_circle_count
})
```