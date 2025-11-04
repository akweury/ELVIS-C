# AI Model Interface Guide

## 🎯 **Designed for AI Model Workflow**

This interface is specifically designed for AI models that:
1. **Perceive parameters** from existing video datasets
2. **Decide on interventions** based on analysis
3. **Request specific videos** with exact parameter values
4. **Receive generated videos** for further analysis

## 🚀 **Core Workflow: Parameter → Video**

```python
from elvis_env.interface import AIVideoInterface, generate_intervention_video

# Step 1: AI perceives parameters from existing video
ai_perceived_params = {
    "hole_diameter": 35,      # AI detected hole size
    "wind_strength": 1.5,     # AI detected wind
    "num_circles": 8,         # AI counted circles
    "width": 400,             # Video dimensions  
    "height": 400,
    "num_frames": 80
}

# Step 2: AI decides on intervention (e.g., bigger hole to reduce jams)
result = generate_intervention_video(
    baseline_parameters=ai_perceived_params,
    intervention_target="hole_diameter", 
    intervention_value=70,    # AI's chosen intervention
    seed=42
)

# Step 3: AI receives videos
baseline_video = result.baseline_result.frames     # List of numpy arrays
intervention_video = result.intervention_result.frames  # List of numpy arrays

# Step 4: AI analyzes results
print(f"Effect: {result.effect_description}")
```

## 🎮 **Simple Interface Methods**

### Generate Single Video
```python
interface = AIVideoInterface()

# AI specifies exact parameters
params = {
    "hole_diameter": 45,
    "wind_strength": 2.0,
    "num_circles": 6,
    "width": 400,
    "height": 400
}

result = interface.generate_video(params, seed=42)
video_frames = result.frames  # numpy arrays ready for analysis
```

### Generate Intervention Pair
```python
# AI perceived baseline parameters
baseline = {
    "hole_diameter": 30,
    "wind_strength": 0.0,
    "num_circles": 10,
    "width": 400,
    "height": 400
}

# AI wants to test larger hole
intervention = baseline.copy()
intervention["hole_diameter"] = 60

result = interface.create_intervention_pair(baseline, intervention, seed=42)

# AI gets both videos for comparison
baseline_frames = result.baseline_result.frames
intervention_frames = result.intervention_result.frames
```

### Quick Single Parameter Change
```python
result = interface.quick_intervention(
    baseline_parameters=ai_perceived_params,
    intervention_target="wind_strength",
    intervention_value=4.0,
    seed=42
)
```

## 📊 **Parameter Management**

### Get Parameter Template
```python
interface = AIVideoInterface()
template = interface.get_parameter_template()
print(template)
# Shows all available parameters with defaults
```

### Validate Parameters  
```python
ai_params = {"hole_diameter": 45, "wind_strength": 2.0}
is_valid, message = interface.validate_parameters(ai_params)

if not is_valid:
    print(f"Parameter issue: {message}")
```

### Available Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| `hole_diameter` | int | Size of exit hole (affects traffic flow) |
| `wind_strength` | float | Horizontal wind force |
| `num_circles` | int | Maximum circles to spawn |
| `circle_size_min/max` | int | Circle size range |
| `spawn_rate` | float | Spawning probability per frame |
| `hole_x_position` | float | Hole position (0.0=left, 1.0=right) |
| `wind_direction` | int | Wind direction (1=right, -1=left) |
| `width/height` | int | Video dimensions |
| `num_frames` | int | Video length |

## 💾 **Saving Results**

```python
# Save intervention videos as GIFs
saved_files = interface.save_videos(
    result, 
    output_dir="ai_experiments",
    prefix="test1"
)
# Creates: test1_baseline.gif, test1_intervention.gif

# Access file paths
baseline_path = saved_files["baseline"]
intervention_path = saved_files["intervention"]
```

## 🔬 **Advanced AI Patterns**

### Hypothesis Testing
```python
interface = AIVideoInterface()

# Test multiple hypotheses
hypotheses = [
    ("hole_diameter", 60, "Larger hole reduces jams"),
    ("wind_strength", 3.0, "Wind helps circle flow"),
    ("num_circles", 4, "Fewer circles prevents jams")
]

for param, value, hypothesis in hypotheses:
    result = interface.quick_intervention(
        baseline_parameters=ai_baseline,
        intervention_target=param,
        intervention_value=value,
        seed=42
    )
    
    # AI analyzes each result
    if result.success:
        print(f"Testing: {hypothesis}")
        print(f"Result: {result.effect_description}")
```

### Adaptive Parameter Search
```python
# AI iteratively refines parameter search
current_params = ai_perceived_baseline.copy()
best_outcome = None

for hole_size in [20, 40, 60, 80]:
    test_params = current_params.copy()
    test_params["hole_diameter"] = hole_size
    
    result = interface.generate_video(test_params, seed=42)
    outcome = result.metadata.get('actual_jam_type')
    
    # AI learning logic
    if outcome == "no_jam":  # AI's desired outcome
        best_outcome = hole_size
        break

print(f"AI found optimal hole size: {best_outcome}")
```

### Batch Video Generation
```python
# AI generates multiple scenarios efficiently
scenarios = [
    {"hole_diameter": 30, "wind_strength": 0.0},
    {"hole_diameter": 30, "wind_strength": 2.0},
    {"hole_diameter": 50, "wind_strength": 0.0},
    {"hole_diameter": 50, "wind_strength": 2.0}
]

results = []
for i, params in enumerate(scenarios):
    result = interface.generate_video(params, seed=42)
    if result.success:
        results.append(result)
        interface.save_videos(result, prefix=f"scenario_{i}")

# AI analyzes all scenarios
print(f"Generated {len(results)} videos for analysis")
```

## ⚡ **Key Benefits for AI Models**

1. **Parameter-Driven**: Specify exact parameters, get exact videos
2. **No Defaults**: AI has full control over all parameters
3. **Reproducible**: Same parameters + seed = identical videos  
4. **Fast Generation**: Optimized for batch processing
5. **Error Handling**: Graceful failure with informative messages
6. **Flexible Output**: Numpy arrays ready for ML pipelines

## 🎯 **Perfect for AI Research**

- **Causal Discovery**: Test causal hypotheses systematically
- **Counterfactual Analysis**: Generate perfect intervention pairs
- **Parameter Sensitivity**: Study effect of parameter changes
- **Dataset Augmentation**: Create training data with specific properties
- **Ablation Studies**: Isolate effects of individual parameters

The interface is designed to be **transparent and predictable** - what you specify is exactly what you get! �