# AI Model Interface - Complete Guide

## ✨ **Perfect for AI Models!**

This interface is specifically designed for AI models that want to:
1. **Perceive parameters** from existing videos
2. **Specify interventions** they want to test  
3. **Generate intervention videos** with those exact parameters
4. **Analyze the results** for learning

## 🚀 **One-Line Usage**

```python
from elvis_env.interface import generate_intervention_video

# AI provides parameters it perceived + desired intervention
result = generate_intervention_video(
    baseline_parameters={
        "hole_diameter": 35,    # AI detected from video
        "wind_strength": 1.5,   # AI detected wind
        "num_circles": 8,       # AI counted circles
        "width": 400, "height": 400, "num_frames": 80
    },
    intervention_target="hole_diameter",
    intervention_value=70,      # AI's choice
    seed=42
)

# AI gets back both videos + analysis
baseline_video = result.baseline_result.frames
intervention_video = result.intervention_result.frames
```

## 📊 **What AI Models Get Back**

```python
result.success                          # True/False
result.baseline_result.frames          # List of numpy arrays (original video)
result.intervention_result.frames      # List of numpy arrays (intervention video)
result.parameters_changed              # {"hole_diameter": (35, 70)}
result.effect_description              # "Changed outcome: partial_jam → no_jam"
result.baseline_result.metadata        # Physics simulation details
result.intervention_result.metadata    # Physics simulation details
```

## 🎯 **Available Parameters to Intervene On**

| Parameter | Type | Range | Effect |
|-----------|------|--------|---------|
| `hole_diameter` | int | 10-200 | Larger = easier exit |
| `wind_strength` | float | 0.0-10.0 | Higher = more sideways push |
| `num_circles` | int | 1-50 | More = potential congestion |
| `circle_size_min` | int | 3-30 | Larger = harder to fit |
| `circle_size_max` | int | 3-30 | Larger = harder to fit |
| `spawn_rate` | float | 0.0-1.0 | Higher = faster spawning |
| `hole_x_position` | float | 0.0-1.0 | Where hole is located |
| `wind_direction` | int | -1,1 | Direction of wind |
| `gravity` | float | 0.1-2.0 | Strength of gravity |

## 🎬 **Save Videos for Analysis**

```python
from PIL import Image

def save_as_gif(frames, filename):
    pil_frames = [Image.fromarray(f.astype('uint8')) for f in frames]
    pil_frames[0].save(filename, save_all=True, append_images=pil_frames[1:], 
                      duration=100, loop=0)

# Save both videos
save_as_gif(result.baseline_result.frames, "before.gif")
save_as_gif(result.intervention_result.frames, "after.gif")
```

## 🤖 **Typical AI Workflow**

```python
# 1. AI perceives current video parameters
current_params = ai_model.perceive_video_parameters(video_data)

# 2. AI decides on intervention strategy
if ai_model.detects_traffic_jam(video_data):
    intervention = ("hole_diameter", current_params["hole_diameter"] * 2)
elif ai_model.detects_too_easy(video_data):
    intervention = ("wind_strength", 5.0)

# 3. AI generates intervention video
result = generate_intervention_video(
    baseline_parameters=current_params,
    intervention_target=intervention[0],
    intervention_value=intervention[1]
)

# 4. AI analyzes results for learning
if result.success:
    outcome_changed = (result.baseline_result.metadata.get('actual_jam_type') !=
                      result.intervention_result.metadata.get('actual_jam_type'))
    ai_model.update_knowledge(intervention, outcome_changed)
```

## 📚 **Examples**

### Example 1: Reduce Traffic Jams
```python
# AI sees small hole causing jams
result = generate_intervention_video(
    baseline_parameters={"hole_diameter": 25, "num_circles": 15, "width": 400, "height": 400, "num_frames": 80},
    intervention_target="hole_diameter", 
    intervention_value=80
)
# Likely result: partial_jam → no_jam
```

### Example 2: Add Challenge  
```python
# AI sees easy scenario, wants to add difficulty
result = generate_intervention_video(
    baseline_parameters={"wind_strength": 0.0, "num_circles": 5, "width": 400, "height": 400, "num_frames": 80},
    intervention_target="wind_strength",
    intervention_value=6.0
)
# Likely result: no_jam → full_jam
```

### Example 3: Test Circle Size Effect
```python
# AI wants to test if smaller circles flow better
result = generate_intervention_video(
    baseline_parameters={"circle_size_max": 15, "hole_diameter": 40, "width": 400, "height": 400, "num_frames": 80},
    intervention_target="circle_size_max",
    intervention_value=8
)
# Will show size effect on flow
```

## 🧪 **Test Scripts**

- `test_interface.py` - Simple single intervention test
- `test_ai_comprehensive.py` - Multiple intervention scenarios

Run them:
```bash
cd elvis_env/scripts
python test_interface.py
```

## 💡 **AI Tips**

1. **Always specify all required parameters**: `width`, `height`, `num_frames`
2. **Use consistent seeds**: For reproducible experiments
3. **Check `result.success`**: Before processing results
4. **Save videos**: For visual analysis and debugging
5. **Compare outcomes**: Use `actual_jam_type` in metadata
6. **Learn from failures**: Even unsuccessful interventions provide information

## 🎯 **Perfect for AI Research**

This interface enables AI models to:
- **Learn causal relationships** by testing interventions
- **Develop intervention strategies** through trial and error
- **Build physics intuition** about object flow dynamics
- **Practice counterfactual reasoning** with visual feedback
- **Validate hypotheses** about parameter effects

**The AI just provides the parameters and intervention - the interface handles all the complexity!** 🚀