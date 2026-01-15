# Intervention Video Generation - Two Parts Dataset

## Overview

The Two Parts dataset generator now supports creating intervention videos alongside observation videos. In intervention videos, some objects exhibit reversed movement patterns compared to the normal behavior.

## Normal vs Intervention Behavior

### Normal (Observation) Videos
- **Left side objects**: Move DOWN continuously  
- **Right side objects**: Move UP continuously
- All objects follow consistent directional movement

### Intervention Videos  
- **Some left objects**: Move UP (reversed)
- **Some right objects**: Move DOWN (reversed)
- 1-2 objects per side are randomly selected for intervention
- Intervention objects are visually highlighted with white borders

## Dataset Structure

```
dataset_root/
├── observation/           # Normal videos
│   └── observation_XXXXXX_YYYYYY/
│       ├── frames/        # PNG frames
│       ├── meta.json      # Metadata
│       └── stats.csv      # Frame statistics
├── intervention/          # Intervention videos  
│   └── intervention_XXXXXX_YYYYYY/
│       ├── frames/        # PNG frames
│       ├── meta.json      # Metadata (includes intervention info)
│       └── stats.csv      # Frame statistics
└── visualization/         # GIF previews
    ├── observation_*.gif  # Normal video previews
    └── intervention_*.gif # Intervention video previews
```

## Metadata Format

### Intervention Video Metadata

```json
{
  "sample_id": "intervention_000001_123456",
  "sample_type": "intervention",
  "params": {
    "left_objects": 3,
    "right_objects": 4,
    "is_intervention": true,
    "left_intervention_indices": [0, 1],    // Objects 0,1 on left move UP
    "right_intervention_indices": [2],      // Object 2 on right moves DOWN  
    "intervention_type": "reversed_movement"
  },
  "physics_simulation": {
    "is_intervention": true,
    "intervention_info": {
      "left_intervention_indices": [0, 1],
      "right_intervention_indices": [2], 
      "intervention_type": "reversed_movement"
    },
    "entity_mapping": {
      "entities": {
        "left_1": {
          "has_intervention": true    // This object has reversed movement
        },
        "left_2": {
          "has_intervention": true
        },
        "right_3": {
          "has_intervention": true
        }
      }
    }
  }
}
```

## Usage

### Command Line

```bash
# Generate dataset with intervention videos
python elvis_env/src/generate_two_parts_dataset.py \
  --num_videos 100 \
  --num_intervention_videos 50 \
  --output data/two_parts_intervention

# Generate only observation videos (original behavior)
python elvis_env/src/generate_two_parts_dataset.py \
  --num_videos 100 \
  --output data/two_parts_observation
```

### Programmatic Usage

```python
from generate_two_parts_dataset import TwoPartsDatasetGenerator

# Create generator
generator = TwoPartsDatasetGenerator("data/my_dataset")

# Generate mixed dataset
generator.generate_dataset(
    num_videos=100,              # Observation videos
    num_intervention_videos=50,   # Intervention videos  
    workers=4,
    create_gifs=True
)
```

## Visual Indicators

### In Video Frames
- **Normal objects**: Standard colored circles
- **Intervention objects**: Colored circles with white borders
- **Labels**: Show "INTERVENTION" indicator when present
- **Direction labels**: Show "UP*" or "DOWN*" for sides with interventions

### In GIF Previews
- Clear visual distinction between normal and intervention objects
- Labels help identify movement patterns
- Useful for human inspection and debugging

## Applications

### Causal Inference Research
- Study how directional changes affect system dynamics
- Compare counterfactual scenarios (normal vs intervention)
- Analyze causal relationships between movement and outcomes

### AI Model Training
- Train models to detect intervention patterns
- Learn to predict consequences of directional changes
- Develop understanding of object movement causality

### Dataset Analysis
- Compare statistics between observation and intervention videos
- Study parameter independence under different conditions
- Validate causal effect detection methods

## Technical Details

### Intervention Selection
- Randomly selects 1-2 objects per side for intervention
- Uses `random.sample()` to avoid object index overlap
- Stores intervention indices in metadata for reproducibility

### Movement Implementation  
- Modified `TwoPartsEnv.step()` method checks intervention flags
- Reverses velocity direction for flagged objects
- Maintains physics consistency (boundary wrapping, positioning)

### Quality Assurance
- Intervention videos undergo same quality checks as observation videos
- Metadata validation ensures intervention parameters are correctly saved
- Visual indicators help verify intervention application

## Configuration

The intervention system uses the same configuration as normal videos, with additional intervention-specific parameters generated automatically during sampling.

```yaml
parameter_ranges:
  left_objects: {min: 2, max: 8}
  right_objects: {min: 2, max: 8}
  velocity: {min: 1.0, max: 4.0}
  object_radius: {min: 8, max: 20}

# Intervention parameters are auto-generated:
# - left_intervention_indices: [0-1] random indices  
# - right_intervention_indices: [0-1] random indices
# - intervention_type: "reversed_movement"
```

This enhancement provides a powerful tool for generating controlled intervention datasets while maintaining compatibility with existing observation video workflows.