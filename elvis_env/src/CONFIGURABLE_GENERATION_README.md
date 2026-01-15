# Configurable Video Generation System

## Overview

This system provides a **rule-based, configuration-driven approach** to video generation that makes modifications like fixing object colors, changing movement directions, and applying interventions extremely easy.

## Key Benefits

- 🎨 **Easy Color Management**: Fix object colors through simple YAML configuration
- 🎯 **Flexible Movement Rules**: Change object directions and speeds without code changes  
- 🔧 **Intervention System**: Apply interventions like reverse movement, freezing objects, etc.
- 📝 **Configuration-Driven**: All modifications through YAML files or programmatic config
- 🧩 **Extensible Rules**: Easy to add new rule types for future needs

## Architecture

```
rule_system/
├── base_rules.py          # Core rule classes (ColorRule, MovementRule, etc.)
├── config_system.py       # Configuration management and VideoConfig
└── __init__.py

extensible_two_parts.py    # Enhanced environment using rule system
configurable_dataset_generator.py  # Dataset generator with config support
demo_configurable_generation.py   # Usage examples
```

## Quick Start

### 1. Basic Usage with Configuration File

```yaml
# configs/my_config.yaml
colors:
  fixed_colors:
    left_1: [255, 0, 0]    # Red
    left_2: [255, 0, 0] 
    right_1: [0, 0, 255]   # Blue
    right_2: [0, 0, 255]

movement:
  patterns:
    left:
      direction: down
      velocity: 2
    right:
      direction: up
      velocity: 3

interventions:
  enabled: true
  type: reverse_movement
  target_indices:
    left: [0]              # First left object reverses
    right: [1]             # Second right object reverses
```

```python
from configurable_dataset_generator import ConfigurableDatasetGenerator

# Generate videos using configuration
generator = ConfigurableDatasetGenerator(config_path="configs/my_config.yaml")
generator.generate_dataset_with_variations(num_videos=10)
```

### 2. Programmatic Configuration

```python
from rule_system.config_system import VideoConfig

# Create config programmatically
config_dict = {
    'colors': {
        'fixed_colors': {
            'left_1': [0, 255, 0],   # Green
            'right_1': [255, 165, 0] # Orange
        }
    },
    'movement': {
        'patterns': {
            'left': {'direction': 'right', 'velocity': 4},
            'right': {'direction': 'left', 'velocity': 2}
        }
    }
}

config = VideoConfig(config_dict=config_dict)
```

## Common Modifications Made Easy

### Fix Object Colors

```yaml
colors:
  fixed_colors:
    left_1: [255, 0, 0]    # Red
    left_2: [0, 255, 0]    # Green  
    left_3: [0, 0, 255]    # Blue
    right_1: [255, 255, 0] # Yellow
    # etc.
```

### Change Movement Directions

```yaml
movement:
  patterns:
    left:
      direction: right     # Instead of down
      velocity: 3
    right:
      direction: left      # Instead of up  
      velocity: 2
```

### Apply Interventions

```yaml
interventions:
  enabled: true
  type: reverse_movement   # or 'freeze' or 'change_color'
  target_indices:
    left: [0, 2]          # Objects at indices 0 and 2
    right: [1]            # Object at index 1
```

### Different Video Dimensions

```yaml
video:
  width: 600
  height: 400
  num_frames: 80
  fps: 30

objects:
  left_objects: 8
  right_objects: 6
  radius: 20
  velocity: 4
```

## Usage Examples

### Generate Fixed Red-Blue Videos

```bash
python configurable_dataset_generator.py --config configs/red_blue_colors.yaml --num_videos 20
```

### Generate with Reverse Movement Intervention

```bash
python configurable_dataset_generator.py --config configs/reverse_movement.yaml --num_videos 15
```

### Generate Multiple Variations

```python
# Generate base videos + variations automatically
generator = ConfigurableDatasetGenerator()
variations = [
    {'colors': {'fixed_colors': {'left_1': [255, 0, 0]}}},  # Red variation
    {'interventions': {'enabled': True, 'type': 'reverse_movement'}},  # Intervention
    {'movement': {'patterns': {'left': {'direction': 'right'}}}}  # Horizontal movement
]

generator.generate_dataset_with_variations(
    num_videos=10, 
    variations=variations
)
```

### Run Demo Scripts

```bash
# See all examples in action
python demo_configurable_generation.py
```

## Extending the System

### Add New Rule Types

```python
from rule_system.base_rules import BaseRule

class SizeRule(BaseRule):
    def __init__(self, size_mapping):
        super().__init__("size_rule", priority=75)
        self.size_mapping = size_mapping
    
    def apply(self, objects, frame_info):
        for obj in objects:
            entity_id = obj['entity_id']
            if entity_id in self.size_mapping:
                obj['radius'] = self.size_mapping[entity_id]
        return objects
```

### Add New Intervention Types

```python
# In InterventionRule._apply_intervention()
elif intervention_type == 'bounce':
    obj['bounce_mode'] = True
elif intervention_type == 'spiral':
    obj['spiral_mode'] = True
```

## File Structure

```
configs/
├── red_blue_colors.yaml      # Fixed red/blue colors
├── reverse_movement.yaml     # Reverse movement intervention
└── horizontal_movement.yaml  # Horizontal movement example

src/
├── rule_system/              # Core rule system
├── extensible_two_parts.py   # Enhanced environment
├── configurable_dataset_generator.py
└── demo_configurable_generation.py
```

## Benefits Over Original System

| Aspect | Original | New System |
|--------|----------|------------|
| Color Changes | Edit Python code | Edit YAML config |
| Movement Changes | Modify environment class | Change config file |
| Adding Interventions | Code in step() method | Add to intervention config |
| Creating Variations | Duplicate & modify code | Multiple config files |
| Testing New Rules | Restart environment | Hot-swap configurations |

This system makes video generation modifications as simple as editing a configuration file, enabling rapid experimentation and iteration without touching the core codebase.