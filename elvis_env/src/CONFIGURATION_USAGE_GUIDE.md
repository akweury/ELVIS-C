# Configuration-Based Video Generation

## Overview

This system enables **completely configuration-driven video generation** where all video parameters, object behaviors, colors, and interventions are defined in separate YAML files. No code changes are needed to create different types of videos.

## 📁 Configuration Files

All configurations are stored in separate YAML files in the `configs/` directory:

### Base Configurations
- **`two_parts_default.yaml`** - Default two-parts environment settings
- **`quick_test.yaml`** - Small, fast videos for testing
- **`high_resolution.yaml`** - High-quality, large videos

### Color Configurations  
- **`red_blue_colors.yaml`** - Fixed red and blue object colors
- **`rainbow_colors.yaml`** - Rainbow spectrum colors
- **`horizontal_movement.yaml`** - Green and orange with horizontal movement

### Intervention Configurations
- **`reverse_movement.yaml`** - Reverse movement intervention
- **`multi_intervention.yaml`** - Multiple objects with interventions  
- **`freeze_intervention.yaml`** - Freeze specific objects

### Advanced Configurations
- **`variable_speeds.yaml`** - Different speeds for different objects
- **`diagonal_movement.yaml`** - Diagonal movement patterns

## 🚀 Quick Usage

### Generate from Single Configuration

```bash
# Generate 20 videos with red-blue colors
python configurable_dataset_generator.py --config configs/red_blue_colors.yaml --num_videos 20

# Generate videos with reverse movement intervention
python configurable_dataset_generator.py --config configs/reverse_movement.yaml --num_videos 15

# Generate high-resolution videos
python configurable_dataset_generator.py --config configs/high_resolution.yaml --num_videos 10
```

### Configuration Management

```bash
# List all available configurations
python configuration_manager.py --list

# Validate a specific configuration
python configuration_manager.py --validate red_blue_colors

# Generate from specific configuration
python configuration_manager.py --generate rainbow_colors --num-videos 25

# Generate comparison dataset from ALL configurations
python configuration_manager.py --comparison --num-videos 10

# Generate from all configurations
python configuration_manager.py --generate-all --num-videos 5
```

## 📝 Configuration File Structure

### Basic Configuration Template

```yaml
video:
  width: 400          # Video width in pixels
  height: 300         # Video height in pixels  
  num_frames: 60      # Number of frames
  fps: 30            # Frames per second

objects:
  left_objects: 5     # Number of objects on left side
  right_objects: 5    # Number of objects on right side
  radius: 15         # Object radius in pixels
  velocity: 2        # Base velocity

colors:
  # Colors that cycle through objects (BGR format)
  default_colors:
    - [100, 150, 255]  # Light orange
    - [255, 150, 100]  # Light blue
    - [150, 255, 100]  # Light green
  
  # Fixed colors for specific objects
  fixed_colors:
    left_1: [255, 0, 0]    # First left object = red
    right_1: [0, 0, 255]   # First right object = blue
  
  background: [0, 0, 0]    # Background color

movement:
  patterns:
    left:
      direction: down        # down, up, left, right, down_right, up_left, etc.
      velocity: 2
    right:
      direction: up
      velocity: 2

interventions:
  enabled: true              # Enable/disable interventions
  type: reverse_movement     # reverse_movement, freeze, change_color
  target_indices:
    left: [0, 2]            # Apply to objects at indices 0 and 2
    right: [1]              # Apply to object at index 1

# Optional: Object-specific speed overrides  
object_speeds:
  left_1: 4                 # First left object moves at speed 4
  right_2: 1                # Second right object moves at speed 1

# Optional: Rendering settings
rendering:
  show_labels: true         # Show directional labels
  antialias: true          # Enable antialiasing
  dividing_line: true      # Show center dividing line
```

## 🎨 Common Modifications

### Fix Object Colors

Create a config file with fixed colors:

```yaml
colors:
  fixed_colors:
    # All left objects red
    left_1: [255, 0, 0]
    left_2: [255, 0, 0]
    left_3: [255, 0, 0]
    
    # All right objects blue  
    right_1: [0, 0, 255]
    right_2: [0, 0, 255]
    right_3: [0, 0, 255]
```

### Change Movement Directions

```yaml
movement:
  patterns:
    left:
      direction: right      # Move right instead of down
      velocity: 3
    right:
      direction: left       # Move left instead of up
      velocity: 2
```

### Apply Interventions

```yaml
interventions:
  enabled: true
  type: reverse_movement
  target_indices:
    left: [0, 1]           # First two left objects reverse
    right: [0]             # First right object reverses
```

### Variable Object Speeds

```yaml
object_speeds:
  left_1: 4              # Fast red objects
  left_3: 4
  left_2: 1              # Slow green objects  
  left_4: 1
  right_1: 2             # Medium blue objects
  right_2: 2
```

### Diagonal Movement

```yaml
movement:
  patterns:
    left:
      direction: down_right  # Diagonal movement
      velocity: 2.5
    right: 
      direction: up_left     # Opposite diagonal
      velocity: 2.5

rendering:
  dividing_line: false     # No center line for diagonal movement
```

## 🔧 Creating New Configurations

### Method 1: Copy and Modify Existing

```bash
# Copy an existing configuration
cp configs/red_blue_colors.yaml configs/my_custom.yaml

# Edit the new file with your changes
nano configs/my_custom.yaml
```

### Method 2: Generate Template

```bash
# Create a template configuration
python configuration_manager.py --template

# Edit the generated template  
nano configs/template.yaml
```

### Method 3: Programmatic Creation

```python
from rule_system.config_system import VideoConfig

# Create custom configuration
config_dict = {
    'video': {'width': 600, 'height': 400},
    'colors': {
        'fixed_colors': {
            'left_1': [0, 255, 0],   # Green
            'right_1': [255, 165, 0] # Orange  
        }
    },
    'interventions': {
        'enabled': True,
        'type': 'freeze',
        'target_indices': {'left': [0], 'right': [1]}
    }
}

# Save as YAML file
config = VideoConfig(config_dict=config_dict)
config.save('configs/my_programmatic_config.yaml')
```

## 📊 Dataset Generation Workflows

### Single Configuration Dataset

```bash
# Generate dataset with specific config
python configurable_dataset_generator.py \
    --config configs/rainbow_colors.yaml \
    --num_videos 50 \
    --output data/rainbow_dataset
```

### Multi-Configuration Comparison

```bash
# Generate comparison dataset with all configurations
python configuration_manager.py --comparison --num-videos 20
```

This creates a structured dataset:
```
data/comparison_20241215_143022/
├── red_blue_colors/
├── rainbow_colors/ 
├── reverse_movement/
├── freeze_intervention/
└── comparison_info.json
```

### Configuration Validation

```bash
# Check if configuration is valid before generation
python configuration_manager.py --validate my_custom_config

# List all available configurations
python configuration_manager.py --list
```

## 🎯 Benefits

| Task | Before | After |
|------|--------|-------|
| Change colors | Edit Python code | Edit YAML file |
| Add intervention | Modify environment class | Update config file |
| Create new scenario | Write new Python class | Copy & modify config |
| Test variations | Multiple code versions | Multiple config files |
| Batch generation | Complex scripts | Single command |

## 🔍 Advanced Features

### Intervention Types
- **`reverse_movement`** - Objects move in opposite direction
- **`freeze`** - Objects stop moving completely  
- **`change_color`** - Objects change to specified color

### Movement Directions
- **Basic**: `down`, `up`, `left`, `right`
- **Diagonal**: `down_right`, `down_left`, `up_right`, `up_left`
- **Custom**: Define custom movement functions

### Color Modes
- **Default cycling** - Colors cycle through objects automatically
- **Fixed assignment** - Specific colors for specific objects
- **Mixed mode** - Some fixed, some cycling

This configuration-driven approach makes video generation extremely flexible and maintainable - all without touching a single line of code!