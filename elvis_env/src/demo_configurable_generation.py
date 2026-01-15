#!/usr/bin/env python3
"""
Example usage of the configurable video generation system
Demonstrates how to easily create videos with different rules
"""

from rule_system.config_system import VideoConfig
from configurable_dataset_generator import ConfigurableDatasetGenerator
from extensible_two_parts import ExtensibleTwoPartsEnv


def demo_fixed_colors():
    """Demo: Generate videos with fixed red and blue colors"""
    print("🎥 Demo: Fixed Red-Blue Colors")
    
    config = VideoConfig("configs/red_blue_colors.yaml")
    generator = ConfigurableDatasetGenerator(config_path="configs/red_blue_colors.yaml", 
                                           output_dir="demo/red_blue_demo")
    
    # Generate 5 videos with fixed colors
    for i in range(5):
        result = generator.generate_with_config(i, config, 'red_blue')
        if result['status'] == 'success':
            print(f"✅ Generated {result['sample_id']}")


def demo_reverse_movement():
    """Demo: Generate videos with reverse movement intervention"""
    print("🎥 Demo: Reverse Movement Intervention")
    
    config = VideoConfig("configs/reverse_movement.yaml")
    generator = ConfigurableDatasetGenerator(config_path="configs/reverse_movement.yaml",
                                           output_dir="demo/reverse_demo")
    
    # Generate 3 videos with intervention
    for i in range(3):
        result = generator.generate_with_config(i, config, 'intervention')
        if result['status'] == 'success':
            print(f"✅ Generated {result['sample_id']}")


def demo_horizontal_movement():
    """Demo: Generate videos with horizontal movement"""
    print("🎥 Demo: Horizontal Movement")
    
    config = VideoConfig("configs/horizontal_movement.yaml")
    generator = ConfigurableDatasetGenerator(config_path="configs/horizontal_movement.yaml",
                                           output_dir="demo/horizontal_demo")
    
    # Generate 3 videos with horizontal movement
    for i in range(3):
        result = generator.generate_with_config(i, config, 'horizontal')
        if result['status'] == 'success':
            print(f"✅ Generated {result['sample_id']}")


def demo_custom_config():
    """Demo: Create custom configuration programmatically"""
    print("🎥 Demo: Custom Programmatic Configuration")
    
    # Create custom config with unique settings
    custom_config_dict = {
        'video': {'width': 600, 'height': 400, 'num_frames': 40},
        'objects': {'left_objects': 6, 'right_objects': 4, 'radius': 12, 'velocity': 1},
        'colors': {
            'fixed_colors': {
                'left_1': [255, 255, 0],    # Yellow
                'left_2': [255, 255, 0],
                'left_3': [255, 0, 255],    # Magenta
                'left_4': [255, 0, 255],
                'left_5': [0, 255, 255],    # Cyan
                'left_6': [0, 255, 255],
                'right_1': [128, 0, 128],   # Purple
                'right_2': [128, 0, 128],
                'right_3': [255, 69, 0],    # Red-orange
                'right_4': [255, 69, 0]
            }
        },
        'movement': {
            'patterns': {
                'left': {'direction': 'down', 'velocity': 1},
                'right': {'direction': 'up', 'velocity': 3}  # Faster right objects
            }
        },
        'interventions': {
            'enabled': True,
            'type': 'reverse_movement',
            'target_indices': {'left': [2, 4], 'right': [1]}  # Specific objects reverse
        }
    }
    
    config = VideoConfig(config_dict=custom_config_dict)
    generator = ConfigurableDatasetGenerator(output_dir="demo/custom_demo")
    
    # Generate 2 videos with custom config
    for i in range(2):
        result = generator.generate_with_config(i, config, 'custom')
        if result['status'] == 'success':
            print(f"✅ Generated {result['sample_id']}")


def demo_variations():
    """Demo: Generate dataset with multiple variations"""
    print("🎥 Demo: Dataset with Variations")
    
    generator = ConfigurableDatasetGenerator(output_dir="demo/variations_demo")
    
    # Define variations
    variations = [
        # Variation 1: Different speeds
        {
            'objects': {'velocity': 4},
            'movement': {
                'patterns': {
                    'left': {'direction': 'down', 'velocity': 4},
                    'right': {'direction': 'up', 'velocity': 1}
                }
            }
        },
        
        # Variation 2: Freeze some objects
        {
            'interventions': {
                'enabled': True,
                'type': 'freeze',
                'target_indices': {'left': [0], 'right': [1]}
            }
        },
        
        # Variation 3: Rainbow colors
        {
            'colors': {
                'fixed_colors': {
                    'left_1': [255, 0, 0],    # Red
                    'left_2': [255, 127, 0],  # Orange
                    'left_3': [255, 255, 0],  # Yellow
                    'right_1': [0, 255, 0],   # Green
                    'right_2': [0, 0, 255],   # Blue
                    'right_3': [148, 0, 211]  # Violet
                }
            }
        }
    ]
    
    # Generate dataset with variations
    generator.generate_dataset_with_variations(
        num_videos=2,
        variations=variations
    )


if __name__ == "__main__":
    print("🚀 Configuration-Based Video Generation Demos")
    print("=" * 50)
    
    # Run all demos
    demo_fixed_colors()
    print()
    
    demo_reverse_movement() 
    print()
    
    demo_horizontal_movement()
    print()
    
    demo_custom_config()
    print()
    
    demo_variations()
    
    print("\n✨ All demos complete! Check the demo/ folder for generated videos.")