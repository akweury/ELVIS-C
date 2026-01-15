#!/usr/bin/env python3
"""
Configuration-driven video generation
Allows easy modification of video generation rules through configuration files
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from .base_rules import RuleEngine, ColorRule, MovementRule, InterventionRule, SpeedRule, PlacementRule


class VideoConfig:
    """Configuration class for video generation"""
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict] = None):
        """
        Initialize configuration from file or dictionary
        
        Args:
            config_path: Path to YAML configuration file
            config_dict: Configuration dictionary (overrides file)
        """
        if config_dict:
            self.config = config_dict
        elif config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Use default configuration file
            default_config_path = Path(__file__).parent.parent.parent / "configs" / "two_parts_default.yaml"
            if default_config_path.exists():
                with open(default_config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            else:
                self.config = self._get_fallback_config()
    
    def _get_fallback_config(self) -> Dict:
        """Get minimal fallback configuration when no config file is found"""
        print("Warning: No configuration file found, using minimal fallback config")
        return {
            'video': {'width': 400, 'height': 300, 'num_frames': 60, 'fps': 30},
            'objects': {'left_objects': 5, 'right_objects': 5, 'radius': 15, 'velocity': 2},
            'colors': {
                'default_colors': [[100, 150, 255], [255, 150, 100], [150, 255, 100]],
                'fixed_colors': {}, 'background': [0, 0, 0]
            },
            'movement': {
                'patterns': {'left': {'direction': 'down', 'velocity': 2}, 'right': {'direction': 'up', 'velocity': 2}}
            },
            'interventions': {'enabled': False, 'type': 'reverse_movement', 'target_indices': {'left': [], 'right': []}}
        }
    
    def get_rule_engine(self) -> RuleEngine:
        """Create and configure rule engine based on configuration"""
        engine = RuleEngine()
        
        # Add color rule
        color_config = self.config.get('colors', {})
        default_colors = [tuple(c) for c in color_config.get('default_colors', [])]
        fixed_colors = {k: tuple(v) for k, v in color_config.get('fixed_colors', {}).items()}
        
        color_rule = ColorRule(
            color_mapping=fixed_colors,
            default_colors=default_colors if default_colors else None
        )
        engine.add_rule(color_rule)
        
        # Add movement rule
        movement_config = self.config.get('movement', {})
        movement_rule = MovementRule(
            movement_patterns=movement_config.get('patterns', {})
        )
        engine.add_rule(movement_rule)
        
        # Add speed rule for object-specific speeds
        speed_overrides = self.get_object_speeds()
        if speed_overrides:
            speed_rule = SpeedRule(speed_overrides)
            engine.add_rule(speed_rule)
        
        # Add placement rule for cross-placement scenarios
        placement_config = self.config.get('placement', {})
        if placement_config.get('enabled', False):
            placement_rule = PlacementRule(placement_config)
            engine.add_rule(placement_rule)
        
        # Add intervention rule
        intervention_config = self.config.get('interventions', {})
        intervention_rule = InterventionRule(intervention_config)
        engine.add_rule(intervention_rule)
        
        return engine
    
    def get_video_params(self) -> Dict:
        """Get video parameters"""
        return self.config.get('video', {})
    
    def get_object_params(self) -> Dict:
        """Get object parameters"""
        return self.config.get('objects', {})
    
    def get_object_speeds(self) -> Dict:
        """Get object-specific speed overrides"""
        return self.config.get('object_speeds', {})
    
    def get_rendering_params(self) -> Dict:
        """Get rendering parameters"""
        return self.config.get('rendering', {})
    
    def get_physics_params(self) -> Dict:
        """Get physics parameters"""
        return self.config.get('physics', {})
    
    def get_observation_variants(self) -> Dict:
        """Get observation variant configurations"""
        return self.config.get('observation_variants', {})
    
    def get_placement_config(self) -> Dict:
        """Get placement configuration"""
        return self.config.get('placement', {})
    
    def create_variant_config(self, variant_name: str) -> 'VideoConfig':
        """Create a new config with specific observation variant applied"""
        variants = self.get_observation_variants()
        if variant_name not in variants:
            return self
        
        variant_config = variants[variant_name]
        new_config = self.config.copy()
        
        # Deep merge variant config into base config
        def deep_merge(base, update):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
        
        deep_merge(new_config, variant_config)
        return VideoConfig(config_dict=new_config)
    
    def save(self, output_path: str):
        """Save configuration to file"""
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)


class ConfigurableVideoGenerator:
    """Video generator that uses configuration-driven rules"""
    
    def __init__(self, config: VideoConfig):
        self.config = config
        self.rule_engine = config.get_rule_engine()
    
    def generate_video_params(self) -> Dict:
        """Generate video parameters based on configuration"""
        video_params = self.config.get_video_params()
        object_params = self.config.get_object_params()
        
        return {
            **video_params,
            **object_params
        }
    
    def create_sample_configs(self):
        """Create sample configuration files for different scenarios"""
        
        # 1. Fixed red-blue colors
        red_blue_config = VideoConfig().config.copy()
        red_blue_config['colors']['fixed_colors'] = {
            'left_1': [255, 0, 0], 'left_2': [255, 0, 0], 'left_3': [255, 0, 0],
            'right_1': [0, 0, 255], 'right_2': [0, 0, 255], 'right_3': [0, 0, 255]
        }
        
        # 2. Reverse movement intervention
        reverse_config = VideoConfig().config.copy()
        reverse_config['interventions'] = {
            'enabled': True,
            'type': 'reverse_movement',
            'target_indices': {'left': [0, 1], 'right': [0]}
        }
        
        # 3. Custom movement patterns
        custom_movement_config = VideoConfig().config.copy()
        custom_movement_config['movement']['patterns'] = {
            'left': {'direction': 'right', 'velocity': 3},
            'right': {'direction': 'left', 'velocity': 2}
        }
        
        # 4. Freeze intervention
        freeze_config = VideoConfig().config.copy()
        freeze_config['interventions'] = {
            'enabled': True,
            'type': 'freeze',
            'target_indices': {'left': [1], 'right': [2]}
        }
        
        # Save sample configs
        configs_dir = Path("configs/samples")
        configs_dir.mkdir(parents=True, exist_ok=True)
        
        VideoConfig(config_dict=red_blue_config).save(configs_dir / "red_blue_colors.yaml")
        VideoConfig(config_dict=reverse_config).save(configs_dir / "reverse_movement.yaml")
        VideoConfig(config_dict=custom_movement_config).save(configs_dir / "custom_movement.yaml")
        VideoConfig(config_dict=freeze_config).save(configs_dir / "freeze_objects.yaml")
        
        print(f"Sample configurations saved to {configs_dir}")
        return configs_dir


# Example usage functions
def create_red_blue_config() -> VideoConfig:
    """Create configuration with fixed red and blue colors"""
    config_dict = {
        'colors': {
            'fixed_colors': {
                'left_1': [255, 0, 0], 'left_2': [255, 0, 0], 'left_3': [255, 0, 0],
                'right_1': [0, 0, 255], 'right_2': [0, 0, 255], 'right_3': [0, 0, 255]
            }
        }
    }
    base_config = VideoConfig().config
    base_config.update(config_dict)
    return VideoConfig(config_dict=base_config)


def create_reverse_movement_config() -> VideoConfig:
    """Create configuration with reverse movement intervention"""
    config_dict = {
        'interventions': {
            'enabled': True,
            'type': 'reverse_movement',
            'target_indices': {'left': [0, 1], 'right': [0]}
        }
    }
    base_config = VideoConfig().config
    base_config.update(config_dict)
    return VideoConfig(config_dict=base_config)