#!/usr/bin/env python3
"""
Configuration Manager for Video Generation
Manages multiple configuration files and generates datasets based on them
"""

import os
import yaml
import json
import argparse
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from configurable_dataset_generator import ConfigurableDatasetGenerator


class ConfigurationManager:
    """Manage multiple configurations and dataset generation"""
    
    def __init__(self, config_dir: str = "configs"):
        # Resolve config directory relative to this file's location
        if not Path(config_dir).is_absolute():
            # If relative path, resolve from parent directory (elvis_env)
            base_dir = Path(__file__).parent.parent  # Go up to elvis_env/
            self.config_dir = base_dir / config_dir
        else:
            self.config_dir = Path(config_dir)
        self.available_configs = self._discover_configs()
    
    def _discover_configs(self) -> Dict[str, Path]:
        """Discover all available configuration files"""
        configs = {}
        if self.config_dir.exists():
            for config_file in self.config_dir.glob("*.yaml"):
                if config_file.name != "default.yaml":  # Skip the original default
                    config_name = config_file.stem
                    configs[config_name] = config_file
        return configs
    
    def list_configurations(self):
        """List all available configurations with descriptions"""
        print("📋 Available Configurations:")
        print("=" * 50)
        
        for name, path in self.available_configs.items():
            try:
                with open(path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Extract key info for display
                video_info = config.get('video', {})
                objects_info = config.get('objects', {})
                interventions_info = config.get('interventions', {})
                
                print(f"\n🎬 {name}")
                print(f"   File: {path}")
                print(f"   Resolution: {video_info.get('width', '?')}x{video_info.get('height', '?')}")
                print(f"   Objects: {objects_info.get('left_objects', '?')} left, {objects_info.get('right_objects', '?')} right")
                print(f"   Frames: {video_info.get('num_frames', '?')}")
                
                if interventions_info.get('enabled'):
                    print(f"   Intervention: {interventions_info.get('type', 'unknown')}")
                else:
                    print(f"   Intervention: None")
                
                # Show special features
                if 'fixed_colors' in config.get('colors', {}) and config['colors']['fixed_colors']:
                    print(f"   ✨ Fixed colors defined")
                if 'object_speeds' in config:
                    print(f"   ⚡ Variable speeds defined")
                
            except Exception as e:
                print(f"\n❌ {name}: Error reading config - {e}")
    
    def validate_configuration(self, config_name: str) -> bool:
        """Validate a configuration file"""
        if config_name not in self.available_configs:
            print(f"❌ Configuration '{config_name}' not found")
            return False
        
        config_path = self.available_configs[config_name]
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Basic validation
            required_sections = ['video', 'objects', 'colors', 'movement']
            for section in required_sections:
                if section not in config:
                    print(f"❌ Missing required section: {section}")
                    return False
            
            # Validate video section
            video = config['video']
            if not all(key in video for key in ['width', 'height', 'num_frames', 'fps']):
                print(f"❌ Incomplete video configuration")
                return False
            
            # Validate objects section
            objects = config['objects']
            if not all(key in objects for key in ['left_objects', 'right_objects']):
                print(f"❌ Incomplete objects configuration")
                return False
            
            print(f"✅ Configuration '{config_name}' is valid")
            return True
            
        except Exception as e:
            print(f"❌ Error validating '{config_name}': {e}")
            return False
    
    def generate_from_config(self, config_name: str, num_videos: int = 10, output_dir: str = None):
        """Generate dataset from a specific configuration"""
        if not self.validate_configuration(config_name):
            return False
        
        config_path = str(self.available_configs[config_name])
        output_dir = output_dir or f"data/{config_name}_dataset"
        
        print(f"🎬 Generating dataset from '{config_name}' configuration")
        print(f"   Config: {config_path}")
        print(f"   Output: {output_dir}")
        print(f"   Videos: {num_videos}")
        
        generator = ConfigurableDatasetGenerator(
            config_path=config_path,
            output_dir=output_dir,
            clean_old=True
        )
        
        results = []
        from rule_system.config_system import VideoConfig
        config = VideoConfig(config_path)
        
        for i in range(num_videos):
            result = generator.generate_with_config(i, config, 'observation')
            results.append(result)
            if result['status'] == 'success':
                print(f"   ✅ Generated {result['sample_id']}")
            else:
                print(f"   ❌ Failed {result.get('sample_id', 'unknown')}: {result.get('error', 'unknown')}")
        
        successful = [r for r in results if r['status'] == 'success']
        print(f"\n📊 Generation complete: {len(successful)}/{num_videos} successful")
        return True
    
    def generate_observation_variants(self, config_name: str, num_videos_per_variant: int = 5):
        """Generate multiple observation variants for a single configuration"""
        if not self.validate_configuration(config_name):
            return False
        
        config_path = str(self.available_configs[config_name])
        from rule_system.config_system import VideoConfig
        base_config = VideoConfig(config_path)
        
        variants = base_config.get_observation_variants()
        
        if not variants:
            print(f"⚠️  No observation variants defined in '{config_name}' configuration")
            return self.generate_from_config(config_name, num_videos_per_variant)
        
        print(f"🎭 Generating observation variants for '{config_name}' configuration")
        print(f"   Variants: {list(variants.keys())}")
        print(f"   Videos per variant: {num_videos_per_variant}")
        
        # Create main data folder and handle clean_old only once
        base_output_dir = f"data/{config_name}"
        
        # Resolve to absolute path
        if not Path(base_output_dir).is_absolute():
            # Resolve relative to the current working directory
            base_path = Path.cwd() / base_output_dir
        else:
            base_path = Path(base_output_dir)
        
        # Remove old data if it exists (only once)
        if base_path.exists():
            print(f"🗑️  Removing old data from {base_path}")
            import shutil
            shutil.rmtree(base_path)
        
        # Create base directory structure
        base_path.mkdir(parents=True, exist_ok=True)
        base_dirs = {
            'observation': base_path / 'observation',
            'intervention': base_path / 'intervention',
            'visualization': base_path / 'visualization',
            'configs': base_path / 'configs'
        }
        for dir_path in base_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        
        all_results = []
        
        for variant_name in variants.keys():
            print(f"\n🎬 Generating variant: {variant_name}")
            
            # Create variant-specific config
            variant_config = base_config.create_variant_config(variant_name)
            
            # Create output directory for this variant
            variant_output_dir = str(base_dirs['observation'] / variant_name)
            
            generator = ConfigurableDatasetGenerator(
                output_dir=variant_output_dir,
                clean_old=False  # Don't clean individual variant folders
            )
            
            variant_results = []
            for i in range(num_videos_per_variant):
                result = generator.generate_with_config(i, variant_config, variant_name)
                variant_results.append(result)
                if result['status'] == 'success':
                    print(f"   ✅ Generated {result['sample_id']}")
                else:
                    print(f"   ❌ Failed {result.get('sample_id', 'unknown')}: {result.get('error', 'unknown')}")
            
            all_results.extend(variant_results)
        
        successful = [r for r in all_results if r['status'] == 'success']
        total_expected = len(variants) * num_videos_per_variant
        print(f"\n📊 All variants complete: {len(successful)}/{total_expected} successful")
        print(f"   Generated {len(variants)} observation variants")
        return True
    
    def generate_from_multiple_configs(self, config_names: List[str], num_videos_each: int = 5):
        """Generate datasets from multiple configurations"""
        print(f"🎭 Generating from {len(config_names)} configurations")
        
        for config_name in config_names:
            print(f"\n{'='*60}")
            success = self.generate_from_config(
                config_name, 
                num_videos_each, 
                f"data/multi_config/{config_name}"
            )
            if not success:
                print(f"⚠️  Skipping '{config_name}' due to errors")
    
    def generate_comparison_dataset(self, num_videos: int = 5):
        """Generate a comparison dataset using all available configurations"""
        print(f"🔬 Generating comparison dataset with all configurations")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = f"data/comparison_{timestamp}"
        
        comparison_info = {
            'generated_at': datetime.now().isoformat(),
            'configurations_used': [],
            'videos_per_config': num_videos,
            'total_videos': 0
        }
        
        total_successful = 0
        
        for config_name in self.available_configs.keys():
            print(f"\n🎬 Processing {config_name}...")
            output_dir = f"{base_output_dir}/{config_name}"
            
            success = self.generate_from_config(config_name, num_videos, output_dir)
            if success:
                comparison_info['configurations_used'].append(config_name)
                total_successful += num_videos
        
        comparison_info['total_videos'] = total_successful
        
        # Save comparison metadata
        os.makedirs(base_output_dir, exist_ok=True)
        with open(f"{base_output_dir}/comparison_info.json", 'w') as f:
            json.dump(comparison_info, f, indent=2)
        
        print(f"\n🎉 Comparison dataset complete!")
        print(f"   Location: {base_output_dir}")
        print(f"   Configurations: {len(comparison_info['configurations_used'])}")
        print(f"   Total videos: {total_successful}")
    
    def create_config_template(self, output_path: str = "configs/template.yaml"):
        """Create a template configuration file"""
        template = {
            'video': {
                'width': 400,
                'height': 300,
                'num_frames': 60,
                'fps': 30
            },
            'objects': {
                'left_objects': 5,
                'right_objects': 5,
                'radius': 15,
                'velocity': 2
            },
            'colors': {
                'default_colors': [
                    [100, 150, 255], [255, 150, 100], [150, 255, 100]
                ],
                'fixed_colors': {
                    'left_1': [255, 0, 0],
                    'right_1': [0, 0, 255]
                },
                'background': [0, 0, 0]
            },
            'movement': {
                'patterns': {
                    'left': {'direction': 'down', 'velocity': 2},
                    'right': {'direction': 'up', 'velocity': 2}
                }
            },
            'interventions': {
                'enabled': False,
                'type': 'reverse_movement',
                'target_indices': {'left': [], 'right': []}
            },
            'object_speeds': {
                'left_1': 4,
                'right_1': 1
            },
            'rendering': {
                'show_labels': True,
                'antialias': True,
                'dividing_line': True
            }
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(template, f, default_flow_style=False, indent=2)
        
        print(f"📝 Template configuration created: {output_path}")


def main():
    """Main configuration management interface"""
    parser = argparse.ArgumentParser(description='Configuration Manager for Video Generation')
    parser.add_argument('--list', action='store_true', help='List all available configurations')
    parser.add_argument('--validate', type=str, help='Validate a specific configuration')
    parser.add_argument('--generate', type=str, help='Generate dataset from specific configuration')
    parser.add_argument('--generate-variants', type=str, help='Generate observation variants for specific configuration')
    parser.add_argument('--generate-all', action='store_true', help='Generate from all configurations')
    parser.add_argument('--comparison', action='store_true', help='Generate comparison dataset')
    parser.add_argument('--template', action='store_true', help='Create configuration template')
    parser.add_argument('--num-videos', type=int, default=10, help='Number of videos to generate')
    parser.add_argument('--config-dir', type=str, default='configs', help='Configuration directory')
    
    args = parser.parse_args()
    
    manager = ConfigurationManager(config_dir=args.config_dir)
    
    if args.list:
        manager.list_configurations()
    elif args.validate:
        manager.validate_configuration(args.validate)
    elif args.generate:
        manager.generate_from_config(args.generate, args.num_videos)
    elif args.generate_variants:
        manager.generate_observation_variants(args.generate_variants, args.num_videos)
    elif args.generate_all:
        config_names = list(manager.available_configs.keys())
        manager.generate_from_multiple_configs(config_names, args.num_videos)
    elif args.comparison:
        manager.generate_comparison_dataset(args.num_videos)
    elif args.template:
        manager.create_config_template()
    else:
        print("🎛️  Configuration Manager")
        print("Use --help to see available options")
        print("\nQuick commands:")
        print("  --list                        List all configurations")
        print("  --generate <config_name>      Generate from specific config")
        print("  --generate-variants <config>  Generate observation variants")
        print("  --comparison                  Generate comparison dataset")


if __name__ == "__main__":
    main()