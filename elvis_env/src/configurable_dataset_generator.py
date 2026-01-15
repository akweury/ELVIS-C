#!/usr/bin/env python3
"""
Configuration-driven dataset generator
Uses rule-based system for easy video generation modifications
"""

import os
import json
import random
import argparse
import cv2
import shutil
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Dict, Optional, List
from PIL import Image

from rule_system.config_system import VideoConfig, ConfigurableVideoGenerator
from extensible_two_parts import ExtensibleTwoPartsEnv


class ConfigurableDatasetGenerator:
    """Generate datasets using configuration-driven rules"""
    
    def __init__(self, config_path: Optional[str] = None, output_dir: str = "data/configurable_two_parts", 
                 clean_old: bool = False):
        """
        Initialize generator with configuration
        
        Args:
            config_path: Path to YAML configuration file
            output_dir: Output directory for dataset
            clean_old: Whether to remove old data
        """
        self.config = VideoConfig(config_path) if config_path else VideoConfig()
        self.output_dir = Path(output_dir)
        self.generator = ConfigurableVideoGenerator(self.config)
        
    
    def generate_with_config(self, video_id: int, config: VideoConfig, sample_type: str = 'observation') -> Dict:
        """Generate single video using specific configuration"""
        sample_id = f"{sample_type}_{video_id:06d}_{random.randint(100000, 999999)}"
        
        try:
            # Create environment with configuration
            video_params = config.get_video_params()
            object_params = config.get_object_params()
            
            env = ExtensibleTwoPartsEnv(
                config=config,
                width=video_params.get('width', 400),
                height=video_params.get('height', 300),
                left_objects=object_params.get('left_objects', 5),
                right_objects=object_params.get('right_objects', 5),
                object_radius=object_params.get('radius', 15),
                velocity=object_params.get('velocity', 2)
            )
            
            # Generate video
            seed = random.randint(1, 1000000)
            num_frames = video_params.get('num_frames', 60)
            fps = video_params.get('fps', 30)
            
            frames, metadata = env.generate_video(
                num_frames=num_frames, 
                fps=fps, 
                seed=seed, 
                include_labels=False
            )
            
            if not frames:
                return {'sample_id': sample_id, 'status': 'failed', 'error': 'No frames generated'}
            
            # Save frames
            sample_dir = self.output_dir / sample_id
            frames_dir = sample_dir / 'frames'
            frames_dir.mkdir(parents=True, exist_ok=True)
            
            for i, frame in enumerate(frames):
                cv2.imwrite(str(frames_dir / f"{i:06d}.png"), frame)
            
            # Save metadata with configuration
            meta = {
                'sample_id': sample_id,
                'sample_type': sample_type,
                'configuration': config.config,
                'generation_info': {
                    'timestamp': datetime.now().isoformat(),
                    'seed': seed
                },
                'video_stats': {
                    'num_frames': len(frames),
                    'frame_size': list(frames[0].shape),
                    'fps': fps
                },
                'rule_metadata': metadata
            }
            
            with open(sample_dir / 'meta.json', 'w') as f:
                json.dump(meta, f, indent=2)
            
            # Save configuration file for this sample
            config.save(sample_dir / 'config.yaml')
            
            # Generate visualization GIF
            labeled_frames, _ = env.generate_video(
                num_frames=min(20, num_frames), 
                fps=fps, 
                seed=seed, 
                include_labels=True
            )
            
            if labeled_frames:
                gif_path = self.output_dir / f"{sample_id}.gif"
                pil_images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in labeled_frames]
                if pil_images:
                    pil_images[0].save(gif_path, save_all=True, append_images=pil_images[1:], 
                                     duration=200, loop=0)
            
            return {
                'sample_id': sample_id,
                'sample_type': sample_type,
                'status': 'success',
                'num_frames': len(frames),
                'config_applied': config.config
            }
            
        except Exception as e:
            return {'sample_id': sample_id, 'status': 'failed', 'error': str(e)}
    
    def generate_dataset_with_variations(self, base_config_path: Optional[str] = None, 
                                       num_videos: int = 10, variations: List[Dict] = None):
        """Generate dataset with configuration variations"""
        print(f"🎬 Generating Configurable Dataset with Variations")
        print(f"Base videos: {num_videos}, Output: {self.output_dir}")
        
        variations = variations or []
        results = []
        
        # Generate base configuration videos
        base_config = VideoConfig(base_config_path) if base_config_path else VideoConfig()
        
        for video_id in tqdm(range(num_videos), desc="Generating base videos"):
            result = self.generate_with_config(video_id, base_config, 'observation')
            results.append(result)
        
        # Generate videos with variations
        for variation_idx, variation in enumerate(variations):
            variation_config = VideoConfig(config_dict={**base_config.config, **variation})
            
            for video_id in tqdm(range(num_videos), desc=f"Generating variation {variation_idx + 1}"):
                sample_type = f"variation_{variation_idx + 1}"
                result = self.generate_with_config(video_id, variation_config, sample_type)
                results.append(result)
        
        # Save master configuration index
        config_index = {
            'base_config': base_config.config,
            'variations': variations,
            'generation_info': {
                'timestamp': datetime.now().isoformat(),
                'num_base_videos': num_videos,
                'num_variations': len(variations)
            }
        }
        
        with open(self.dirs['configs'] / 'index.json', 'w') as f:
            json.dump(config_index, f, indent=2)
        
        # Summary
        successful = [r for r in results if r['status'] == 'success']
        print(f"\n✅ Complete! Successful: {len(successful)}/{len(results)}")
        print(f"   Location: {self.output_dir}")
        
        return results


def create_sample_variations():
    """Create sample variation configurations"""
    return [
        # Red-blue color variation
        {
            'colors': {
                'fixed_colors': {
                    'left_1': [255, 0, 0], 'left_2': [255, 0, 0], 'left_3': [255, 0, 0],
                    'right_1': [0, 0, 255], 'right_2': [0, 0, 255], 'right_3': [0, 0, 255]
                }
            }
        },
        
        # Reverse movement intervention
        {
            'interventions': {
                'enabled': True,
                'type': 'reverse_movement',
                'target_indices': {'left': [0, 1], 'right': [0]}
            }
        },
        
        # Horizontal movement
        {
            'movement': {
                'patterns': {
                    'left': {'direction': 'right', 'velocity': 2},
                    'right': {'direction': 'left', 'velocity': 2}
                }
            }
        },
        
        # Different speeds
        {
            'objects': {
                'velocity': 4
            },
            'movement': {
                'patterns': {
                    'left': {'direction': 'down', 'velocity': 4},
                    'right': {'direction': 'up', 'velocity': 1}
                }
            }
        }
    ]


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Generate Configurable Two Parts Dataset')
    parser.add_argument('--config', type=str, 
                       help='Path to YAML configuration file')
    parser.add_argument('--num_videos', type=int, default=10,
                       help='Number of videos per configuration')
    parser.add_argument('--output', type=str, default='data/configurable_two_parts',
                       help='Output directory for dataset')
    parser.add_argument('--clean_old', action='store_true',
                       help='Remove old generated data before creating new dataset')
    parser.add_argument('--create_samples', action='store_true',
                       help='Create sample configuration files')
    parser.add_argument('--use_variations', action='store_true',
                       help='Generate videos with predefined variations')
    
    args = parser.parse_args()
    
    # Create sample configurations if requested
    if args.create_samples:
        generator = ConfigurableVideoGenerator(VideoConfig())
        configs_dir = generator.create_sample_configs()
        print(f"Sample configurations created in {configs_dir}")
        return
    
    # Create generator
    generator = ConfigurableDatasetGenerator(
        config_path=args.config,
        output_dir=args.output,
        clean_old=args.clean_old
    )
    
    if args.use_variations:
        # Generate dataset with variations
        variations = create_sample_variations()
        generator.generate_dataset_with_variations(
            base_config_path=args.config,
            num_videos=args.num_videos,
            variations=variations
        )
    else:
        # Generate single configuration dataset
        config = VideoConfig(args.config) if args.config else VideoConfig()
        for video_id in range(args.num_videos):
            result = generator.generate_with_config(video_id, config, 'observation')
            if result['status'] == 'success':
                print(f"✅ Generated {result['sample_id']}")
            else:
                print(f"❌ Failed {result.get('sample_id', 'unknown')}: {result.get('error', 'unknown error')}")


if __name__ == "__main__":
    main()