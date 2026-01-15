#!/usr/bin/env python3
"""
Two Parts Dataset Generator - Streamlined Version

Generates observation and intervention videos for the two-parts environment.
Intervention videos have some objects moving in reversed directions.

Usage:
    python generate_two_parts_dataset.py --num_videos 10 --num_intervention_videos 5 --output data/two_parts
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
from typing import Dict, Optional
from two_parts import TwoPartsEnv


class TwoPartsDatasetGenerator:
    """Generate datasets with observation and intervention videos"""
    
    def __init__(self, output_dir: str, clean_old: bool = False):
        self.output_dir = Path(output_dir)
        self._setup_directories(clean_old)
        
    def _setup_directories(self, clean_old: bool = False):
        """Create directory structure"""
        if clean_old and self.output_dir.exists():
            print(f"🗑️  Removing old data from {self.output_dir}")
            shutil.rmtree(self.output_dir)
            
        self.dirs = {
            'observation': self.output_dir / 'observation',
            'intervention': self.output_dir / 'intervention',
            'visualization': self.output_dir / 'visualization'
        }
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def sample_parameters(self) -> Dict:
        """Sample random parameters for video generation"""
        return {
            'width': random.choice([224, 400]),
            'height': random.choice([224, 300]),
            'left_objects': random.randint(2, 8),
            'right_objects': random.randint(2, 8),
            'velocity': random.uniform(1.0, 4.0),
            'object_radius': random.randint(8, 20)
        }
    
    def sample_intervention_parameters(self) -> Dict:
        """Sample parameters for intervention videos with reversed movement"""
        params = self.sample_parameters()
        total_left, total_right = params['left_objects'], params['right_objects']
        
        # Select 1-2 objects per side for intervention
        left_indices = random.sample(range(total_left), random.randint(1, min(2, total_left)))
        right_indices = random.sample(range(total_right), random.randint(1, min(2, total_right)))
        
        params.update({
            'is_intervention': True, 'left_intervention_indices': left_indices,
            'right_intervention_indices': right_indices, 'intervention_type': 'reversed_movement'
        })
        return params
    
    def generate_single_video(self, video_id: int, params: Dict, sample_type: str = 'observation') -> Dict:
        """Generate a single video with given parameters"""
        sample_id = f"{sample_type}_{video_id:06d}_{random.randint(100000, 999999)}"
        
        try:
            # Create environment with intervention support
            intervention_params = None
            if sample_type == 'intervention':
                intervention_params = {
                    'is_intervention': params.get('is_intervention', False),
                    'left_intervention_indices': params.get('left_intervention_indices', []),
                    'right_intervention_indices': params.get('right_intervention_indices', []),
                    'intervention_type': params.get('intervention_type', 'reversed_movement')
                }
            
            env = TwoPartsEnv(
                left_objects=params['left_objects'], right_objects=params['right_objects'],
                width=params['width'], height=params['height'], intervention_params=intervention_params
            )
            env.object_radius = params['object_radius']
            env.velocity = params['velocity']
            
            # Generate video
            seed = random.randint(1, 1000000)
            frames, metadata = env.generate_video(num_frames=60, fps=30, seed=seed, include_labels=False)
            
            if not frames:
                return {'sample_id': sample_id, 'status': 'failed', 'error': 'No frames generated'}
            
            # Save frames
            sample_dir = self.dirs[sample_type] / sample_id
            frames_dir = sample_dir / 'frames'
            frames_dir.mkdir(parents=True, exist_ok=True)
            
            for i, frame in enumerate(frames):
                cv2.imwrite(str(frames_dir / f"{i:06d}.png"), frame)
            
            # Save metadata
            meta = {'sample_id': sample_id, 'sample_type': sample_type, 'params': params,
                   'generation_info': {'timestamp': datetime.now().isoformat(), 'seed': seed},
                   'video_stats': {'num_frames': len(frames), 'frame_size': list(frames[0].shape)},
                   'physics_simulation': metadata}
            
            with open(sample_dir / 'meta.json', 'w') as f:
                json.dump(meta, f, indent=2)
            
            # Generate visualization GIF
            labeled_frames, _ = env.generate_video(num_frames=60, fps=30, seed=seed, include_labels=True)
            if labeled_frames:
                from PIL import Image
                gif_path = self.dirs['visualization'] / f"{sample_id}.gif"
                preview_frames = labeled_frames[:20]
                pil_images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in preview_frames]
                if pil_images:
                    pil_images[0].save(gif_path, save_all=True, append_images=pil_images[1:], duration=200, loop=0)
            
            return {'sample_id': sample_id, 'sample_type': sample_type, 'status': 'success', 'num_frames': len(frames), 'params': params}
            
        except Exception as e:
            return {'sample_id': sample_id, 'status': 'failed', 'error': str(e)}
    
    def generate_dataset(self, num_videos: int, num_intervention_videos: int = 0, create_gifs: bool = True):
        """Generate complete dataset with observation and intervention videos"""
        print(f"🎬 Generating Two Parts Dataset")
        print(f"Observation: {num_videos}, Intervention: {num_intervention_videos}, Output: {self.output_dir}")
        
        results = []
        
        # Generate videos
        for video_id in tqdm(range(num_videos), desc="Generating observation videos"):
            params = self.sample_parameters()
            result = self.generate_single_video(video_id, params, 'observation')
            results.append(result)
        
        for video_id in tqdm(range(num_intervention_videos), desc="Generating intervention videos"):
            params = self.sample_intervention_parameters()
            result = self.generate_single_video(video_id, params, 'intervention')
            results.append(result)
        
        # Summary
        successful = [r for r in results if r['status'] == 'success']
        obs_count = len([r for r in successful if r.get('sample_type', 'observation') == 'observation'])
        int_count = len([r for r in successful if r.get('sample_type', '') == 'intervention'])
        
        print(f"\n✅ Complete! Successful: {len(successful)}/{num_videos + num_intervention_videos}")
        print(f"   Observation: {obs_count}, Intervention: {int_count}")
        print(f"   Location: {self.output_dir}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Generate Two Parts Dataset')
    parser.add_argument('--num_videos', type=int, default=10,
                       help='Number of observation videos to generate')
    parser.add_argument('--num_intervention_videos', type=int, default=2,
                       help='Number of intervention videos to generate')
    parser.add_argument('--output', type=str, default='data/two_parts',
                       help='Output directory for dataset')
    parser.add_argument('--clean_old', action='store_true',
                       help='Remove old generated data before creating new dataset')
    
    args = parser.parse_args()
    
    # Create generator and generate dataset
    generator = TwoPartsDatasetGenerator(args.output, clean_old=args.clean_old)
    generator.generate_dataset(
        num_videos=args.num_videos,
        num_intervention_videos=args.num_intervention_videos
    )


if __name__ == "__main__":
    main()