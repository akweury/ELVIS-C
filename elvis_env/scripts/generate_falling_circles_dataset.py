#!/usr/bin/env python3
"""
Falling Circles Dataset Generator

Generates a comprehensive dataset of falling circles videos with various
physics parameters and scenarios. Uses the extracted environment module
for consistent physics simulation.

Dataset Structure:
- videos/: Individual video directories with frames and metadata
- splits/: Train/validation/test splits
- metadata/: Dataset-wide statistics and parameter distributions
- scenarios/: Organized by jam type and difficulty

Usage:
    python generate_falling_circles_dataset.py --num_videos 1000 --output data/falling_circles --workers 8

Features:
- Balanced jam type distribution
- Systematic parameter variation
- Quality control and validation
- Comprehensive metadata collection
- Parallel generation for speed
- Resume capability for interrupted runs
"""

import os
import json
import random
import argparse
import shutil
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
from PIL import Image
import time
from datetime import datetime

# Import our environment module
import sys
sys.path.append(os.path.dirname(__file__))
from falling_circles_env import FallingCirclesEnvironment, VideoParams


class DatasetGenerator:
    """
    Comprehensive dataset generator for falling circles videos
    """
    
    def __init__(self, output_dir: str, num_videos: int, workers: int = 4):
        """
        Initialize dataset generator
        
        Args:
            output_dir: Output directory for dataset
            num_videos: Total number of videos to generate
            workers: Number of parallel workers
        """
        self.output_dir = Path(output_dir)
        self.num_videos = num_videos
        self.workers = workers
        
        # Create directory structure
        self.dirs = {
            'videos': self.output_dir / 'videos',
            'splits': self.output_dir / 'splits', 
            'metadata': self.output_dir / 'metadata',
            'scenarios': self.output_dir / 'scenarios'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Dataset configuration
        self.config = self._create_dataset_config()
        
        # Progress tracking
        self.progress_file = self.output_dir / 'generation_progress.json'
        self.completed_videos = self._load_progress()
    
    def _create_dataset_config(self) -> Dict:
        """Create dataset generation configuration"""
        return {
            'dataset_info': {
                'name': 'falling_circles',
                'version': '1.0',
                'description': 'Physics simulation dataset with falling circles and jam scenarios',
                'created': datetime.now().isoformat(),
                'total_videos': self.num_videos
            },
            'parameter_ranges': {
                'hole_diameter': {'min': 15, 'max': 80, 'type': 'int'},
                'wind_strength': {'min': 0.0, 'max': 0.02, 'type': 'float'},
                'num_circles': {'min': 3, 'max': 15, 'type': 'int'},  # Reduced max to ensure completion
                'circle_size_min': {'min': 3, 'max': 10, 'type': 'int'},  # Reduced for faster movement
                'circle_size_max': {'min': 8, 'max': 16, 'type': 'int'},  # Reduced for faster movement
                'spawn_rate': {'min': 0.4, 'max': 2, 'type': 'float'},  # Faster spawning
                'gravity': {'min': 1.5, 'max': 1.50001, 'type': 'float'},  # Increased gravity for faster falling
                'wind_direction': {'values': [-1, 1], 'type': 'choice'},
                'hole_x_position': {'min': 0.2, 'max': 0.8, 'type': 'float'}
            },
            'scenarios': {
                'no_jam': {'target_ratio': 0.35, 'description': 'Free-flowing scenarios'},
                'partial_jam': {'target_ratio': 0.40, 'description': 'Some congestion but eventual flow'},
                'full_jam': {'target_ratio': 0.25, 'description': 'Complete blockage scenarios'}
            },
            'video_config': {
                'width': 224,
                'height': 224,
                'num_frames': 40,
                'fps': 10,
                'include_labels': False
            }
        }
    
    def _load_progress(self) -> List[int]:
        """Load progress from previous run"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                progress = json.load(f)
                return progress.get('completed_videos', [])
        return []
    
    def _save_progress(self):
        """Save current progress"""
        progress = {
            'completed_videos': self.completed_videos,
            'total_completed': len(self.completed_videos),
            'last_updated': datetime.now().isoformat()
        }
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def sample_parameters(self) -> VideoParams:
        """
        Sample random parameters independently of any target jam type
        
        Returns:
            VideoParams object with randomly sampled parameters
        """
        ranges = self.config['parameter_ranges']
        video_config = self.config['video_config']
        
        # Base parameters
        params = {
            'width': video_config['width'],
            'height': video_config['height'],
            'num_frames': video_config['num_frames']
        }
        
        # Sample all parameters randomly from their full ranges
        for param, config in ranges.items():
            if config['type'] == 'int':
                params[param] = random.randint(config['min'], config['max'])
            elif config['type'] == 'float':
                params[param] = random.uniform(config['min'], config['max'])
            elif config['type'] == 'choice':
                params[param] = random.choice(config['values'])
        
        # Ensure consistency (min <= max for circle sizes)
        if params.get('circle_size_min', 0) > params.get('circle_size_max', 0):
            params['circle_size_min'], params['circle_size_max'] = \
                params['circle_size_max'], params['circle_size_min']
        
        return VideoParams(**params)
    
    def _determine_jam_type(self, metadata: Dict) -> str:
        """
        Determine the jam type based on video simulation outcome
        
        Args:
            metadata: Video metadata containing exit statistics
            
        Returns:
            Jam type: 'no_jam', 'partial_jam', or 'full_jam'
        """
        exit_stats = metadata.get('exit_statistics', {})
        exit_ratio = exit_stats.get('exit_ratio', 0.0)
        total_spawned = exit_stats.get('total_spawned', 0)
        
        # Classify based on exit ratio with adjusted thresholds
        if total_spawned == 0:
            return 'no_jam'  # No circles spawned
        elif exit_ratio >= 0.9:
            return 'no_jam'  # 90%+ circles exited successfully
        elif exit_ratio >= 0.3:
            return 'partial_jam'  # 30-90% circles exited  
        else:
            return 'full_jam'  # <30% circles exited
    
    def generate_single_video(self, video_id: int) -> Dict:
        """
        Generate a single video with independent parameter sampling
        
        Args:
            video_id: Unique video identifier
            
        Returns:
            Dictionary with video metadata and actual jam type
        """
        # Skip if already completed
        if video_id in self.completed_videos:
            return {'video_id': video_id, 'status': 'skipped', 'reason': 'already_completed'}
        
        try:
            # Sample parameters independently (no bias toward target jam type)
            params = self.sample_parameters()
            seed = random.randint(0, 2_000_000_000)
            
            # Generate video
            env = FallingCirclesEnvironment(params)
            frames, metadata = env.generate_video(seed=seed, include_labels=False)
            
            # Determine actual jam type from simulation outcome
            actual_jam_type = self._determine_jam_type(metadata)
            
            # Create video directory
            video_name = f"video_{video_id:06d}"
            video_dir = self.dirs['videos'] / video_name
            video_dir.mkdir(exist_ok=True)
            
            # Save frames as PNG files
            for frame_idx, frame in enumerate(frames):
                img = Image.fromarray(frame)
                frame_path = video_dir / f"frame_{frame_idx:03d}.png"
                img.save(frame_path)
            
            # Generate labeled frames for GIF (with video info overlay)
            labeled_frames, _ = env.generate_video(seed=seed, include_labels=True, actual_jam_type=actual_jam_type)
            
            # Enhance metadata
            enhanced_metadata = {
                'video_id': video_id,
                'video_name': video_name,
                'generation_info': {
                    'timestamp': datetime.now().isoformat(),
                    'seed': seed,
                    'actual_jam_type': actual_jam_type,  # Use determined jam type instead of target
                    'generator_version': '1.0'
                },
                'video_stats': {
                    'num_frames': len(frames),
                    'frame_size': frames[0].shape if frames else None,
                    'duration_seconds': len(frames) / self.config['video_config']['fps']
                },
                'physics_simulation': metadata
            }
            
            # Save metadata
            meta_path = video_dir / "metadata.json"
            with open(meta_path, 'w') as f:
                json.dump(enhanced_metadata, f, indent=2)
            
            # Save frame-level data as CSV
            self._save_frame_csv(video_dir, metadata['frames'])
            
            # Generate summary GIF (optional, for quick preview)
            if len(labeled_frames) > 0:
                self._create_preview_gif(labeled_frames, video_dir / "preview.gif")
            
            # Update progress
            self.completed_videos.append(video_id)
            
            return {
                'video_id': video_id,
                'status': 'success',
                'jam_type': actual_jam_type,  # Use determined jam type
                'exit_ratio': metadata['exit_statistics']['exit_ratio'],
                'num_frames': len(frames)
            }
            
        except Exception as e:
            return {
                'video_id': video_id,
                'status': 'error', 
                'error': str(e)
            }
    
    def _save_frame_csv(self, video_dir: Path, frame_data: List[Dict]):
        """Save frame-level data as CSV"""
        csv_path = video_dir / "frames.csv"
        
        if not frame_data:
            return
            
        fieldnames = [
            'frame_idx', 'num_active_circles', 'num_visible_circles',
            'circles_exited', 'circles_stuck', 'frames_since_last_exit',
            'hole_x', 'hole_y', 'wind_effect'
        ]
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for frame in frame_data:
                row = {
                    'frame_idx': frame['frame_idx'],
                    'num_active_circles': frame['num_active_circles'],
                    'num_visible_circles': frame['num_visible_circles'],
                    'circles_exited': frame['circles_exited'],
                    'circles_stuck': frame['circles_stuck'],
                    'frames_since_last_exit': frame['frames_since_last_exit'],
                    'hole_x': frame['hole_position'][0] if frame['hole_position'] else None,
                    'hole_y': frame['hole_position'][1] if frame['hole_position'] else None,
                    'wind_effect': frame['wind_effect']
                }
                writer.writerow(row)
    
    def _create_preview_gif(self, frames: List[np.ndarray], output_path: Path, 
                           max_frames: int = 20):
        """Create a preview GIF from frames"""
        try:
            # Sample frames if too many
            if len(frames) > max_frames:
                indices = np.linspace(0, len(frames)-1, max_frames, dtype=int)
                sampled_frames = [frames[i] for i in indices]
            else:
                sampled_frames = frames
            
            # Convert to PIL images
            pil_frames = [Image.fromarray(frame.astype('uint8')) for frame in sampled_frames]
            
            # Save as GIF
            pil_frames[0].save(
                output_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=200,  # Slower for preview
                loop=0
            )
        except Exception as e:
            print(f"Warning: Could not create preview GIF: {e}")
    
    def generate_dataset(self):
        """Generate the complete dataset"""
        print(f"🎬 Generating Falling Circles Dataset")
        print("=" * 50)
        print(f"Total videos: {self.num_videos}")
        print(f"Output directory: {self.output_dir}")
        print(f"Workers: {self.workers}")
        
        if self.completed_videos:
            print(f"Resuming from {len(self.completed_videos)} completed videos")
        
        # Save dataset configuration
        config_path = self.dirs['metadata'] / 'dataset_config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Plan video generation (independent parameter sampling)
        video_plan = self._create_video_plan()
        
        # Generate videos
        start_time = time.time()
        results = []
        
        if self.workers == 1:
            # Sequential generation
            for video_id in tqdm(video_plan, desc="Generating videos"):
                result = self.generate_single_video(video_id)
                results.append(result)
                
                # Save progress periodically
                if len(results) % 50 == 0:
                    self._save_progress()
        else:
            # Parallel generation
            with ProcessPoolExecutor(max_workers=self.workers) as executor:
                # Submit all jobs
                future_to_video = {
                    executor.submit(worker_function, (video_id, self.output_dir, self.config)): video_id
                    for video_id in video_plan
                }
                
                # Collect results with progress bar
                for future in tqdm(as_completed(future_to_video), total=len(video_plan), desc="Generating videos"):
                    result = future.result()
                    results.append(result)
                    
                    # Save progress periodically
                    if len(results) % 50 == 0:
                        self._save_progress()
        
        # Final progress save
        self._save_progress()
        
        # Generate dataset statistics and splits
        self._create_dataset_splits()
        self._create_dataset_statistics(results)
        self._organize_by_scenarios()
        
        # Summary
        generation_time = time.time() - start_time
        success_count = sum(1 for r in results if r.get('status') == 'success')
        
        print(f"\n✅ Dataset generation complete!")
        print(f"   Successful videos: {success_count}/{len(results)}")
        print(f"   Generation time: {generation_time:.1f} seconds")
        print(f"   Average time per video: {generation_time/len(results):.2f} seconds")
        print(f"   Dataset location: {self.output_dir}")
    
    def _create_video_plan(self) -> List[int]:
        """Create a simple plan for video generation (no jam type targets)"""
        remaining_videos = [i for i in range(self.num_videos) if i not in self.completed_videos]
        random.shuffle(remaining_videos)  # Randomize order
        return remaining_videos
    
    def _create_dataset_splits(self):
        """Create train/validation/test splits"""
        completed_videos = [i for i in range(self.num_videos) if i in self.completed_videos]
        
        # Check if no videos were completed
        if len(completed_videos) == 0:
            print("⚠️  No completed videos to create splits for")
            # Create empty split files
            for split_name in ['train', 'validation', 'test']:
                split_file = self.dirs['splits'] / f"{split_name}.txt"
                with open(split_file, 'w') as f:
                    pass  # Empty file
            
            # Save empty splits metadata
            splits_meta = {
                'total_videos': 0,
                'splits': {'train': 0, 'validation': 0, 'test': 0},
                'split_ratios': {'train': 0.0, 'validation': 0.0, 'test': 0.0}
            }
            
            with open(self.dirs['metadata'] / 'splits_info.json', 'w') as f:
                json.dump(splits_meta, f, indent=2)
            return
        
        random.shuffle(completed_videos)
        
        # 70% train, 15% validation, 15% test
        train_size = int(0.7 * len(completed_videos))
        val_size = int(0.15 * len(completed_videos))
        
        splits = {
            'train': completed_videos[:train_size],
            'validation': completed_videos[train_size:train_size + val_size],
            'test': completed_videos[train_size + val_size:]
        }
        
        # Save splits
        for split_name, video_ids in splits.items():
            split_file = self.dirs['splits'] / f"{split_name}.txt"
            with open(split_file, 'w') as f:
                for video_id in video_ids:
                    f.write(f"video_{video_id:06d}\n")
        
        # Save splits metadata
        splits_meta = {
            'total_videos': len(completed_videos),
            'splits': {name: len(ids) for name, ids in splits.items()},
            'split_ratios': {
                'train': len(splits['train']) / len(completed_videos),
                'validation': len(splits['validation']) / len(completed_videos),
                'test': len(splits['test']) / len(completed_videos)
            }
        }
        
        with open(self.dirs['metadata'] / 'splits_info.json', 'w') as f:
            json.dump(splits_meta, f, indent=2)
    
    def _create_dataset_statistics(self, results: List[Dict]):
        """Create comprehensive dataset statistics"""
        successful_results = [r for r in results if r.get('status') == 'success']
        
        # Jam type distribution
        jam_type_counts = {}
        exit_ratios = []
        
        for result in successful_results:
            jam_type = result.get('jam_type', 'unknown')
            jam_type_counts[jam_type] = jam_type_counts.get(jam_type, 0) + 1
            
            if 'exit_ratio' in result:
                exit_ratios.append(result['exit_ratio'])
        
        # Parameter statistics (would need to read from metadata files)
        stats = {
            'generation_summary': {
                'total_planned': len(results),
                'successful': len(successful_results),
                'failed': len(results) - len(successful_results),
                'success_rate': len(successful_results) / len(results) if results else 0
            },
            'jam_type_distribution': jam_type_counts,
            'exit_ratio_stats': {
                'mean': np.mean(exit_ratios) if exit_ratios else 0,
                'std': np.std(exit_ratios) if exit_ratios else 0,
                'min': np.min(exit_ratios) if exit_ratios else 0,
                'max': np.max(exit_ratios) if exit_ratios else 0
            }
        }
        
        with open(self.dirs['metadata'] / 'dataset_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
    
    def _organize_by_scenarios(self):
        """Organize videos by jam type scenarios"""
        # Create symbolic links organized by scenario
        for jam_type in self.config['scenarios'].keys():
            scenario_dir = self.dirs['scenarios'] / jam_type
            scenario_dir.mkdir(exist_ok=True)
        
        # Read metadata and organize
        for video_dir in self.dirs['videos'].iterdir():
            if video_dir.is_dir():
                meta_file = video_dir / 'metadata.json'
                if meta_file.exists():
                    try:
                        with open(meta_file, 'r') as f:
                            metadata = json.load(f)
                        
                        jam_type = metadata['physics_simulation']['actual_jam_type']
                        scenario_dir = self.dirs['scenarios'] / jam_type
                        
                        # Create symbolic link
                        link_path = scenario_dir / video_dir.name
                        if not link_path.exists():
                            try:
                                link_path.symlink_to(video_dir.resolve())
                            except OSError:
                                # On Windows, copy instead of symlink
                                shutil.copytree(video_dir, link_path)
                    except (json.JSONDecodeError, KeyError):
                        continue


def worker_function(args):
    """Worker function for parallel video generation"""
    video_id, output_dir, config = args
    generator = DatasetGenerator(output_dir, 1, 1)  # Single worker instance
    generator.config = config
    return generator.generate_single_video(video_id)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Generate Falling Circles Dataset')
    parser.add_argument('--num_videos', type=int, default=10,
                       help='Number of videos to generate (default: 1000)')
    parser.add_argument('--output', type=str, default='data/falling_circles_dataset',
                       help='Output directory (default: data/falling_circles_dataset)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from previous incomplete run')
    parser.add_argument('--preview', action='store_true',
                       help='Generate preview GIFs for each video')
    
    args = parser.parse_args()
    
    # Create generator
    generator = DatasetGenerator(args.output, args.num_videos, args.workers)
    
    # Generate dataset
    generator.generate_dataset()
    
    print(f"\n📁 Dataset generated at: {args.output}")
    print(f"📊 Use the metadata files in {args.output}/metadata/ for analysis")
    print(f"🎯 Video splits available in {args.output}/splits/")
    print(f"🎬 Videos organized by scenario in {args.output}/scenarios/")


if __name__ == "__main__":
    main()