#!/usr/bin/env python3
"""
Two Parts Dataset Generator

Generates a comprehensive dataset of two-parts videos with various
physics parameters and scenarios. Uses the TwoPartsEnv environment
for consistent physics simulation.

Dataset Structure:
- videos/: Individual video directories with frames and metadata
- splits/: Train/validation/test splits
- metadata/: Dataset-wide statistics and parameter distributions
- scenarios/: Organized by object count and movement patterns
- visualization/: Preview GIFs for quick inspection

Usage:
    python generate_two_parts_dataset.py --num_videos 1000 --output data/two_parts --workers 8

Features:
- Balanced object count distribution
- Systematic parameter variation
- Quality control and validation
- Comprehensive metadata collection
- Parallel generation for speed
- Resume capability for interrupted runs
- Preview GIF generation
"""

import os
import sys
import time
import json
import random
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import csv
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import warnings
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image
import cv2
import yaml
warnings.filterwarnings('ignore')

# Import our environment module
from two_parts import TwoPartsEnv


class TwoPartsDatasetGenerator:
    """
    Generate comprehensive datasets of two-parts simulations
    """
    
    def __init__(self, output_dir: str, config_path: Optional[str] = None):
        """
        Initialize dataset generator
        
        Args:
            output_dir: Base directory for dataset output
            config_path: Path to configuration file (uses defaults if None)
        """
        self.output_dir = Path(output_dir)
        self.config = self._load_config(config_path)
        self.completed_videos = set()
        self.progress_file = self.output_dir / "generation_progress.json"
        
        # Create directory structure
        self._setup_directories()
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load configuration from YAML file or use defaults"""
        if config_path is None:
            # Use default config file path
            config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'two_parts_dataset_config.yaml')
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                print(f"📄 Loaded configuration from {config_path}")
                return config
            except yaml.YAMLError as e:
                print(f"⚠️  Error parsing YAML config: {e}")
                return self._get_default_config()
        
        # Use default configuration
        print("📄 Using default configuration")
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration parameters"""
        return {
            'video_config': {
                'width': 400,
                'height': 300,
                'fps': 30,
                'duration': 2  # seconds
            },
            'parameter_ranges': {
                'left_objects': {'min': 2, 'max': 8},
                'right_objects': {'min': 2, 'max': 8},
                'velocity': {'min': 1.0, 'max': 4.0},
                'object_radius': {'min': 8, 'max': 20}
            },
            'splits': {
                'train': 0.7,
                'val': 0.15,
                'test': 0.15
            },
            'quality_control': {
                'min_movement_threshold': 5.0,
                'max_collision_rate': 0.8
            },
            'generation': {
                'num_workers': 4,
                'batch_size': 100
            },
            'output': {
                'formats': ['frames', 'gif'],
                'frame_format': 'png',
                'gif_settings': {
                    'duration': 100,
                    'loop': 0
                }
            }
        }
    
    def _setup_directories(self):
        """Create necessary directory structure"""
        self.dirs = {
            'observation': self.output_dir / 'observation',
            'intervention': self.output_dir / 'intervention', 
            'ood': self.output_dir / 'ood',
            'splits': self.output_dir / 'splits',
            'metadata': self.output_dir / 'metadata',
            'scenarios': self.output_dir / 'scenarios',
            'visualization': self.output_dir / 'visualization',
            'audits': self.output_dir / 'audits',
            'index': self.output_dir / 'index'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _load_progress(self):
        """Load previous progress if exists"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                    self.completed_videos = set(progress.get('completed_videos', []))
                    print(f"📋 Loaded progress: {len(self.completed_videos)} videos completed")
            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠️  Warning: Could not load progress: {e}")
                self.completed_videos = set()
    
    def _save_progress(self):
        """Save current progress"""
        try:
            progress = {
                'completed_videos': list(self.completed_videos),
                'total_completed': len(self.completed_videos),
                'last_updated': datetime.now().isoformat()
            }
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
        except (IOError, OSError) as e:
            print(f"⚠️  Warning: Could not save progress: {e}")
    
    def sample_parameters(self) -> Dict:
        """
        Sample random parameters for video generation
        
        Returns:
            Dictionary of sampled parameters
        """
        ranges = self.config['parameter_ranges']
        video_config = self.config['video_config']
        
        # Base parameters
        params = {
            'width': video_config['width'],
            'height': video_config['height']
        }
        
        # Sample all parameters randomly from their ranges
        for param, config in ranges.items():
            if config['type'] == 'int':
                params[param] = random.randint(config['min'], config['max'])
            elif config['type'] == 'float':
                params[param] = random.uniform(config['min'], config['max'])
            elif config['type'] == 'choice':
                params[param] = random.choice(config['values'])
        
        return params
    
    def _determine_scenario(self, params: Dict) -> str:
        """
        Determine scenario type based on parameters
        
        Args:
            params: Video parameters
            
        Returns:
            Scenario type string
        """
        total_objects = params['left_objects'] + params['right_objects']
        
        if total_objects <= 6:
            return 'sparse'
        elif total_objects <= 10:
            return 'medium'
        else:
            return 'dense'
    
    def _analyze_video_quality(self, metadata: Dict) -> Dict:
        """
        Analyze video quality metrics
        
        Args:
            metadata: Video metadata
            
        Returns:
            Quality analysis results
        """
        frames_data = metadata.get('frames', [])
        if not frames_data:
            return {'quality': 'poor', 'issues': ['no_frame_data']}
        
        # Calculate movement metrics
        total_movement = 0
        object_counts = []
        
        for frame_data in frames_data:
            object_counts.append(frame_data.get('total_objects', 0))
            
            # Calculate movement for this frame
            for obj in frame_data.get('objects', []):
                # Approximate movement (in real implementation, track between frames)
                total_movement += 1  # Placeholder
        
        avg_objects = np.mean(object_counts) if object_counts else 0
        movement_score = total_movement / len(frames_data) if frames_data else 0
        
        # Quality assessment
        issues = []
        if movement_score < self.config['quality_control']['min_movement_threshold']:
            issues.append('low_movement')
        
        if avg_objects == 0:
            issues.append('no_objects')
        
        quality = 'good' if not issues else 'poor'
        
        return {
            'quality': quality,
            'movement_score': movement_score,
            'avg_objects': avg_objects,
            'issues': issues
        }
    
    def generate_single_video(self, video_id: int, params: Dict, 
                             sample_type: str = 'observation',
                             create_gifs: bool = True) -> Dict:
        """
        Generate a single video with given parameters
        
        Args:
            video_id: Unique video identifier
            params: Video generation parameters
            sample_type: Type of sample ('observation', 'intervention', 'ood')
            create_gifs: Whether to generate labeled GIF previews
            
        Returns:
            Generation result dictionary
        """
        video_config = self.config['video_config']
        sample_id = f"{sample_type}_{video_id:06d}_{random.randint(100000, 999999)}"
        
        try:
            # Skip if already completed
            if sample_id in self.completed_videos:
                return {'sample_id': sample_id, 'status': 'skipped'}
            
            # Create environment
            env = TwoPartsEnv(
                left_objects=params['left_objects'],
                right_objects=params['right_objects'],
                width=params['width'],
                height=params['height']
            )
            env.object_radius = params['object_radius']
            env.velocity = params['velocity']
            
            # Generate video
            seed = random.randint(1, 1000000)
            frames, metadata = env.generate_video(
                num_frames=video_config['num_frames'],
                fps=video_config['fps'],
                seed=seed,
                include_labels=False
            )

            labeled_frames: List[np.ndarray] = []
            if create_gifs:
                labeled_env = TwoPartsEnv(
                    left_objects=params['left_objects'],
                    right_objects=params['right_objects'],
                    width=params['width'],
                    height=params['height']
                )
                labeled_env.object_radius = params['object_radius']
                labeled_env.velocity = params['velocity']
                labeled_frames, _ = labeled_env.generate_video(
                    num_frames=video_config['num_frames'],
                    fps=video_config['fps'],
                    seed=seed,
                    include_labels=True
                )
            
            if not frames:
                return {'sample_id': sample_id, 'status': 'failed', 'error': 'No frames generated'}
            
            # Determine scenario
            scenario = self._determine_scenario(params)
            
            # Analyze quality
            quality_info = self._analyze_video_quality(metadata)
            
            # Create sample directory
            if sample_type == 'observation':
                sample_dir = self.dirs['observation'] / sample_id
            elif sample_type == 'intervention':
                sample_dir = self.dirs['intervention'] / sample_id
            else:  # ood
                sample_dir = self.dirs['ood'] / sample_id
                
            sample_dir.mkdir(parents=True, exist_ok=True)
            
            # Save frames
            frames_dir = sample_dir / 'frames'
            frames_dir.mkdir(exist_ok=True)
            
            for i, frame in enumerate(frames):
                frame_path = frames_dir / f"{i:06d}.png"
                cv2.imwrite(str(frame_path), frame)
            
            # Create comprehensive metadata with entity mapping
            comprehensive_metadata = {
                'sample_id': sample_id,
                'sample_type': sample_type,
                'scenario': scenario,
                'params': params,
                'generation_info': {
                    'timestamp': datetime.now().isoformat(),
                    'seed': seed,
                    'generator_version': '1.0'
                },
                'video_stats': {
                    'num_frames': len(frames),
                    'frame_size': list(frames[0].shape),
                    'duration_seconds': len(frames) / video_config['fps']
                },
                'quality_analysis': quality_info,
                'entity_info': {
                    'entity_colors': metadata.get('entity_mapping', {}).get('colors', {}),
                    'entity_definitions': metadata.get('entity_mapping', {}).get('entities', {}),
                    'consistent_mapping': True,
                    'mapping_version': '1.0'
                },
                'physics_simulation': metadata
            }
            
            # Save metadata
            with open(sample_dir / 'meta.json', 'w') as f:
                json.dump(comprehensive_metadata, f, indent=2)
            
            # Save frame-by-frame statistics with entity information
            frame_stats = []
            for frame_data in metadata.get('frames', []):
                # Basic frame stats
                frame_stat = {
                    'frame_idx': frame_data.get('frame_idx', 0),
                    'left_objects': frame_data.get('left_objects', 0),
                    'right_objects': frame_data.get('right_objects', 0),
                    'total_objects': frame_data.get('total_objects', 0)
                }
                
                # Add entity-specific positions for tracking
                objects = frame_data.get('objects', [])
                for obj in objects:
                    entity_id = obj.get('entity_id', 'unknown')
                    frame_stat[f'{entity_id}_x'] = obj.get('x', 0)
                    frame_stat[f'{entity_id}_y'] = obj.get('y', 0)
                    
                frame_stats.append(frame_stat)
            
            # Save stats as CSV
            if frame_stats:
                stats_df = pd.DataFrame(frame_stats)
                stats_df.to_csv(sample_dir / 'stats.csv', index=False)
            
            # Generate visualization GIF with labeled frames when requested
            if create_gifs:
                gif_path = self.dirs['visualization'] / f"{sample_id}.gif"
                gif_source_frames = labeled_frames if labeled_frames else frames
                preview_frames = gif_source_frames[:min(20, len(gif_source_frames))]
                
                pil_images = []
                for frame in preview_frames:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    pil_images.append(pil_image)
                
                if pil_images:
                    pil_images[0].save(
                        gif_path,
                        save_all=True,
                        append_images=pil_images[1:],
                        duration=200,  # 200ms per frame
                        loop=0
                    )
            
            # Create scenario link
            scenario_dir = self.dirs['scenarios'] / scenario
            scenario_dir.mkdir(exist_ok=True)
            scenario_link = scenario_dir / f"{sample_id}.json"
            
            with open(scenario_link, 'w') as f:
                json.dump({
                    'sample_id': sample_id,
                    'scenario': scenario,
                    'path': str(sample_dir.relative_to(self.output_dir))
                }, f, indent=2)
            
            self.completed_videos.add(sample_id)
            
            return {
                'sample_id': sample_id,
                'scenario': scenario,
                'status': 'success',
                'quality': quality_info['quality'],
                'num_frames': len(frames),
                'params': params
            }
            
        except Exception as e:
            return {
                'sample_id': sample_id,
                'status': 'failed',
                'error': str(e)
            }
    
    def create_dataset_splits(self, samples: List[Dict]):
        """Create train/val/test splits"""
        if not samples:
            return
        
        # Shuffle samples
        random.shuffle(samples)
        
        # Calculate split sizes
        total = len(samples)
        train_size = int(total * self.config['splits']['train'])
        val_size = int(total * self.config['splits']['val'])
        
        # Create splits
        train_samples = samples[:train_size]
        val_samples = samples[train_size:train_size + val_size]
        test_samples = samples[train_size + val_size:]
        
        # Save splits
        splits = {
            'train': train_samples,
            'val': val_samples,
            'test': test_samples
        }
        
        for split_name, split_samples in splits.items():
            split_file = self.dirs['splits'] / f"{split_name}.json"
            with open(split_file, 'w') as f:
                json.dump([s['sample_id'] for s in split_samples], f, indent=2)
        
        print(f"✅ Created dataset splits with {len(samples)} samples:")
        for split_name, split_samples in splits.items():
            percentage = len(split_samples) / len(samples) * 100
            print(f"   {split_name}: {len(split_samples)} samples ({percentage:.1f}%)")
    
    def create_samples_index(self, samples: List[Dict]):
        """Create comprehensive samples index"""
        index_data = {
            'dataset_info': {
                'name': 'two_parts_dataset',
                'version': '1.0',
                'created': datetime.now().isoformat(),
                'total_samples': len(samples),
                'generator_config': self.config
            },
            'samples': []
        }
        
        for sample in samples:
            if sample['status'] == 'success':
                index_data['samples'].append({
                    'sample_id': sample['sample_id'],
                    'scenario': sample['scenario'],
                    'quality': sample['quality'],
                    'num_frames': sample['num_frames'],
                    'params': sample['params']
                })
        
        # Save comprehensive index
        with open(self.dirs['index'] / 'samples.json', 'w') as f:
            json.dump(index_data, f, indent=2)
        
        print(f"📋 Created samples index with {len(index_data['samples'])} entries")
    
    def analyze_dataset(self, samples: List[Dict]):
        """Generate dataset analysis and visualizations"""
        if not samples:
            return
        
        successful_samples = [s for s in samples if s['status'] == 'success']
        
        # Parameter distributions
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Left objects distribution
        left_objects = [s['params']['left_objects'] for s in successful_samples]
        axes[0, 0].hist(left_objects, bins=20, alpha=0.7, color='blue')
        axes[0, 0].set_title('Left Objects Distribution')
        axes[0, 0].set_xlabel('Number of Left Objects')
        axes[0, 0].set_ylabel('Frequency')
        
        # Right objects distribution
        right_objects = [s['params']['right_objects'] for s in successful_samples]
        axes[0, 1].hist(right_objects, bins=20, alpha=0.7, color='red')
        axes[0, 1].set_title('Right Objects Distribution')
        axes[0, 1].set_xlabel('Number of Right Objects')
        axes[0, 1].set_ylabel('Frequency')
        
        # Velocity distribution
        velocities = [s['params']['velocity'] for s in successful_samples]
        axes[1, 0].hist(velocities, bins=20, alpha=0.7, color='green')
        axes[1, 0].set_title('Velocity Distribution')
        axes[1, 0].set_xlabel('Velocity')
        axes[1, 0].set_ylabel('Frequency')
        
        # Scenario distribution
        scenarios = [s['scenario'] for s in successful_samples]
        scenario_counts = pd.Series(scenarios).value_counts()
        axes[1, 1].pie(scenario_counts.values, labels=scenario_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Scenario Distribution')
        
        plt.tight_layout()
        plt.savefig(self.dirs['audits'] / 'parameter_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create JSON-serializable analysis report
        left_stats = {
            'mean': float(np.mean(left_objects)) if left_objects else None,
            'std': float(np.std(left_objects)) if left_objects else None
        }
        right_stats = {
            'mean': float(np.mean(right_objects)) if right_objects else None,
            'std': float(np.std(right_objects)) if right_objects else None
        }
        velocity_stats = {
            'mean': float(np.mean(velocities)) if velocities else None,
            'std': float(np.std(velocities)) if velocities else None
        }
        scenario_distribution = {
            str(key): int(value) for key, value in scenario_counts.to_dict().items()
        }
        quality_distribution_series = pd.Series([s['quality'] for s in successful_samples])
        quality_distribution = {
            str(key): int(value) for key, value in quality_distribution_series.value_counts().to_dict().items()
        }

        analysis = {
            'total_samples': int(len(samples)),
            'successful_samples': int(len(successful_samples)),
            'failed_samples': int(len(samples) - len(successful_samples)),
            'parameter_stats': {
                'left_objects': left_stats,
                'right_objects': right_stats,
                'velocity': velocity_stats
            },
            'scenario_distribution': scenario_distribution,
            'quality_distribution': quality_distribution
        }
        
        with open(self.dirs['audits'] / 'dataset_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print("📊 Dataset analysis saved to audits/dataset_analysis.json")
        print("📊 Parameter distributions saved to audits/parameter_distributions.png")
    
    def generate_dataset(self, num_videos: int, workers: int = 1, resume: bool = False, 
                        create_gifs: bool = True):
        """
        Generate complete dataset
        
        Args:
            num_videos: Number of videos to generate
            workers: Number of parallel workers
            resume: Whether to resume from previous run
            create_gifs: Whether to create preview GIFs
        """
        start_time = time.time()
        
        if resume:
            self._load_progress()
        
        print(f"🎬 Generating Two Parts Dataset")
        print("=" * 50)
        print(f"Total videos: {num_videos}")
        print(f"Output directory: {self.output_dir}")
        print(f"Workers: {workers}")

        # Share GIF preference with worker processes
        self.config['create_gifs'] = create_gifs
        
        # Generate videos
        if workers == 1:
            # Single-threaded generation
            results = []
            for video_id in tqdm(range(num_videos), desc="Generating videos"):
                if video_id in self.completed_videos:
                    continue
                
                params = self.sample_parameters()
                result = self.generate_single_video(video_id, params, create_gifs=create_gifs)
                results.append(result)
                
                # Save progress periodically
                if len(results) % 10 == 0:
                    self._save_progress()
        else:
            # Multi-threaded generation
            results = []
            video_tasks = []
            
            for video_id in range(num_videos):
                if video_id in self.completed_videos:
                    continue
                params = self.sample_parameters()
                video_tasks.append((video_id, params))
            
            with ProcessPoolExecutor(max_workers=workers) as executor:
                future_to_task = {
                    executor.submit(worker_generate_video, task, self.config, str(self.output_dir)): task 
                    for task in video_tasks
                }
                
                for future in tqdm(as_completed(future_to_task), total=len(video_tasks), desc="Generating videos"):
                    result = future.result()
                    results.append(result)
                    
                    # Save progress periodically
                    if len(results) % 10 == 0:
                        self._save_progress()
        
        # Final progress save
        self._save_progress()
        
        # Create dataset organization
        successful_results = [r for r in results if r['status'] == 'success']
        
        self.create_samples_index(successful_results)
        self.create_dataset_splits(successful_results)
        self.analyze_dataset(results)
        
        # Summary
        generation_time = time.time() - start_time
        success_count = len(successful_results)
        
        print(f"\n✅ Dataset generation complete!")
        print(f"   Successful videos: {success_count}/{num_videos}")
        print(f"   Generation time: {generation_time:.1f} seconds")
        print(f"   Average time per video: {generation_time/num_videos:.2f} seconds")
        print(f"   Dataset location: {self.output_dir}")
        
        print(f"\n📁 Dataset generated at: {self.output_dir}")
        print(f"📊 Use the metadata files in {self.output_dir}/metadata/ for analysis")
        print(f"🎯 Video splits available in {self.output_dir}/splits/")
        print(f"🎬 Videos organized by scenario in {self.output_dir}/scenarios/")


def worker_generate_video(task: Tuple, config: Dict, output_dir: str) -> Dict:
    """
    Worker function for parallel video generation
    
    Args:
        task: Tuple of (video_id, params)
        config: Configuration dictionary
        output_dir: Output directory path
        
    Returns:
        Generation result dictionary
    """
    try:
        from two_parts import TwoPartsEnv
        import random
        import cv2
        from PIL import Image
        from pathlib import Path
        import numpy as np
        import pandas as pd
        import json
        from datetime import datetime
        
        video_id, params = task
        video_config = config['video_config']
        create_gifs = config.get('create_gifs', True)
        sample_id = f"obs_{video_id:06d}_{random.randint(100000, 999999)}"
        
        output_path = Path(output_dir)
        observation_dir = output_path / 'observation'
        visualization_dir = output_path / 'visualization'
        scenarios_dir = output_path / 'scenarios'
        
        # Create environment
        env = TwoPartsEnv(
            left_objects=params['left_objects'],
            right_objects=params['right_objects'],
            width=params['width'],
            height=params['height']
        )
        env.object_radius = params['object_radius']
        env.velocity = params['velocity']
        
        # Generate video
        seed = random.randint(1, 1000000)
        frames, metadata = env.generate_video(
            num_frames=video_config['num_frames'],
            fps=video_config['fps'],
            seed=seed,
            include_labels=False
        )

        labeled_frames: List[np.ndarray] = []
        if create_gifs:
            labeled_env = TwoPartsEnv(
                left_objects=params['left_objects'],
                right_objects=params['right_objects'],
                width=params['width'],
                height=params['height']
            )
            labeled_env.object_radius = params['object_radius']
            labeled_env.velocity = params['velocity']
            labeled_frames, _ = labeled_env.generate_video(
                num_frames=video_config['num_frames'],
                fps=video_config['fps'],
                seed=seed,
                include_labels=True
            )
        
        if not frames:
            return {'sample_id': sample_id, 'status': 'failed', 'error': 'No frames generated'}
        
        # Determine scenario
        total_objects = params['left_objects'] + params['right_objects']
        if total_objects <= 6:
            scenario = 'sparse'
        elif total_objects <= 10:
            scenario = 'medium'
        else:
            scenario = 'dense'
        
        # Create sample directory
        sample_dir = observation_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Save frames
        frames_dir = sample_dir / 'frames'
        frames_dir.mkdir(exist_ok=True)
        
        for i, frame in enumerate(frames):
            frame_path = frames_dir / f"{i:06d}.png"
            cv2.imwrite(str(frame_path), frame)
        
        # Create comprehensive metadata
        comprehensive_metadata = {
            'sample_id': sample_id,
            'sample_type': 'observation',
            'scenario': scenario,
            'params': params,
            'generation_info': {
                'timestamp': datetime.now().isoformat(),
                'seed': seed,
                'generator_version': '1.0'
            },
            'video_stats': {
                'num_frames': len(frames),
                'frame_size': list(frames[0].shape),
                'duration_seconds': len(frames) / video_config['fps']
            },
            'physics_simulation': metadata
        }
        
        # Save metadata
        with open(sample_dir / 'meta.json', 'w') as f:
            json.dump(comprehensive_metadata, f, indent=2)
        
        # Save frame-by-frame statistics
        frame_stats = []
        for frame_data in metadata.get('frames', []):
            frame_stats.append({
                'frame_idx': frame_data.get('frame_idx', 0),
                'left_objects': frame_data.get('left_objects', 0),
                'right_objects': frame_data.get('right_objects', 0),
                'total_objects': frame_data.get('total_objects', 0)
            })
        
        # Save stats as CSV
        if frame_stats:
            stats_df = pd.DataFrame(frame_stats)
            stats_df.to_csv(sample_dir / 'stats.csv', index=False)
        
        # Generate GIF if requested
        if create_gifs:
            visualization_dir.mkdir(parents=True, exist_ok=True)
            gif_path = visualization_dir / f"{sample_id}.gif"
            gif_source_frames = labeled_frames if labeled_frames else frames
            preview_frames = gif_source_frames[:min(20, len(gif_source_frames))]
            
            pil_images = []
            for frame in preview_frames:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                pil_images.append(pil_image)
            
            if pil_images:
                pil_images[0].save(
                    gif_path,
                    save_all=True,
                    append_images=pil_images[1:],
                    duration=200,
                    loop=0
                )
        
        return {
            'sample_id': sample_id,
            'scenario': scenario,
            'status': 'success',
            'quality': 'good',  # Simplified for worker
            'num_frames': len(frames),
            'params': params
        }
        
    except Exception as e:
        return {
            'sample_id': sample_id,
            'status': 'failed',
            'error': str(e)
        }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Generate Two Parts Dataset')
    parser.add_argument('--num_videos', type=int, default=10,
                       help='Number of videos to generate')
    parser.add_argument('--output', type=str, default='data/two_parts',
                       help='Output directory for dataset')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of parallel workers')
    parser.add_argument('--config', type=str, default=None,
                       help='Configuration file path')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from previous incomplete run')
    parser.add_argument('--preview', action='store_true',
                       help='Generate preview GIFs for each video')
    
    args = parser.parse_args()
    
    # Create generator
    generator = TwoPartsDatasetGenerator(args.output, args.config)
    
    # Determine if GIFs should be created: use config default unless --preview is explicitly set
    create_gifs = generator.config.get('output', {}).get('create_gifs', True)
    if args.preview:
        create_gifs = True  # Command line flag overrides config
    
    # Generate dataset
    generator.generate_dataset(
        num_videos=args.num_videos,
        workers=args.workers,
        resume=args.resume,
        create_gifs=create_gifs
    )


if __name__ == "__main__":
    main()