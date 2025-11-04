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
from scipy.stats import pearsonr, spearmanr, chi2_contingency
from scipy.spatial.distance import pdist, squareform
import warnings
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image
warnings.filterwarnings('ignore')

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
            'metadata': self.output_dir / 'metadata',
            'videos': self.output_dir / 'videos',
            'splits': self.output_dir / 'splits',
            'scenarios': self.output_dir / 'scenarios',
            
            # New structure matching falling_circles_v1
            'observation': self.output_dir / 'observation',
            'intervention': self.output_dir / 'intervention',
            'index': self.output_dir / 'index',
            'audits': self.output_dir / 'audits',
            'visualization': self.output_dir / 'visualization',
            'ood': self.output_dir / 'ood'
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
            
            # Create observation directory (matching falling_circles_v1 structure)
            sample_id = f"obs_{video_id:06d}_{seed % 1000000:06d}"
            obs_dir = self.dirs['observation'] / sample_id
            obs_dir.mkdir(parents=True, exist_ok=True)
            
            # Create frames subdirectory
            frames_dir = obs_dir / 'frames'
            frames_dir.mkdir(exist_ok=True)
            
            # Save frames as PNG files in frames/ subdirectory
            for frame_idx, frame in enumerate(frames):
                img = Image.fromarray(frame)
                frame_path = frames_dir / f"frame_{frame_idx:03d}.png"
                img.save(frame_path)
            
            # Generate labeled frames for GIF (with video info overlay)
            labeled_frames, _ = env.generate_video(seed=seed, include_labels=True, actual_jam_type=actual_jam_type)
            
            # Enhance metadata (matching falling_circles_v1 structure)
            enhanced_metadata = {
                'sample_id': sample_id,
                'sample_type': 'observation',
                'params': {
                    'num_frames': self.config['video_config']['num_frames'],
                    'width': params.width,
                    'height': params.height,
                    'num_circles': params.num_circles,
                    'circle_size_min': params.circle_size_min,
                    'circle_size_max': params.circle_size_max,
                    'hole_diameter': params.hole_diameter,
                    'hole_x_position': params.hole_x_position,
                    'wind_strength': params.wind_strength,
                    'wind_direction': params.wind_direction,
                    'gravity': params.gravity,
                    'spawn_rate': params.spawn_rate,
                    'circle_color': list(params.circle_color),
                    'background_color': list(params.background_color),
                    'noise_level': params.noise_level
                },
                'generation_info': {
                    'timestamp': datetime.now().isoformat(),
                    'seed': seed,
                    'actual_jam_type': actual_jam_type,
                    'generator_version': '1.0'
                },
                'video_stats': {
                    'num_frames': len(frames),
                    'frame_size': frames[0].shape if frames else None,
                    'duration_seconds': len(frames) / self.config['video_config']['fps']
                },
                'physics_simulation': metadata
            }
            
            # Save metadata as meta.json (matching falling_circles_v1 structure)
            meta_path = obs_dir / "meta.json"
            with open(meta_path, 'w') as f:
                json.dump(enhanced_metadata, f, indent=2)
            
            # Save frame-level data as stats.csv (matching falling_circles_v1 structure)
            self._save_frame_csv(obs_dir, metadata['frames'])
            
            # Generate visualization GIF
            if len(labeled_frames) > 0:
                gif_path = self.dirs['visualization'] / f"{sample_id}.gif"
                self._create_preview_gif(labeled_frames, gif_path)
            
            # Update progress
            self.completed_videos.append(video_id)
            
            return {
                'video_id': video_id,
                'sample_id': sample_id,
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
    
    def _save_frame_csv(self, obs_dir: Path, frame_data: List[Dict]):
        """Save frame-level data as stats.csv (matching falling_circles_v1 structure)"""
        csv_path = obs_dir / "stats.csv"
        
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
        
        # Generate dataset statistics and create index files
        self._create_samples_index(results)
        self._create_dataset_splits()
        self._create_dataset_statistics(results)
        self._create_audits()
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
    
    def _create_samples_index(self, results: List[Dict]):
        """Create samples.csv index file (matching falling_circles_v1 structure)"""
        successful_results = [r for r in results if r.get('status') == 'success']
        
        # Collect sample information
        samples_data = []
        for result in successful_results:
            sample_id = result.get('sample_id')
            if sample_id:
                # Read metadata to get parameters
                obs_dir = self.dirs['observation'] / sample_id
                meta_path = obs_dir / "meta.json"
                
                if meta_path.exists():
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)
                    
                    params = metadata.get('params', {})
                    circle_size_avg = (params.get('circle_size_min', 0) + params.get('circle_size_max', 0)) / 2
                    
                    samples_data.append({
                        'sample_id': sample_id,
                        'sample_type': 'observation',
                        'jam_type': result.get('jam_type', 'unknown'),
                        'exit_ratio': result.get('exit_ratio', 0.0),
                        'hole_diameter': params.get('hole_diameter', 0),
                        'wind_strength': params.get('wind_strength', 0),
                        'num_circles': params.get('num_circles', 0),
                        'circle_size_avg': circle_size_avg,
                        'spawn_rate': params.get('spawn_rate', 0),
                        'seed': metadata.get('generation_info', {}).get('seed', 0)
                    })
        
        # Save samples.csv
        if samples_data:
            samples_df_path = self.dirs['index'] / 'samples.csv'
            with open(samples_df_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'sample_id', 'sample_type', 'jam_type', 'exit_ratio',
                    'hole_diameter', 'wind_strength', 'num_circles', 
                    'circle_size_avg', 'spawn_rate', 'seed'
                ])
                writer.writeheader()
                writer.writerows(samples_data)
            
            print(f"📋 Created samples index with {len(samples_data)} entries")

    def _create_audits(self):
        """Create comprehensive audit files and analysis plots"""
        print("🔍 Creating comprehensive dataset audits...")
        
        # Load samples data for analysis
        index_file = self.dirs['index'] / 'samples.csv'
        if not index_file.exists():
            print("⚠️  No samples.csv found for audit analysis")
            return
        
        try:
            df = pd.read_csv(index_file)
        except Exception as e:
            print(f"⚠️  Error reading samples.csv: {e}")
            return
        
        # Create comprehensive audits
        self._audit_basic_stats(df)
        self._audit_parameter_independence(df)
        self._audit_jam_type_distribution(df)
        self._audit_parameter_distributions(df)
        self._create_independence_heatmap(df)
        self._audit_quality_metrics(df)
        
        print("🔍 Created comprehensive audit files and visualizations")
    
    def _audit_basic_stats(self, df: pd.DataFrame):
        """Generate basic dataset statistics"""
        stats = {
            'total_samples': len(df),
            'jam_type_counts': df['jam_type'].value_counts().to_dict(),
            'jam_type_percentages': (df['jam_type'].value_counts() / len(df) * 100).to_dict(),
            'exit_ratio_stats': {
                'mean': float(df['exit_ratio'].mean()),
                'std': float(df['exit_ratio'].std()),
                'min': float(df['exit_ratio'].min()),
                'max': float(df['exit_ratio'].max()),
                'median': float(df['exit_ratio'].median())
            },
            'parameter_stats': {}
        }
        
        # Parameter statistics
        param_columns = [col for col in df.columns if col not in ['sample_id', 'sample_type', 'jam_type']]
        for col in param_columns:
            if df[col].dtype in ['int64', 'float64']:
                stats['parameter_stats'][col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'median': float(df[col].median())
                }
        
        # Save basic stats
        with open(self.dirs['audits'] / 'basic_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
    
    def _audit_parameter_independence(self, df: pd.DataFrame):
        """Audit parameter independence using correlation analysis"""
        # Get numeric parameter columns
        param_columns = [col for col in df.columns 
                        if col not in ['sample_id', 'sample_type', 'jam_type'] 
                        and df[col].dtype in ['int64', 'float64']]
        
        if len(param_columns) < 2:
            return
        
        # Calculate correlation matrix
        param_data = df[param_columns]
        
        # Pearson correlations
        pearson_corr = param_data.corr(method='pearson')
        
        # Spearman correlations (rank-based, more robust)
        spearman_corr = param_data.corr(method='spearman')
        
        # Independence analysis
        independence_results = {
            'parameter_correlations': {
                'pearson': pearson_corr.to_dict(),
                'spearman': spearman_corr.to_dict()
            },
            'high_correlations': {},
            'independence_score': 0.0,
            'independence_assessment': ''
        }
        
        # Find high correlations (excluding diagonal)
        high_corr_threshold = 0.3
        high_correlations = []
        
        for i, param1 in enumerate(param_columns):
            for j, param2 in enumerate(param_columns):
                if i < j:  # Avoid duplicates and diagonal
                    pearson_val = abs(pearson_corr.loc[param1, param2])
                    spearman_val = abs(spearman_corr.loc[param1, param2])
                    
                    if pearson_val > high_corr_threshold or spearman_val > high_corr_threshold:
                        high_correlations.append({
                            'param1': param1,
                            'param2': param2,
                            'pearson_correlation': float(pearson_val),
                            'spearman_correlation': float(spearman_val)
                        })
        
        independence_results['high_correlations'] = high_correlations
        
        # Calculate independence score (1.0 = fully independent, 0.0 = highly dependent)
        all_correlations = []
        for i, param1 in enumerate(param_columns):
            for j, param2 in enumerate(param_columns):
                if i < j:
                    all_correlations.append(abs(pearson_corr.loc[param1, param2]))
        
        if all_correlations:
            avg_correlation = np.mean(all_correlations)
            independence_score = max(0.0, 1.0 - avg_correlation * 2)  # Scale to 0-1
            independence_results['independence_score'] = float(independence_score)
            
            if independence_score > 0.8:
                assessment = "Excellent - Parameters are highly independent"
            elif independence_score > 0.6:
                assessment = "Good - Parameters show low correlation"
            elif independence_score > 0.4:
                assessment = "Moderate - Some parameter dependencies detected"
            else:
                assessment = "Poor - Significant parameter dependencies detected"
            
            independence_results['independence_assessment'] = assessment
        
        # Save independence analysis
        with open(self.dirs['audits'] / 'parameter_independence.json', 'w') as f:
            json.dump(independence_results, f, indent=2)
        
        # Create independence matrix CSV
        independence_matrix_data = []
        for i, param1 in enumerate(param_columns):
            for j, param2 in enumerate(param_columns):
                if i <= j:  # Include diagonal and upper triangle
                    pearson_val = pearson_corr.loc[param1, param2]
                    spearman_val = spearman_corr.loc[param1, param2]
                    
                    independence_matrix_data.append({
                        'parameter_1': param1,
                        'parameter_2': param2,
                        'pearson_correlation': float(pearson_val),
                        'spearman_correlation': float(spearman_val),
                        'abs_pearson': float(abs(pearson_val)),
                        'independence_score': float(1.0 - abs(pearson_val))
                    })
        
        # Save as CSV
        with open(self.dirs['audits'] / 'independence_matrix.csv', 'w', newline='') as f:
            if independence_matrix_data:
                writer = csv.DictWriter(f, fieldnames=independence_matrix_data[0].keys())
                writer.writeheader()
                writer.writerows(independence_matrix_data)
    
    def _audit_jam_type_distribution(self, df: pd.DataFrame):
        """Analyze jam type distribution and exit ratio patterns"""
        jam_analysis = {
            'jam_type_distribution': df['jam_type'].value_counts().to_dict(),
            'exit_ratio_by_jam_type': {},
            'jam_type_thresholds_validation': {},
            'natural_distribution_score': 0.0
        }
        
        # Exit ratio statistics by jam type
        for jam_type in df['jam_type'].unique():
            jam_data = df[df['jam_type'] == jam_type]['exit_ratio']
            jam_analysis['exit_ratio_by_jam_type'][jam_type] = {
                'count': len(jam_data),
                'mean': float(jam_data.mean()),
                'std': float(jam_data.std()),
                'min': float(jam_data.min()),
                'max': float(jam_data.max()),
                'median': float(jam_data.median())
            }
        
        # Validate jam type thresholds
        thresholds = {
            'no_jam': (0.9, 1.0),
            'partial_jam': (0.3, 0.9),
            'full_jam': (0.0, 0.3)
        }
        
        for jam_type, (min_thresh, max_thresh) in thresholds.items():
            jam_data = df[df['jam_type'] == jam_type]['exit_ratio']
            if len(jam_data) > 0:
                within_range = ((jam_data >= min_thresh) & (jam_data < max_thresh)).sum()
                total = len(jam_data)
                accuracy = within_range / total
                jam_analysis['jam_type_thresholds_validation'][jam_type] = {
                    'total_samples': total,
                    'within_threshold': int(within_range),
                    'accuracy': float(accuracy),
                    'expected_range': f"{min_thresh}-{max_thresh}"
                }
        
        # Calculate natural distribution score
        counts = df['jam_type'].value_counts()
        total = len(df)
        proportions = counts / total
        
        # Ideal would be somewhat balanced but natural (not forced 33/33/33)
        # Score based on how "natural" the distribution looks
        if len(proportions) == 3:
            # Penalize extreme imbalances, reward natural-looking distributions
            min_prop = proportions.min()
            max_prop = proportions.max()
            balance_score = 1.0 - (max_prop - min_prop)  # Higher when more balanced
            
            # Also consider if we have at least some of each type
            presence_score = 1.0 if min_prop > 0.05 else min_prop * 20  # Penalize if <5% of any type
            
            natural_score = (balance_score + presence_score) / 2
            jam_analysis['natural_distribution_score'] = float(max(0.0, min(1.0, natural_score)))
        
        # Save jam type analysis
        with open(self.dirs['audits'] / 'jam_type_analysis.json', 'w') as f:
            json.dump(jam_analysis, f, indent=2)
    
    def _audit_parameter_distributions(self, df: pd.DataFrame):
        """Analyze parameter distributions for uniformity and coverage"""
        param_columns = [col for col in df.columns 
                        if col not in ['sample_id', 'sample_type', 'jam_type'] 
                        and df[col].dtype in ['int64', 'float64']]
        
        distribution_analysis = {
            'parameter_distributions': {},
            'uniformity_scores': {},
            'coverage_analysis': {}
        }
        
        for param in param_columns:
            data = df[param]
            
            # Basic distribution stats
            distribution_analysis['parameter_distributions'][param] = {
                'mean': float(data.mean()),
                'std': float(data.std()),
                'min': float(data.min()),
                'max': float(data.max()),
                'median': float(data.median()),
                'skewness': float(data.skew()),
                'kurtosis': float(data.kurtosis())
            }
            
            # Uniformity analysis (how close to uniform distribution)
            hist, _ = np.histogram(data, bins=10)
            expected_uniform = len(data) / 10
            chi_sq = np.sum((hist - expected_uniform) ** 2 / expected_uniform)
            uniformity_score = max(0.0, 1.0 - chi_sq / (len(data) * 2))  # Normalize
            
            distribution_analysis['uniformity_scores'][param] = {
                'score': float(uniformity_score),
                'interpretation': 'Highly uniform' if uniformity_score > 0.8 else
                               'Moderately uniform' if uniformity_score > 0.6 else
                               'Non-uniform distribution'
            }
            
            # Coverage analysis (how well we cover the parameter space)
            param_range = data.max() - data.min()
            if param_range > 0:
                coverage_bins = 20
                hist, bin_edges = np.histogram(data, bins=coverage_bins)
                empty_bins = np.sum(hist == 0)
                coverage_score = 1.0 - (empty_bins / coverage_bins)
                
                distribution_analysis['coverage_analysis'][param] = {
                    'coverage_score': float(coverage_score),
                    'empty_bins': int(empty_bins),
                    'total_bins': coverage_bins,
                    'interpretation': 'Excellent coverage' if coverage_score > 0.9 else
                                   'Good coverage' if coverage_score > 0.7 else
                                   'Poor coverage - gaps in parameter space'
                }
        
        # Save distribution analysis
        with open(self.dirs['audits'] / 'parameter_distributions.json', 'w') as f:
            json.dump(distribution_analysis, f, indent=2)
    
    def _create_independence_heatmap(self, df: pd.DataFrame):
        """Create comprehensive independence heatmap visualization"""
        # Get numeric parameter columns
        param_columns = [col for col in df.columns 
                        if col not in ['sample_id', 'sample_type', 'jam_type'] 
                        and df[col].dtype in ['int64', 'float64']]
        
        if len(param_columns) < 2:
            print("⚠️  Not enough numeric parameters for correlation heatmap")
            return
        
        # Prepare data
        param_data = df[param_columns]
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Dataset Parameter Independence Analysis', fontsize=16, fontweight='bold')
        
        # 1. Pearson correlation heatmap
        pearson_corr = param_data.corr(method='pearson')
        sns.heatmap(pearson_corr, annot=True, cmap='RdBu_r', center=0, 
                   square=True, fmt='.2f', cbar_kws={'label': 'Pearson Correlation'},
                   ax=ax1)
        ax1.set_title('Pearson Correlations\n(Linear Relationships)', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.tick_params(axis='y', rotation=0)
        
        # 2. Spearman correlation heatmap
        spearman_corr = param_data.corr(method='spearman')
        sns.heatmap(spearman_corr, annot=True, cmap='RdBu_r', center=0,
                   square=True, fmt='.2f', cbar_kws={'label': 'Spearman Correlation'},
                   ax=ax2)
        ax2.set_title('Spearman Correlations\n(Rank-based, Non-linear)', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.tick_params(axis='y', rotation=0)
        
        # 3. Parameter distribution plot
        # Select a few key parameters for distribution visualization
        key_params = param_columns[:min(4, len(param_columns))]
        for i, param in enumerate(key_params):
            ax3.hist(param_data[param], alpha=0.6, label=param, bins=20)
        ax3.set_title('Parameter Distributions\n(Overlaid)', fontweight='bold')
        ax3.set_xlabel('Parameter Values (Normalized)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        
        # 4. Independence score summary
        ax4.axis('off')
        
        # Calculate independence metrics
        all_correlations = []
        high_corr_pairs = []
        
        for i, param1 in enumerate(param_columns):
            for j, param2 in enumerate(param_columns):
                if i < j:
                    pearson_val = abs(pearson_corr.loc[param1, param2])
                    spearman_val = abs(spearman_corr.loc[param1, param2])
                    all_correlations.append(pearson_val)
                    
                    if pearson_val > 0.3 or spearman_val > 0.3:
                        high_corr_pairs.append(f"{param1} ↔ {param2}: r={pearson_val:.2f}")
        
        avg_correlation = np.mean(all_correlations) if all_correlations else 0
        independence_score = max(0.0, 1.0 - avg_correlation * 2)
        
        # Summary text
        summary_text = f"""Independence Analysis Summary
        
📊 Dataset Size: {len(df):,} samples
🔢 Parameters Analyzed: {len(param_columns)}
📈 Average Correlation: {avg_correlation:.3f}
🎯 Independence Score: {independence_score:.3f}/1.0

🔍 Assessment: {
    'Excellent - Highly Independent' if independence_score > 0.8 else
    'Good - Low Correlation' if independence_score > 0.6 else
    'Moderate - Some Dependencies' if independence_score > 0.4 else
    'Poor - High Dependencies'
}

⚠️ High Correlations (>0.3):
{chr(10).join(high_corr_pairs[:5]) if high_corr_pairs else 'None detected'}
{"..." if len(high_corr_pairs) > 5 else ""}
        """
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save heatmap
        heatmap_path = self.dirs['audits'] / 'independence_heatmap.png'
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Independence heatmap saved: {heatmap_path}")
    
    def _audit_quality_metrics(self, df: pd.DataFrame):
        """Generate overall dataset quality metrics"""
        quality_metrics = {
            'dataset_quality_score': 0.0,
            'quality_breakdown': {},
            'recommendations': [],
            'timestamp': datetime.now().isoformat()
        }
        
        scores = {}
        
        # 1. Sample count adequacy
        sample_count = len(df)
        if sample_count >= 1000:
            scores['sample_size'] = 1.0
        elif sample_count >= 100:
            scores['sample_size'] = 0.8
        elif sample_count >= 50:
            scores['sample_size'] = 0.6
        else:
            scores['sample_size'] = 0.3
            quality_metrics['recommendations'].append("Increase sample size for better statistical power")
        
        # 2. Jam type balance
        jam_counts = df['jam_type'].value_counts()
        min_count = jam_counts.min()
        max_count = jam_counts.max()
        balance_ratio = min_count / max_count if max_count > 0 else 0
        scores['jam_type_balance'] = balance_ratio
        
        if balance_ratio < 0.2:
            quality_metrics['recommendations'].append("Improve jam type balance - some types are underrepresented")
        
        # 3. Parameter independence (from previous analysis)
        try:
            with open(self.dirs['audits'] / 'parameter_independence.json', 'r') as f:
                independence_data = json.load(f)
                scores['parameter_independence'] = independence_data.get('independence_score', 0.5)
        except:
            scores['parameter_independence'] = 0.5
        
        # 4. Exit ratio validity
        valid_ratios = ((df['exit_ratio'] >= 0) & (df['exit_ratio'] <= 1)).sum()
        scores['exit_ratio_validity'] = valid_ratios / len(df)
        
        # 5. Parameter coverage (from distribution analysis)
        try:
            with open(self.dirs['audits'] / 'parameter_distributions.json', 'r') as f:
                dist_data = json.load(f)
                coverage_scores = [data['coverage_score'] for data in dist_data['coverage_analysis'].values()]
                scores['parameter_coverage'] = np.mean(coverage_scores) if coverage_scores else 0.5
        except:
            scores['parameter_coverage'] = 0.5
        
        # Calculate overall quality score
        weights = {
            'sample_size': 0.2,
            'jam_type_balance': 0.2,
            'parameter_independence': 0.3,
            'exit_ratio_validity': 0.15,
            'parameter_coverage': 0.15
        }
        
        overall_score = sum(scores[metric] * weights[metric] for metric in scores.keys())
        quality_metrics['dataset_quality_score'] = float(overall_score)
        quality_metrics['quality_breakdown'] = {k: float(v) for k, v in scores.items()}
        
        # Add quality assessment
        if overall_score >= 0.9:
            quality_metrics['overall_assessment'] = "Excellent - High quality dataset"
        elif overall_score >= 0.8:
            quality_metrics['overall_assessment'] = "Good - Dataset meets quality standards"
        elif overall_score >= 0.7:
            quality_metrics['overall_assessment'] = "Acceptable - Some quality issues detected"
        else:
            quality_metrics['overall_assessment'] = "Poor - Significant quality issues"
            quality_metrics['recommendations'].append("Consider regenerating dataset with improved parameters")
        
        # Save quality metrics
        with open(self.dirs['audits'] / 'quality_metrics.json', 'w') as f:
            json.dump(quality_metrics, f, indent=2)
        
        # Create label sensitivity report
        self._create_label_sensitivity_report(df)
    
    def _create_label_sensitivity_report(self, df: pd.DataFrame):
        """Create comprehensive label sensitivity analysis report"""
        report_content = []
        report_content.append("Dataset Label Sensitivity Analysis")
        report_content.append("=" * 50)
        report_content.append(f"Generated: {datetime.now().isoformat()}")
        report_content.append(f"Total samples analyzed: {len(df)}")
        report_content.append("")
        
        # Jam type distribution
        report_content.append("JAM TYPE DISTRIBUTION:")
        jam_counts = df['jam_type'].value_counts()
        for jam_type, count in jam_counts.items():
            percentage = (count / len(df)) * 100
            report_content.append(f"  {jam_type}: {count} samples ({percentage:.1f}%)")
        report_content.append("")
        
        # Exit ratio thresholds validation
        report_content.append("EXIT RATIO THRESHOLD VALIDATION:")
        thresholds = {
            'no_jam': (0.9, 1.0),
            'partial_jam': (0.3, 0.9), 
            'full_jam': (0.0, 0.3)
        }
        
        for jam_type, (min_thresh, max_thresh) in thresholds.items():
            jam_data = df[df['jam_type'] == jam_type]['exit_ratio']
            if len(jam_data) > 0:
                within_range = ((jam_data >= min_thresh) & (jam_data < max_thresh)).sum()
                accuracy = within_range / len(jam_data) * 100
                report_content.append(f"  {jam_type} ({min_thresh}-{max_thresh}):")
                report_content.append(f"    Accuracy: {accuracy:.1f}% ({within_range}/{len(jam_data)})")
                report_content.append(f"    Exit ratio range: {jam_data.min():.3f} - {jam_data.max():.3f}")
                report_content.append(f"    Mean exit ratio: {jam_data.mean():.3f} ± {jam_data.std():.3f}")
        report_content.append("")
        
        # Parameter sensitivity analysis
        report_content.append("PARAMETER SENSITIVITY TO JAM TYPE:")
        param_columns = [col for col in df.columns 
                        if col not in ['sample_id', 'sample_type', 'jam_type'] 
                        and df[col].dtype in ['int64', 'float64']]
        
        for param in param_columns:
            report_content.append(f"\n  {param.upper()}:")
            
            # Statistics by jam type
            for jam_type in df['jam_type'].unique():
                jam_data = df[df['jam_type'] == jam_type][param]
                if len(jam_data) > 0:
                    report_content.append(f"    {jam_type}: {jam_data.mean():.3f} ± {jam_data.std():.3f} "
                                        f"(range: {jam_data.min():.3f}-{jam_data.max():.3f})")
            
            # ANOVA F-statistic for parameter sensitivity
            try:
                from scipy.stats import f_oneway
                groups = [df[df['jam_type'] == jt][param].values for jt in df['jam_type'].unique()]
                f_stat, p_value = f_oneway(*groups)
                sensitivity = "High" if p_value < 0.01 else "Medium" if p_value < 0.05 else "Low"
                report_content.append(f"    Sensitivity: {sensitivity} (F={f_stat:.2f}, p={p_value:.4f})")
            except:
                report_content.append(f"    Sensitivity: Unable to calculate")
        
        report_content.append("")
        report_content.append("CLASSIFICATION STABILITY:")
        report_content.append("  Method: Exit ratio based")
        report_content.append("  Thresholds: no_jam≥90%, partial_jam=30-90%, full_jam<30%")
        
        # Calculate how many samples are near boundaries
        boundary_tolerance = 0.05  # 5% tolerance around boundaries
        near_boundaries = 0
        
        for _, row in df.iterrows():
            exit_ratio = row['exit_ratio']
            if (abs(exit_ratio - 0.3) < boundary_tolerance or 
                abs(exit_ratio - 0.9) < boundary_tolerance):
                near_boundaries += 1
        
        boundary_percentage = (near_boundaries / len(df)) * 100
        report_content.append(f"  Samples near boundaries (±5%): {near_boundaries} ({boundary_percentage:.1f}%)")
        
        stability_score = 1.0 - (boundary_percentage / 100)
        stability_assessment = ("High" if stability_score > 0.9 else
                              "Medium" if stability_score > 0.7 else "Low")
        report_content.append(f"  Classification stability: {stability_assessment} (score: {stability_score:.3f})")
        
        report_content.append("")
        report_content.append("RECOMMENDATIONS:")
        
        # Generate recommendations based on analysis
        if boundary_percentage > 20:
            report_content.append("  - High number of samples near classification boundaries")
            report_content.append("    Consider adjusting thresholds or adding buffer zones")
        
        if len(df) < 100:
            report_content.append("  - Small dataset size may affect reliability")
            report_content.append("    Consider generating more samples for robust analysis")
        
        # Check for severe imbalances
        min_count = jam_counts.min()
        max_count = jam_counts.max()
        if min_count / max_count < 0.2:
            report_content.append("  - Severe class imbalance detected")
            report_content.append("    Consider stratified sampling or class balancing")
        
        # Save the report
        report_path = self.dirs['audits'] / 'label_sensitivity_report.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_content))
        
        print(f"📋 Label sensitivity report saved: {report_path}")

    def _organize_by_scenarios(self):
        """Create scenario organization (updated for new structure)"""
        # For the new structure, scenarios are organized in observation/intervention directories
        # This method can be used for additional scenario-based organization if needed
        scenario_stats = {}
        
        # Count samples by jam type
        for obs_dir in self.dirs['observation'].iterdir():
            if obs_dir.is_dir():
                meta_path = obs_dir / "meta.json"
                if meta_path.exists():
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)
                    
                    jam_type = metadata.get('generation_info', {}).get('actual_jam_type', 'unknown')
                    scenario_stats[jam_type] = scenario_stats.get(jam_type, 0) + 1
        
        # Save scenario statistics
        scenario_path = self.dirs['metadata'] / 'scenario_distribution.json'
        with open(scenario_path, 'w') as f:
            json.dump(scenario_stats, f, indent=2)
        
        print(f"📊 Scenario distribution: {scenario_stats}")

    def _create_dataset_splits(self):
        """Create train/validation/test splits (matching falling_circles_v1 structure)"""
        # Get all observation sample IDs
        sample_ids = []
        for obs_dir in self.dirs['observation'].iterdir():
            if obs_dir.is_dir():
                sample_ids.append(obs_dir.name)
        
        # Check if no samples were completed
        if len(sample_ids) == 0:
            print("⚠️  No completed samples to create splits for")
            # Create empty split files
            split_files = {
                'train_ids.txt': [],
                'val_ids.txt': [],
                'test_ids.txt': [],
                'test_ood_ids.txt': [],  # Out-of-distribution test set
                'train_pair_ids.txt': [],  # For intervention pairs (empty for observation-only)
                'val_pair_ids.txt': [],
                'test_pair_ids.txt': []
            }
            
            for filename, ids in split_files.items():
                split_file = self.dirs['splits'] / filename
                with open(split_file, 'w') as f:
                    for sample_id in ids:
                        f.write(f"{sample_id}\n")
            
            # Save empty splits metadata
            splits_meta = {
                'total_samples': 0,
                'splits': {name.replace('_ids.txt', ''): 0 for name in split_files.keys()},
                'split_ratios': {name.replace('_ids.txt', ''): 0.0 for name in split_files.keys()}
            }
            
            with open(self.dirs['metadata'] / 'splits_info.json', 'w') as f:
                json.dump(splits_meta, f, indent=2)
            return
        
        random.shuffle(sample_ids)
        
        # 70% train, 15% validation, 15% test
        train_size = int(0.7 * len(sample_ids))
        val_size = int(0.15 * len(sample_ids))
        
        splits = {
            'train_ids': sample_ids[:train_size],
            'val_ids': sample_ids[train_size:train_size + val_size],
            'test_ids': sample_ids[train_size + val_size:],
            'test_ood_ids': [],  # Could be populated with specific OOD samples
            'train_pair_ids': [],  # Empty for observation-only dataset
            'val_pair_ids': [],
            'test_pair_ids': []
        }
        
        # Save splits
        for split_name, sample_list in splits.items():
            split_file = self.dirs['splits'] / f"{split_name}.txt"
            with open(split_file, 'w') as f:
                for sample_id in sample_list:
                    f.write(f"{sample_id}\n")
        
        # Save splits metadata
        splits_meta = {
            'total_samples': len(sample_ids),
            'splits': {name.replace('_ids', ''): len(ids) for name, ids in splits.items()},
            'split_ratios': {
                'train': len(splits['train_ids']) / len(sample_ids),
                'val': len(splits['val_ids']) / len(sample_ids),
                'test': len(splits['test_ids']) / len(sample_ids),
                'test_ood': 0.0,
                'train_pair': 0.0,
                'val_pair': 0.0,
                'test_pair': 0.0
            }
        }
        
        with open(self.dirs['metadata'] / 'splits_info.json', 'w') as f:
            json.dump(splits_meta, f, indent=2)
        
        print(f"✅ Created dataset splits with {len(sample_ids)} samples:")
        for split_name, count in splits_meta['splits'].items():
            if count > 0:
                ratio = splits_meta['split_ratios'][split_name]
                print(f"   {split_name}: {count} samples ({ratio:.1%})")
    
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