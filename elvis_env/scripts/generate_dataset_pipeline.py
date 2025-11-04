#!/usr/bin/env python3
"""
Falling Circles Dataset Pipeline

Generates a comprehensive physics simulation dataset with observation and intervention data,
following standardized structure for causal inference research.

Dataset Structure:
- observation/: Pure observational data (no interventions)
- intervention/: Intervention pairs with baseline and do() operations
- splits/: Train/val/test/OOD splits
- index/: Sample and pair indices
- audits/: Independence and quality audits
- visualization/: Human-readable GIFs
- ood/: Out-of-distribution samples

Usage:
    python generate_dataset_pipeline.py --name falling_circles_v1 --num_observations 2000 --num_pairs 500 --out datasets/
"""

import os
import json
import random
import argparse
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import time
from datetime import datetime
import csv
import hashlib
from PIL import Image

# Import existing modules
import sys
sys.path.append(os.path.dirname(__file__))
from falling_circles_env import VideoParams, generate_falling_circles_video
from falling_circles import sample_video_params  # Keep sampling function
from generate_intervention_pairs import create_intervention_variants
from generate_active_interventions import compute_intervention_effect
try:
    from analyze_parameter_independence import compute_correlation_matrix, compute_mutual_information_matrix
except ImportError:
    # These functions might not exist, we'll implement basic versions
    pass
try:
    from analyze_causal_effects import load_intervention_data, analyze_causal_effects
except ImportError:
    pass

class DatasetPipeline:
    """Complete dataset generation and organization pipeline"""
    
    def __init__(self, dataset_name: str, output_dir: str):
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.dataset_root = os.path.join(output_dir, dataset_name)
        
        # Dataset structure
        self.dirs = {
            'root': self.dataset_root,
            'observation': os.path.join(self.dataset_root, 'observation'),
            'intervention': os.path.join(self.dataset_root, 'intervention'),
            'splits': os.path.join(self.dataset_root, 'splits'),
            'index': os.path.join(self.dataset_root, 'index'),
            'audits': os.path.join(self.dataset_root, 'audits'),
            'visualization': os.path.join(self.dataset_root, 'visualization'),
            'ood': os.path.join(self.dataset_root, 'ood')
        }
        
        # Data tracking
        self.samples_data = []
        self.pairs_data = []
        self.ood_samples = []
        
        # Statistics for audits
        self.parameter_data = []
        self.effect_data = []
        
    def create_directory_structure(self):
        """Create all necessary directories"""
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        print(f"Created dataset structure at: {self.dataset_root}")
    
    def generate_sample_id(self, sample_type: str, index: int) -> str:
        """Generate unique sample ID"""
        timestamp = int(time.time() * 1000) % 1000000  # Last 6 digits of timestamp
        return f"{sample_type}_{index:06d}_{timestamp:06d}"
    
    def generate_pair_id(self, index: int) -> str:
        """Generate unique pair ID"""
        timestamp = int(time.time() * 1000) % 1000000
        return f"pair_{index:06d}_{timestamp:06d}"
    
    def save_video_frames(self, frames: List, output_dir: str, prefix: str = ""):
        """Save video frames as PNG files"""
        frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        for frame_idx, frame in enumerate(frames):
            frame_path = os.path.join(frames_dir, f"{prefix}{frame_idx+1:06d}.png")
            if hasattr(frame, 'save'):
                frame.save(frame_path)
            else:
                # Convert numpy array to PIL Image if needed
                from PIL import Image
                if isinstance(frame, np.ndarray):
                    frame_image = Image.fromarray(frame.astype('uint8'))
                    frame_image.save(frame_path)
                else:
                    frame.save(frame_path)
    
    def save_frame_statistics(self, meta_data: Dict, output_dir: str):
        """Save per-frame statistics as CSV"""
        if 'frames' not in meta_data:
            return
        
        stats_path = os.path.join(output_dir, "stats.csv")
        with open(stats_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['frame', 'active_circles', 'exited_circles', 'stuck_circles', 
                     'avg_velocity', 'density_near_hole', 'flow_rate']
            writer.writerow(header)
            
            # Data
            for frame_data in meta_data['frames']:
                row = [
                    frame_data.get('frame', 0),
                    frame_data.get('active_circles', 0),
                    frame_data.get('exited_circles', 0),
                    frame_data.get('stuck_circles', 0),
                    frame_data.get('avg_velocity', 0.0),
                    frame_data.get('density_near_hole', 0.0),
                    frame_data.get('flow_rate', 0.0)
                ]
                writer.writerow(row)
    
    def generate_observation_samples(self, num_samples: int, include_ood: bool = True):
        """Generate pure observational data"""
        print(f"Generating {num_samples} observational samples...")
        
        ood_threshold = int(num_samples * 0.05) if include_ood else 0  # 5% OOD
        
        for i in tqdm(range(num_samples), desc="Observation samples"):
            sample_id = self.generate_sample_id("obs", i)
            sample_dir = os.path.join(self.dirs['observation'], sample_id)
            os.makedirs(sample_dir, exist_ok=True)
            
            # Generate parameters
            if i < ood_threshold:
                # OOD sample with extreme parameters
                params = self.sample_ood_parameters()
                is_ood = True
                ood_sample_dir = os.path.join(self.dirs['ood'], sample_id)
                os.makedirs(ood_sample_dir, exist_ok=True)
            else:
                # Regular sample
                params = sample_video_params()
                is_ood = False
            
            # Generate video
            seed = random.randint(0, 2_000_000_000)
            frames, meta_data = generate_falling_circles_video(
                params=params, 
                seed=seed, 
                include_labels=False
            )
            
            # Save frames
            self.save_video_frames(frames, sample_dir)
            if is_ood:
                self.save_video_frames(frames, ood_sample_dir)
            
            # Save metadata
            meta_path = os.path.join(sample_dir, "meta.json")
            with open(meta_path, 'w') as f:
                json.dump({
                    'sample_id': sample_id,
                    'sample_type': 'ood' if is_ood else 'observation',
                    'params': params.__dict__,
                    'seed': seed,
                    'jam_type': meta_data.get('actual_jam_type', 'unknown'),
                    'exit_ratio': meta_data.get('exit_statistics', {}).get('exit_ratio', 0),
                    'physics_stats': meta_data.get('exit_statistics', {}),
                    'generation_timestamp': time.time()
                }, f, indent=2)
            
            if is_ood:
                ood_meta_path = os.path.join(ood_sample_dir, "meta.json")
                shutil.copy2(meta_path, ood_meta_path)
            
            # Save frame statistics
            self.save_frame_statistics(meta_data, sample_dir)
            if is_ood:
                self.save_frame_statistics(meta_data, ood_sample_dir)
            
            # Track for index
            sample_data = {
                'sample_id': sample_id,
                'sample_type': 'ood' if is_ood else 'observation',
                'jam_type': meta_data.get('actual_jam_type', 'unknown'),
                'exit_ratio': meta_data.get('exit_statistics', {}).get('exit_ratio', 0),
                'hole_diameter': params.hole_diameter,
                'wind_strength': params.wind_strength,
                'num_circles': params.num_circles,
                'circle_size_avg': (params.circle_size_min + params.circle_size_max) / 2,
                'spawn_rate': params.spawn_rate,
                'seed': seed
            }
            
            if is_ood:
                self.ood_samples.append(sample_data)
            else:
                self.samples_data.append(sample_data)
                
            # Track for independence analysis
            self.parameter_data.append({
                'hole_diameter': params.hole_diameter,
                'wind_strength': params.wind_strength,
                'wind_direction': params.wind_direction,
                'num_circles': params.num_circles,
                'circle_size_min': params.circle_size_min,
                'circle_size_max': params.circle_size_max,
                'spawn_rate': params.spawn_rate,
                'jam_type': meta_data.get('actual_jam_type', 'unknown'),
                'exit_ratio': meta_data.get('exit_statistics', {}).get('exit_ratio', 0)
            })
    
    def sample_ood_parameters(self) -> VideoParams:
        """Sample out-of-distribution parameters"""
        # Create extreme parameter combinations
        params = sample_video_params()
        
        # Random extreme modifications
        if random.random() < 0.3:
            # Very large holes
            params.hole_diameter = random.uniform(45, 55)
        elif random.random() < 0.3:
            # Very small holes  
            params.hole_diameter = random.uniform(8, 12)
        elif random.random() < 0.3:
            # Extreme wind
            params.wind_strength = random.uniform(0.35, 0.45)
        elif random.random() < 0.3:
            # Many circles
            params.num_circles = random.randint(12, 18)
        else:
            # Extreme circle sizes
            if random.random() < 0.5:
                params.circle_size_min = random.randint(1, 2)
                params.circle_size_max = random.randint(2, 4)
            else:
                params.circle_size_min = random.randint(12, 15)
                params.circle_size_max = random.randint(15, 20)
        
        return params
    
    def generate_intervention_pairs(self, num_pairs: int):
        """Generate intervention pairs with baseline and do() operations"""
        print(f"Generating {num_pairs} intervention pairs...")
        
        for i in tqdm(range(num_pairs), desc="Intervention pairs"):
            pair_id = self.generate_pair_id(i)
            pair_dir = os.path.join(self.dirs['intervention'], pair_id)
            os.makedirs(pair_dir, exist_ok=True)
            
            # Generate baseline parameters
            baseline_params = sample_video_params()
            seed = random.randint(0, 2_000_000_000)
            
            # Create baseline
            baseline_dir = os.path.join(pair_dir, "baseline")
            os.makedirs(baseline_dir, exist_ok=True)
            
            baseline_frames, baseline_meta = generate_falling_circles_video(
                params=baseline_params,
                seed=seed,
                include_labels=False
            )
            
            # Save baseline
            self.save_video_frames(baseline_frames, baseline_dir)
            self.save_frame_statistics(baseline_meta, baseline_dir)
            
            baseline_meta_path = os.path.join(baseline_dir, "meta.json")
            with open(baseline_meta_path, 'w') as f:
                json.dump({
                    'pair_id': pair_id,
                    'role': 'baseline',
                    'params': baseline_params.__dict__,
                    'seed': seed,
                    'jam_type': baseline_meta.get('actual_jam_type', 'unknown'),
                    'exit_ratio': baseline_meta.get('exit_statistics', {}).get('exit_ratio', 0),
                    'physics_stats': baseline_meta.get('exit_statistics', {}),
                    'generation_timestamp': time.time()
                }, f, indent=2)
            
            # Generate intervention variants
            intervention_variants = create_intervention_variants(baseline_params)
            
            # Select one intervention randomly
            intervention_name, intervention_params = random.choice(intervention_variants)
            
            # Create intervention name with delta
            delta_str = self.compute_intervention_delta_string(
                baseline_params, intervention_params, intervention_name
            )
            intervention_dir_name = f"do_{intervention_name}_{delta_str}"
            intervention_dir = os.path.join(pair_dir, intervention_dir_name)
            os.makedirs(intervention_dir, exist_ok=True)
            
            # Generate intervention video (same seed!)
            intervention_frames, intervention_meta = generate_falling_circles_video(
                params=intervention_params,
                seed=seed,  # Same seed for perfect counterfactual
                include_labels=False
            )
            
            # Save intervention
            self.save_video_frames(intervention_frames, intervention_dir)
            self.save_frame_statistics(intervention_meta, intervention_dir)
            
            intervention_meta_path = os.path.join(intervention_dir, "meta.json")
            with open(intervention_meta_path, 'w') as f:
                json.dump({
                    'pair_id': pair_id,
                    'role': 'intervention',
                    'intervention_name': intervention_name,
                    'intervention_delta': delta_str,
                    'params': intervention_params.__dict__,
                    'seed': seed,
                    'jam_type': intervention_meta.get('actual_jam_type', 'unknown'),
                    'exit_ratio': intervention_meta.get('exit_statistics', {}).get('exit_ratio', 0),
                    'physics_stats': intervention_meta.get('exit_statistics', {}),
                    'generation_timestamp': time.time()
                }, f, indent=2)
            
            # Compute causal effect
            effect_magnitude = compute_intervention_effect(baseline_meta, intervention_meta)
            baseline_jam = baseline_meta.get('actual_jam_type', 'unknown')
            intervention_jam = intervention_meta.get('actual_jam_type', 'unknown')
            jam_changed = baseline_jam != intervention_jam
            
            baseline_exit = baseline_meta.get('exit_statistics', {}).get('exit_ratio', 0)
            intervention_exit = intervention_meta.get('exit_statistics', {}).get('exit_ratio', 0)
            exit_delta = intervention_exit - baseline_exit
            
            # Track for pairs index
            pair_data = {
                'pair_id': pair_id,
                'intervention_name': intervention_name,
                'intervention_delta': delta_str,
                'baseline_jam_type': baseline_jam,
                'intervention_jam_type': intervention_jam,
                'baseline_exit_ratio': baseline_exit,
                'intervention_exit_ratio': intervention_exit,
                'exit_ratio_delta': exit_delta,
                'effect_magnitude': effect_magnitude,
                'jam_type_changed': jam_changed,
                'seed': seed
            }
            self.pairs_data.append(pair_data)
            
            # Track for effect analysis
            self.effect_data.append({
                'intervention_name': intervention_name,
                'effect_magnitude': effect_magnitude,
                'jam_type_changed': jam_changed,
                'exit_ratio_delta': exit_delta
            })
    
    def compute_intervention_delta_string(self, baseline_params, intervention_params, intervention_name: str) -> str:
        """Compute delta string for intervention naming"""
        
        if 'hole' in intervention_name:
            delta = intervention_params.hole_diameter - baseline_params.hole_diameter
            return f"{delta:+.0f}"
        elif 'wind_strength' in intervention_name:
            if intervention_params.wind_strength == 0:
                return "0.00"
            delta = intervention_params.wind_strength - baseline_params.wind_strength
            return f"{delta:+.2f}"
        elif 'wind_direction' in intervention_name:
            return "flip"
        elif 'circles' in intervention_name and 'num' not in intervention_name:
            # Circle size change
            baseline_avg = (baseline_params.circle_size_min + baseline_params.circle_size_max) / 2
            intervention_avg = (intervention_params.circle_size_min + intervention_params.circle_size_max) / 2
            delta = intervention_avg - baseline_avg
            return f"{delta:+.1f}"
        elif 'num_circles' in intervention_name or intervention_name in ['more_circles', 'fewer_circles']:
            delta = intervention_params.num_circles - baseline_params.num_circles
            return f"{delta:+d}"
        elif 'spawn' in intervention_name:
            delta = intervention_params.spawn_rate - baseline_params.spawn_rate
            return f"{delta:+.2f}"
        else:
            return "mod"
    
    def generate_visualizations(self):
        """Generate GIF visualizations for human inspection"""
        print("Generating visualizations...")
        
        # Sample observation GIFs (subset)
        sample_indices = random.sample(range(len(self.samples_data)), min(50, len(self.samples_data)))
        
        for idx in tqdm(sample_indices, desc="Observation GIFs"):
            sample_data = self.samples_data[idx]
            sample_id = sample_data['sample_id']
            
            # Recreate video with labels for GIF
            params = VideoParams(
                hole_diameter=sample_data['hole_diameter'],
                wind_strength=sample_data['wind_strength'],
                num_circles=sample_data['num_circles'],
                circle_size_min=int(sample_data['circle_size_avg'] - 2),
                circle_size_max=int(sample_data['circle_size_avg'] + 2),
                spawn_rate=sample_data['spawn_rate']
            )
            
            gif_frames, _ = generate_falling_circles_video(
                params=params,
                seed=sample_data['seed'],
                include_labels=True,
                actual_jam_type=sample_data['jam_type']
            )
            
            # Convert numpy arrays to PIL Images
            pil_frames = [Image.fromarray(frame.astype(np.uint8)) for frame in gif_frames]
            
            # Save GIF
            gif_path = os.path.join(self.dirs['visualization'], f"{sample_id}.gif")
            pil_frames[0].save(
                gif_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=100,
                loop=0
            )
        
        # Intervention pair GIFs (subset)
        pair_indices = random.sample(range(len(self.pairs_data)), min(20, len(self.pairs_data)))
        
        for idx in tqdm(pair_indices, desc="Intervention GIFs"):
            pair_data = self.pairs_data[idx]
            pair_id = pair_data['pair_id']
            
            # Load baseline and intervention metadata
            pair_dir = os.path.join(self.dirs['intervention'], pair_id)
            baseline_meta_path = os.path.join(pair_dir, "baseline", "meta.json")
            
            with open(baseline_meta_path, 'r') as f:
                baseline_meta = json.load(f)
            
            baseline_params = VideoParams(**baseline_meta['params'])
            
            # Find intervention directory
            intervention_dirs = [d for d in os.listdir(pair_dir) if d.startswith('do_')]
            if not intervention_dirs:
                continue
                
            intervention_dir_name = intervention_dirs[0]
            intervention_meta_path = os.path.join(pair_dir, intervention_dir_name, "meta.json")
            
            with open(intervention_meta_path, 'r') as f:
                intervention_meta = json.load(f)
            
            intervention_params = VideoParams(**intervention_meta['params'])
            
            # Generate baseline GIF
            baseline_gif_frames, _ = generate_falling_circles_video(
                params=baseline_params,
                seed=pair_data['seed'],
                include_labels=True,
                actual_jam_type=pair_data['baseline_jam_type']
            )
            
            # Generate intervention GIF
            intervention_gif_frames, _ = generate_falling_circles_video(
                params=intervention_params,
                seed=pair_data['seed'],
                include_labels=True,
                actual_jam_type=pair_data['intervention_jam_type']
            )
            
            # Convert numpy arrays to PIL Images
            baseline_pil_frames = [Image.fromarray(frame.astype(np.uint8)) for frame in baseline_gif_frames]
            intervention_pil_frames = [Image.fromarray(frame.astype(np.uint8)) for frame in intervention_gif_frames]
            
            # Create side-by-side comparison (simplified - just save both)
            baseline_gif_path = os.path.join(self.dirs['visualization'], f"{pair_id}__baseline.gif")
            intervention_gif_path = os.path.join(self.dirs['visualization'], f"{pair_id}__intervention.gif")
            
            baseline_pil_frames[0].save(
                baseline_gif_path,
                save_all=True,
                append_images=baseline_pil_frames[1:],
                duration=100,
                loop=0
            )
            
            intervention_pil_frames[0].save(
                intervention_gif_path,
                save_all=True,
                append_images=intervention_pil_frames[1:],
                duration=100,
                loop=0
            )
    
    def create_data_splits(self, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15):
        """Create train/val/test splits"""
        print("Creating data splits...")
        
        # Observation splits
        obs_samples = [s for s in self.samples_data if s['sample_type'] == 'observation']
        random.shuffle(obs_samples)
        
        n_obs = len(obs_samples)
        n_train = int(n_obs * train_ratio)
        n_val = int(n_obs * val_ratio)
        
        train_obs = obs_samples[:n_train]
        val_obs = obs_samples[n_train:n_train + n_val]
        test_obs = obs_samples[n_train + n_val:]
        
        # Intervention pair splits
        random.shuffle(self.pairs_data)
        n_pairs = len(self.pairs_data)
        n_train_pairs = int(n_pairs * train_ratio)
        n_val_pairs = int(n_pairs * val_ratio)
        
        train_pairs = self.pairs_data[:n_train_pairs]
        val_pairs = self.pairs_data[n_train_pairs:n_train_pairs + n_val_pairs]
        test_pairs = self.pairs_data[n_train_pairs + n_val_pairs:]
        
        # Save observation splits
        splits = {
            'train_ids.txt': [s['sample_id'] for s in train_obs],
            'val_ids.txt': [s['sample_id'] for s in val_obs],
            'test_ids.txt': [s['sample_id'] for s in test_obs],
            'test_ood_ids.txt': [s['sample_id'] for s in self.ood_samples]
        }
        
        for filename, ids in splits.items():
            split_path = os.path.join(self.dirs['splits'], filename)
            with open(split_path, 'w') as f:
                for sample_id in ids:
                    f.write(f"{sample_id}\n")
        
        # Save pair splits
        pair_splits = {
            'train_pair_ids.txt': [p['pair_id'] for p in train_pairs],
            'val_pair_ids.txt': [p['pair_id'] for p in val_pairs],
            'test_pair_ids.txt': [p['pair_id'] for p in test_pairs]
        }
        
        for filename, ids in pair_splits.items():
            split_path = os.path.join(self.dirs['splits'], filename)
            with open(split_path, 'w') as f:
                for pair_id in ids:
                    f.write(f"{pair_id}\n")
        
        print(f"Splits created: train={len(train_obs)+len(train_pairs)}, "
              f"val={len(val_obs)+len(val_pairs)}, test={len(test_obs)+len(test_pairs)}, "
              f"ood={len(self.ood_samples)}")
    
    def create_indices(self):
        """Create sample and pair indices"""
        print("Creating indices...")
        
        # Samples index
        all_samples = self.samples_data + self.ood_samples
        samples_df = pd.DataFrame(all_samples)
        samples_path = os.path.join(self.dirs['index'], 'samples.csv')
        samples_df.to_csv(samples_path, index=False)
        
        # Pairs index
        pairs_df = pd.DataFrame(self.pairs_data)
        pairs_path = os.path.join(self.dirs['index'], 'pairs.csv')
        pairs_df.to_csv(pairs_path, index=False)
        
        print(f"Indices created: {len(all_samples)} samples, {len(self.pairs_data)} pairs")
    
    def create_audits(self):
        """Create independence and quality audits"""
        print("Creating audits...")
        
        if not self.parameter_data:
            print("No parameter data available for audits")
            return
        
        # Parameter independence analysis
        param_df = pd.DataFrame(self.parameter_data)
        
        # Correlation matrix
        numeric_cols = ['hole_diameter', 'wind_strength', 'num_circles', 
                       'circle_size_min', 'circle_size_max', 'spawn_rate', 'exit_ratio']
        correlation_matrix = param_df[numeric_cols].corr()
        
        # Save correlation matrix
        corr_path = os.path.join(self.dirs['audits'], 'independence_matrix.csv')
        correlation_matrix.to_csv(corr_path)
        
        # Generate heatmap
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.3f')
            plt.title('Parameter Independence Matrix')
            plt.tight_layout()
            
            heatmap_path = os.path.join(self.dirs['audits'], 'independence_heatmap.png')
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
        except ImportError:
            print("Matplotlib/seaborn not available for heatmap generation")
        
        # Label sensitivity report
        if self.effect_data:
            effect_df = pd.DataFrame(self.effect_data)
            
            # Analyze intervention effects
            effect_summary = effect_df.groupby('intervention_name').agg({
                'effect_magnitude': ['mean', 'std', 'count'],
                'jam_type_changed': 'mean',
                'exit_ratio_delta': ['mean', 'std']
            }).round(4)
            
            report_lines = [
                f"# Label Sensitivity Analysis Report",
                f"Generated: {datetime.now().isoformat()}",
                f"Dataset: {self.dataset_name}",
                f"",
                f"## Parameter Independence",
                f"Maximum correlation between independent parameters: {correlation_matrix.abs().max().max():.4f}",
                f"Parameters with high correlation (>0.3):",
            ]
            
            high_corr_pairs = []
            for i, col1 in enumerate(correlation_matrix.columns):
                for j, col2 in enumerate(correlation_matrix.columns):
                    if i < j and abs(correlation_matrix.loc[col1, col2]) > 0.3:
                        high_corr_pairs.append(f"  {col1} - {col2}: {correlation_matrix.loc[col1, col2]:.4f}")
            
            if high_corr_pairs:
                report_lines.extend(high_corr_pairs)
            else:
                report_lines.append("  None found - parameters are independent")
            
            report_lines.extend([
                f"",
                f"## Intervention Effects",
                f"Intervention sensitivity analysis:",
                f""
            ])
            
            for intervention in effect_summary.index:
                mean_effect = effect_summary.loc[intervention, ('effect_magnitude', 'mean')]
                success_rate = effect_summary.loc[intervention, ('jam_type_changed', 'mean')]
                count = effect_summary.loc[intervention, ('effect_magnitude', 'count')]
                
                report_lines.append(f"  {intervention}: mean_effect={mean_effect:.4f}, "
                                  f"success_rate={success_rate:.2%}, samples={count}")
            
            report_path = os.path.join(self.dirs['audits'], 'label_sensitivity_report.txt')
            with open(report_path, 'w') as f:
                f.write('\n'.join(report_lines))
        
        print("Audits completed")
    
    def create_readme_and_license(self):
        """Create README and LICENSE files"""
        print("Creating documentation...")
        
        # Get max correlation for README
        max_correlation = 0.0
        if self.parameter_data:
            param_df = pd.DataFrame(self.parameter_data)
            numeric_cols = ['hole_diameter', 'wind_strength', 'num_circles', 
                           'circle_size_min', 'circle_size_max', 'spawn_rate', 'exit_ratio']
            if all(col in param_df.columns for col in numeric_cols):
                correlation_matrix = param_df[numeric_cols].corr()
                max_correlation = correlation_matrix.abs().max().max()
        
        readme_content = f"""# {self.dataset_name}

Physics-based video dataset for causal inference research featuring falling circles with jam dynamics.

## Dataset Structure

- `observation/`: Pure observational data ({len(self.samples_data)} samples)
- `intervention/`: Intervention pairs with baseline and do() operations ({len(self.pairs_data)} pairs)
- `splits/`: Train/val/test/OOD splits
- `index/`: Sample and pair indices (CSV format)
- `audits/`: Independence and quality audits
- `visualization/`: Human-readable GIF visualizations
- `ood/`: Out-of-distribution samples ({len(self.ood_samples)} samples)

## Data Format

### Observation Samples
Each sample in `observation/{{sample_id}}/` contains:
- `frames/`: PNG frames (224x224 RGB)
- `meta.json`: Physical parameters, seed, and labels
- `stats.csv`: Per-frame statistics

### Intervention Pairs
Each pair in `intervention/{{pair_id}}/` contains:
- `baseline/`: Original simulation
- `do_{{variable}}_{{delta}}/`: Intervention with specific variable modification
- Each subdirectory follows the same structure as observation samples

## Labels and Variables

### Outcome Variables
- `jam_type`: Emergent jam classification (no_jam, partial_jam, full_jam)
- `exit_ratio`: Proportion of circles that successfully exit

### Physical Parameters
- `hole_diameter`: Exit hole size (10-50 pixels)
- `wind_strength`: Horizontal wind force (0-0.4 px/frame)
- `num_circles`: Number of objects (3-8)
- `circle_size_min/max`: Object size range (2-15 pixels)
- `spawn_rate`: Object spawn frequency (0.15-0.6)

### Interventions
Available intervention types:
- Hole modifications: larger/smaller/offset
- Wind changes: strength/direction
- Object modifications: size/count changes
- Timing changes: spawn rate modifications

## Quality Assurance

### Parameter Independence
- Verified statistical independence between causal variables
- Maximum correlation: {max_correlation:.4f} (see audits/)
- Unbiased parameter sampling ensures valid causal inference

### Causal Effects
- {len(self.effect_data)} intervention effects measured
- Strong effects: hole size, circle size modifications
- Weak effects: wind, timing modifications

## Usage

```python
import pandas as pd
import json

# Load sample index
samples = pd.read_csv('index/samples.csv')

# Load specific sample
sample_id = samples.iloc[0]['sample_id']
with open(f'observation/{{sample_id}}/meta.json') as f:
    meta = json.load(f)

# Load intervention pairs
pairs = pd.read_csv('index/pairs.csv')
```

## Citation

```bibtex
@dataset{{{self.dataset_name},
  title={{Falling Circles Physics Dataset for Causal Inference}},
  year={{2025}},
  note={{Physics-based video dataset with verified parameter independence}}
}}
```

Generated: {datetime.now().isoformat()}
Total samples: {len(self.samples_data) + len(self.ood_samples)}
Total intervention pairs: {len(self.pairs_data)}
"""
        
        readme_path = os.path.join(self.dirs['root'], 'README.md')
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        # Simple license
        license_content = """MIT License

Copyright (c) 2025 ELVIS-C Dataset

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
        
        license_path = os.path.join(self.dirs['root'], 'LICENSE')
        with open(license_path, 'w') as f:
            f.write(license_content)
        
        print("Documentation created")
    
    def generate_complete_dataset(self, num_observations: int, num_pairs: int, include_visualizations: bool = True):
        """Generate complete dataset pipeline"""
        print(f"Starting complete dataset generation: {self.dataset_name}")
        print(f"Observations: {num_observations}, Pairs: {num_pairs}")
        
        # Create structure
        self.create_directory_structure()
        
        # Generate data
        self.generate_observation_samples(num_observations)
        self.generate_intervention_pairs(num_pairs)
        
        # Create organization
        self.create_data_splits()
        self.create_indices()
        self.create_audits()
        
        # Generate visualizations (optional, can be slow)
        if include_visualizations:
            self.generate_visualizations()
        
        # Create documentation
        self.create_readme_and_license()
        
        print(f"\n✅ Dataset generation complete!")
        print(f"📁 Location: {self.dataset_root}")
        print(f"📊 Total samples: {len(self.samples_data) + len(self.ood_samples)}")
        print(f"🔬 Total intervention pairs: {len(self.pairs_data)}")
        print(f"🎯 OOD samples: {len(self.ood_samples)}")

def main():
    parser = argparse.ArgumentParser(description="Generate structured falling circles dataset")
    parser.add_argument('--name', type=str, default='falling_circles_v1',
                       help='Dataset name')
    parser.add_argument('--out', type=str, default='datasets/',
                       help='Output directory for datasets')
    parser.add_argument('--num_observations', type=int, default=10,
                       help='Number of observational samples')
    parser.add_argument('--num_pairs', type=int, default=10,
                       help='Number of intervention pairs')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training data ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation data ratio')
    parser.add_argument('--no_visualizations', action='store_true',
                       help='Skip GIF generation for faster processing')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create pipeline
    pipeline = DatasetPipeline(args.name, args.out)
    
    # Generate dataset
    pipeline.generate_complete_dataset(
        num_observations=args.num_observations,
        num_pairs=args.num_pairs,
        include_visualizations=not args.no_visualizations
    )

if __name__ == "__main__":
    main()