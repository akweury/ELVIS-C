#!/usr/bin/env python3
"""
Generate Active Intervention Pairs

Uses intelligent controllers to dynamically select interventions based on past results.
This enables adaptive causal discovery where the system learns which interventions
are most informative and focuses exploration accordingly.

Usage:
    python generate_active_interventions.py --controller exploration --num_pairs 100 --out data/active_pairs
"""

import os
import json
import random
import argparse
import copy
import time
import numpy as np
from typing import Tuple, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path

# Import from falling_circles
import sys
sys.path.append(os.path.dirname(__file__))
from falling_circles_env import (
    VideoParams, generate_falling_circles_video
)
from falling_circles import (
    sample_video_params, _generate_video_worker
)
from active_controllers import (
    ActiveController, create_controller, InterventionResult,
    save_controller_state, load_controller_state
)

def create_dynamic_intervention(controller: ActiveController, baseline_params: VideoParams) -> Tuple[str, VideoParams]:
    """
    Use controller to dynamically choose an intervention
    
    Returns:
        Tuple of (intervention_name, intervention_params)
    """
    # Available intervention types
    available_interventions = [
        'wind_strength_high', 'wind_strength_low', 'wind_direction_flip',
        'hole_larger', 'hole_smaller', 'hole_offset',
        'more_circles', 'fewer_circles', 'larger_circles', 'smaller_circles',
        'spawn_faster', 'spawn_slower'
    ]
    
    # Let controller decide
    intervention_name, custom_params = controller.decide_intervention(baseline_params, available_interventions)
    
    # Create intervention parameters
    intervention_params = copy.deepcopy(baseline_params)
    
    # Apply standard interventions or custom parameters
    if custom_params:
        # Use controller-specified parameters
        for param_name, param_value in custom_params.items():
            setattr(intervention_params, param_name, param_value)
    else:
        # Use default intervention logic
        intervention_params = apply_standard_intervention(intervention_name, intervention_params)
    
    return intervention_name, intervention_params

def apply_standard_intervention(intervention_name: str, params: VideoParams) -> VideoParams:
    """Apply standard intervention modifications"""
    
    if intervention_name == 'wind_strength_high':
        params.wind_strength = min(0.4, params.wind_strength * 2.0)
    elif intervention_name == 'wind_strength_low':
        params.wind_strength = max(0.0, params.wind_strength * 0.5)
    elif intervention_name == 'wind_direction_flip':
        params.wind_direction = -params.wind_direction
    elif intervention_name == 'hole_larger':
        params.hole_diameter = min(50, params.hole_diameter + 10)
    elif intervention_name == 'hole_smaller':
        params.hole_diameter = max(10, params.hole_diameter - 10)
    elif intervention_name == 'hole_offset':
        params.hole_x_position = 0.3 if params.hole_x_position > 0.4 else 0.7
    elif intervention_name == 'more_circles':
        params.num_circles = min(15, params.num_circles + 3)
    elif intervention_name == 'fewer_circles':
        params.num_circles = max(1, params.num_circles - 2)
    elif intervention_name == 'larger_circles':
        # Increase both min and max circle sizes
        params.circle_size_min = min(15, params.circle_size_min + 2)
        params.circle_size_max = min(20, params.circle_size_max + 3)
    elif intervention_name == 'smaller_circles':
        # Decrease both min and max circle sizes
        params.circle_size_min = max(2, params.circle_size_min - 2)
        params.circle_size_max = max(3, params.circle_size_max - 3)
    elif intervention_name == 'spawn_faster':
        params.spawn_rate = min(1.0, params.spawn_rate * 1.5)
    elif intervention_name == 'spawn_slower':
        params.spawn_rate = max(0.05, params.spawn_rate * 0.7)
    
    return params

def compute_intervention_effect(baseline_meta: dict, intervention_meta: dict) -> float:
    """Compute overall effect magnitude between baseline and intervention"""
    
    # Exit ratio difference (primary effect)
    baseline_exit_ratio = baseline_meta.get('exit_statistics', {}).get('exit_ratio', 0)
    intervention_exit_ratio = intervention_meta.get('exit_statistics', {}).get('exit_ratio', 0)
    exit_delta = abs(intervention_exit_ratio - baseline_exit_ratio)
    
    # Jam type change (binary)
    baseline_jam_type = baseline_meta.get('actual_jam_type', 'unknown')
    intervention_jam_type = intervention_meta.get('actual_jam_type', 'unknown')
    jam_changed = int(baseline_jam_type != intervention_jam_type)
    
    # Weighted combination
    effect_magnitude = 0.7 * exit_delta + 0.3 * jam_changed
    
    return effect_magnitude

def generate_active_intervention_pair(
    pair_id: int, 
    controller_state: dict,
    output_dir: str,
    gif_fps: int = 10,
    export_gif: bool = True
) -> dict:
    """
    Generate a single active intervention pair using controller guidance
    
    This function is designed to be called in the main process to maintain
    controller state consistency.
    """
    
    # Sample baseline parameters
    baseline_params = sample_video_params()
    seed = random.randint(0, 1000000)
    
    # Restore controller from state
    controller = create_controller(controller_state['type'], **controller_state.get('kwargs', {}))
    if 'history' in controller_state:
        # Restore history (simplified for this example)
        pass
    
    # Let controller choose intervention
    intervention_name, intervention_params = create_dynamic_intervention(controller, baseline_params)
    
    # Generate baseline video
    baseline_dir = os.path.join(output_dir, f"pair_{pair_id:05d}", "baseline")
    os.makedirs(baseline_dir, exist_ok=True)
    
    # Generate frames and metadata
    baseline_frames, baseline_meta = generate_falling_circles_video(
        params=baseline_params,
        seed=seed,
        include_labels=False
    )
    
    # Save baseline frames
    for frame_idx, frame in enumerate(baseline_frames):
        frame_path = os.path.join(baseline_dir, f"frame_{frame_idx:03d}.png")
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
    
    # Save baseline metadata
    with open(os.path.join(baseline_dir, "meta.json"), 'w') as f:
        json.dump(baseline_meta, f, indent=2)
    
    # Generate intervention video (same seed!)
    intervention_dir = os.path.join(output_dir, f"pair_{pair_id:05d}", f"intervention_{intervention_name}")
    os.makedirs(intervention_dir, exist_ok=True)
    
    # Generate frames and metadata
    intervention_frames, intervention_meta = generate_falling_circles_video(
        params=intervention_params,
        seed=seed,  # Same seed for perfect counterfactual
        include_labels=False
    )
    
    # Save intervention frames
    for frame_idx, frame in enumerate(intervention_frames):
        frame_path = os.path.join(intervention_dir, f"frame_{frame_idx:03d}.png")
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
    
    # Save intervention metadata
    with open(os.path.join(intervention_dir, "meta.json"), 'w') as f:
        json.dump(intervention_meta, f, indent=2)
    
    # Compute causal effect
    effect_magnitude = compute_intervention_effect(baseline_meta, intervention_meta)
    baseline_jam_type = baseline_meta.get('actual_jam_type', 'unknown')
    intervention_jam_type = intervention_meta.get('actual_jam_type', 'unknown')
    jam_type_changed = baseline_jam_type != intervention_jam_type
    
    baseline_exit_ratio = baseline_meta.get('exit_statistics', {}).get('exit_ratio', 0)
    intervention_exit_ratio = intervention_meta.get('exit_statistics', {}).get('exit_ratio', 0)
    exit_ratio_delta = intervention_exit_ratio - baseline_exit_ratio
    
    # Create intervention result
    intervention_result = InterventionResult(
        intervention_name=intervention_name,
        intervention_params={
            k: getattr(intervention_params, k) for k in ['hole_diameter', 'wind_strength', 'wind_direction', 
                                                        'num_circles', 'circle_size_min', 'circle_size_max', 'spawn_rate']
            if hasattr(intervention_params, k)
        },
        baseline_outcome={
            'jam_type': baseline_jam_type,
            'exit_ratio': baseline_exit_ratio
        },
        intervention_outcome={
            'jam_type': intervention_jam_type,
            'exit_ratio': intervention_exit_ratio
        },
        effect_magnitude=effect_magnitude,
        jam_type_changed=jam_type_changed,
        exit_ratio_delta=exit_ratio_delta
    )
    
    # Create comparison metadata
    comparison = {
        'pair_id': pair_id,
        'intervention_name': intervention_name,
        'controller_decision': controller.name,
        'baseline_params': baseline_params.__dict__,
        'intervention_params': intervention_params.__dict__,
        'baseline_outcome': baseline_meta,
        'intervention_outcome': intervention_meta,
        'causal_effect': {
            'effect_magnitude': effect_magnitude,
            'jam_type_changed': jam_type_changed,
            'exit_ratio_delta': exit_ratio_delta,
            'exit_ratio_baseline': baseline_exit_ratio,
            'exit_ratio_intervention': intervention_exit_ratio
        },
        'seed': seed,
        'timestamp': time.time()
    }
    
    # Save comparison
    comparison_path = os.path.join(output_dir, f"pair_{pair_id:05d}", "comparison.json")
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Generate GIFs if requested
    if export_gif:
        gif_dir = os.path.join(output_dir, "gifs", f"pair_{pair_id:05d}")
        os.makedirs(gif_dir, exist_ok=True)
        
        # Generate baseline GIF with labels
        baseline_gif_frames, _ = generate_falling_circles_video(
            params=baseline_params,
            seed=seed,
            include_labels=True,
            actual_jam_type=baseline_meta.get('actual_jam_type')
        )
        
        # Save baseline GIF
        baseline_gif_path = os.path.join(gif_dir, "baseline.gif")
        baseline_gif_frames[0].save(
            baseline_gif_path,
            save_all=True,
            append_images=baseline_gif_frames[1:],
            duration=int(1000/gif_fps),
            loop=0
        )
        
        # Generate intervention GIF with labels
        intervention_gif_frames, _ = generate_falling_circles_video(
            params=intervention_params,
            seed=seed,
            include_labels=True,
            actual_jam_type=intervention_meta.get('actual_jam_type')
        )
        
        # Save intervention GIF
        intervention_gif_path = os.path.join(gif_dir, f"intervention_{intervention_name}.gif")
        intervention_gif_frames[0].save(
            intervention_gif_path,
            save_all=True,
            append_images=intervention_gif_frames[1:],
            duration=int(1000/gif_fps),
            loop=0
        )
    
    return {
        'pair_id': pair_id,
        'comparison': comparison,
        'intervention_result': intervention_result
    }

def generate_active_intervention_dataset(
    controller_type: str,
    num_pairs: int,
    output_dir: str,
    train_ratio: float = 0.8,
    gif_fps: int = 10,
    export_gif: bool = True,
    controller_kwargs: dict = None,
    checkpoint_freq: int = 10
):
    """Generate dataset using active controller"""
    
    if controller_kwargs is None:
        controller_kwargs = {}
    
    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Initialize controller
    controller = create_controller(controller_type, **controller_kwargs)
    
    # Track results
    all_results = []
    controller_decisions = []
    
    # Generate pairs sequentially to maintain controller state
    print(f"Generating {num_pairs} active intervention pairs using {controller.name} controller...")
    
    for pair_id in tqdm(range(num_pairs), desc="Generating pairs"):
        
        # Determine train/test split
        is_train = random.random() < train_ratio
        current_output_dir = train_dir if is_train else test_dir
        
        # Generate pair
        result = generate_active_intervention_pair(
            pair_id=pair_id,
            controller_state={
                'type': controller_type,
                'kwargs': controller_kwargs
            },
            output_dir=current_output_dir,
            gif_fps=gif_fps,
            export_gif=export_gif
        )
        
        # Update controller with results
        controller.update_from_results(result['intervention_result'])
        
        # Track results
        all_results.append(result['comparison'])
        controller_decisions.append({
            'pair_id': pair_id,
            'intervention_chosen': result['intervention_result'].intervention_name,
            'effect_magnitude': result['intervention_result'].effect_magnitude,
            'controller_stats': controller.get_statistics()
        })
        
        # Periodic checkpointing
        if (pair_id + 1) % checkpoint_freq == 0:
            checkpoint_path = os.path.join(output_dir, f"controller_checkpoint_{pair_id + 1}.json")
            save_controller_state(controller, checkpoint_path)
            print(f"Saved controller checkpoint at pair {pair_id + 1}")
    
    # Create dataset manifest
    manifest = {
        'dataset_info': {
            'total_pairs': num_pairs,
            'train_pairs': sum(1 for r in all_results if 'train' in r.get('split', 'train')),
            'test_pairs': sum(1 for r in all_results if 'test' in r.get('split', '')),
            'controller_type': controller_type,
            'controller_kwargs': controller_kwargs,
            'generation_timestamp': time.time()
        },
        'controller_final_stats': controller.get_statistics(),
        'intervention_summary': _compute_intervention_summary(all_results),
        'pairs': all_results,
        'controller_decisions': controller_decisions
    }
    
    # Save manifest
    manifest_path = os.path.join(output_dir, "active_intervention_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Save final controller state
    final_controller_path = os.path.join(output_dir, "final_controller_state.json")
    save_controller_state(controller, final_controller_path)
    
    print(f"\nActive intervention dataset generated successfully!")
    print(f"Output directory: {output_dir}")
    print(f"Controller: {controller.name}")
    print(f"Total pairs: {num_pairs}")
    print(f"Manifest: {manifest_path}")
    
    # Print controller insights
    print(f"\n🧠 Controller Insights:")
    stats = controller.get_statistics()
    print(f"Total interventions tried: {stats['total_interventions']}")
    print(f"Intervention distribution: {stats['intervention_counts']}")
    
    if stats['effect_statistics']:
        print(f"\n📊 Discovered Effects:")
        for intervention, effect_stats in stats['effect_statistics'].items():
            print(f"  {intervention}: mean_effect={effect_stats['mean_effect']:.3f}, "
                  f"success_rate={effect_stats['success_rate']:.2f}")

def _compute_intervention_summary(results: list) -> dict:
    """Compute summary statistics for intervention results"""
    
    if not results:
        return {}
    
    # Count interventions
    intervention_counts = {}
    effect_magnitudes = {}
    jam_changes = {}
    
    for result in results:
        intervention = result['intervention_name']
        effect = result['causal_effect']['effect_magnitude']
        jam_changed = result['causal_effect']['jam_type_changed']
        
        if intervention not in intervention_counts:
            intervention_counts[intervention] = 0
            effect_magnitudes[intervention] = []
            jam_changes[intervention] = 0
        
        intervention_counts[intervention] += 1
        effect_magnitudes[intervention].append(effect)
        if jam_changed:
            jam_changes[intervention] += 1
    
    # Compute statistics
    summary = {
        'intervention_counts': intervention_counts,
        'effect_statistics': {},
        'total_jam_changes': sum(jam_changes.values()),
        'jam_change_rate': sum(jam_changes.values()) / len(results) if results else 0
    }
    
    for intervention in intervention_counts:
        effects = effect_magnitudes[intervention]
        summary['effect_statistics'][intervention] = {
            'count': intervention_counts[intervention],
            'mean_effect': sum(effects) / len(effects),
            'max_effect': max(effects),
            'min_effect': min(effects),
            'jam_change_count': jam_changes[intervention],
            'jam_change_rate': jam_changes[intervention] / intervention_counts[intervention]
        }
    
    return summary

def main():
    parser = argparse.ArgumentParser(description="Generate active intervention pairs")
    parser.add_argument('--controller', type=str, default='exploration',
                       choices=['random', 'max_effect', 'uncertainty', 'exploration', 'adaptive'],
                       help='Controller type for active intervention selection')
    parser.add_argument('--num_pairs', type=int, default=50,
                       help='Number of intervention pairs to generate')
    parser.add_argument('--out', type=str, default='output/active_interventions',
                       help='Output directory for generated data')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Proportion of data for training (default: 0.8)')
    parser.add_argument('--gif_fps', type=int, default=10,
                       help='Frame rate for GIF files')
    parser.add_argument('--no_gif', action='store_true',
                       help='Skip GIF generation for faster processing')
    parser.add_argument('--checkpoint_freq', type=int, default=10,
                       help='Save controller checkpoint every N pairs')
    
    # Controller-specific arguments
    parser.add_argument('--exploration_rate', type=float, default=0.2,
                       help='Exploration rate for max_effect controller')
    parser.add_argument('--exploration_constant', type=float, default=1.4,
                       help='Exploration constant for exploration controller')
    parser.add_argument('--min_samples', type=int, default=3,
                       help='Minimum samples per intervention for uncertainty controller')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                       help='Learning rate for adaptive controller')
    
    args = parser.parse_args()
    
    # Prepare controller kwargs
    controller_kwargs = {}
    if args.controller == 'max_effect':
        controller_kwargs['exploration_rate'] = args.exploration_rate
    elif args.controller == 'exploration':
        controller_kwargs['exploration_constant'] = args.exploration_constant
    elif args.controller == 'uncertainty':
        controller_kwargs['min_samples'] = args.min_samples
    elif args.controller == 'adaptive':
        controller_kwargs['learning_rate'] = args.learning_rate
    
    # Generate dataset
    generate_active_intervention_dataset(
        controller_type=args.controller,
        num_pairs=args.num_pairs,
        output_dir=args.out,
        train_ratio=args.train_ratio,
        gif_fps=args.gif_fps,
        export_gif=not args.no_gif,
        controller_kwargs=controller_kwargs,
        checkpoint_freq=args.checkpoint_freq
    )

if __name__ == "__main__":
    main()