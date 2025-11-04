#!/usr/bin/env python3
"""
Generate Intervention Pairs for Causal Analysis

For each baseline simulation, generate an additional intervention version by modifying 
exactly one causal variable while keeping the random seed and all other parameters identical.

This creates perfect counterfactual pairs for causal inference research.

Usage:
    python generate_intervention_pairs.py --num_videos 100 --out data/intervention_pairs --workers 8
"""

import os
import json
import random
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Import from falling_circles
import sys
sys.path.append(os.path.dirname(__file__))
from falling_circles_env import (
    VideoParams, generate_falling_circles_video
)
from falling_circles import (
    sample_video_params, _generate_video_worker
)

def create_intervention_variants(baseline_params):
    """
    Create intervention variants by modifying exactly one causal variable
    
    Returns list of (intervention_name, modified_params) tuples
    """
    interventions = []
    
    # Define the causal variables that can be intervened upon
    causal_interventions = {
        # Physics interventions
        'wind_strength_high': {'wind_strength': min(0.4, baseline_params.wind_strength * 2.0)},
        'wind_strength_low': {'wind_strength': max(0.0, baseline_params.wind_strength * 0.5)},
        'wind_direction_flip': {'wind_direction': -baseline_params.wind_direction},
        
        # Geometry interventions  
        'hole_larger': {'hole_diameter': min(50, baseline_params.hole_diameter + 10)},
        'hole_smaller': {'hole_diameter': max(10, baseline_params.hole_diameter - 10)},
        'hole_offset': {'hole_x_position': 0.3 if baseline_params.hole_x_position > 0.4 else 0.7},
        
        # Object interventions
        'more_circles': {'num_circles': min(15, baseline_params.num_circles + 3)},
        'fewer_circles': {'num_circles': max(1, baseline_params.num_circles - 3)},
        'larger_circles': {
            'circle_size_min': min(20, baseline_params.circle_size_min + 5),
            'circle_size_max': min(25, baseline_params.circle_size_max + 5)
        },
        'smaller_circles': {
            'circle_size_min': max(3, baseline_params.circle_size_min - 3),
            'circle_size_max': max(5, baseline_params.circle_size_max - 3)
        },
        
        # Timing interventions
        'spawn_faster': {'spawn_rate': min(1.0, baseline_params.spawn_rate * 1.5)},
        'spawn_slower': {'spawn_rate': max(0.05, baseline_params.spawn_rate * 0.5)},
    }
    
    # Create intervention variants
    for intervention_name, modifications in causal_interventions.items():
        # Create a copy of baseline parameters
        intervention_params = copy.deepcopy(baseline_params)
        
        # Apply the specific intervention
        for param_name, new_value in modifications.items():
            setattr(intervention_params, param_name, new_value)
        
        # Ensure circle size consistency
        if hasattr(intervention_params, 'circle_size_min') and hasattr(intervention_params, 'circle_size_max'):
            if intervention_params.circle_size_min >= intervention_params.circle_size_max:
                intervention_params.circle_size_max = intervention_params.circle_size_min + 2
        
        interventions.append((intervention_name, intervention_params))
    
    return interventions

def generate_intervention_pair_worker(pair_idx, baseline_params, output_root, split, 
                                    intervention_name, intervention_params, seed,
                                    export_gif=True, gif_fps=10):
    """
    Worker function to generate a baseline + intervention pair
    """
    import csv
    
    pair_name = f"pair_{pair_idx:05d}"
    pair_dir = os.path.join(output_root, split, pair_name)
    os.makedirs(pair_dir, exist_ok=True)
    
    results = {}
    
    # Generate baseline video
    baseline_frames, baseline_meta = generate_falling_circles_video(
        baseline_params, seed, include_labels=False
    )
    
    # Save baseline
    baseline_dir = os.path.join(pair_dir, "baseline")
    os.makedirs(baseline_dir, exist_ok=True)
    
    for frame_idx, frame in enumerate(baseline_frames):
        from PIL import Image
        img = Image.fromarray(frame)
        frame_path = os.path.join(baseline_dir, f"frame_{frame_idx:03d}.png")
        img.save(frame_path)
    
    baseline_meta_path = os.path.join(baseline_dir, "meta.json")
    with open(baseline_meta_path, 'w') as f:
        json.dump(baseline_meta, f, indent=2)
    
    # Generate intervention video (SAME SEED!)
    intervention_frames, intervention_meta = generate_falling_circles_video(
        intervention_params, seed, include_labels=False
    )
    
    # Save intervention
    intervention_dir = os.path.join(pair_dir, f"intervention_{intervention_name}")
    os.makedirs(intervention_dir, exist_ok=True)
    
    for frame_idx, frame in enumerate(intervention_frames):
        from PIL import Image
        img = Image.fromarray(frame)
        frame_path = os.path.join(intervention_dir, f"frame_{frame_idx:03d}.png")
        img.save(frame_path)
    
    intervention_meta_path = os.path.join(intervention_dir, "meta.json")
    with open(intervention_meta_path, 'w') as f:
        json.dump(intervention_meta, f, indent=2)
    
    # Generate comparison data
    comparison_data = {
        'pair_idx': pair_idx,
        'pair_name': pair_name,
        'seed': seed,
        'intervention_type': intervention_name,
        'baseline': {
            'params': baseline_params.to_dict(),
            'jam_type': baseline_meta['actual_jam_type'],
            'exit_statistics': baseline_meta['exit_statistics']
        },
        'intervention': {
            'params': intervention_params.to_dict(),
            'jam_type': intervention_meta['actual_jam_type'],
            'exit_statistics': intervention_meta['exit_statistics']
        },
        'causal_effect': {
            'jam_type_changed': baseline_meta['actual_jam_type'] != intervention_meta['actual_jam_type'],
            'exit_ratio_change': (intervention_meta['exit_statistics']['exit_ratio'] - 
                                baseline_meta['exit_statistics']['exit_ratio']),
            'total_exited_change': (intervention_meta['exit_statistics']['total_exited'] - 
                                  baseline_meta['exit_statistics']['total_exited'])
        }
    }
    
    comparison_path = os.path.join(pair_dir, "comparison.json")
    with open(comparison_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    # Generate GIFs if requested
    if export_gif:
        gifs_dir = os.path.join(output_root, "gifs", split, pair_name)
        os.makedirs(gifs_dir, exist_ok=True)
        
        # Baseline GIF (with labels including intervention info)
        baseline_gif_frames, _ = generate_falling_circles_video(
            baseline_params, seed, include_labels=True, 
            actual_jam_type=baseline_meta['actual_jam_type']
        )
        
        from PIL import Image
        baseline_gif_path = os.path.join(gifs_dir, "baseline.gif")
        baseline_images = [Image.fromarray(frame) for frame in baseline_gif_frames]
        baseline_images[0].save(
            baseline_gif_path,
            save_all=True,
            append_images=baseline_images[1:],
            duration=int(1000/gif_fps),
            loop=0
        )
        
        # Intervention GIF
        intervention_gif_frames, _ = generate_falling_circles_video(
            intervention_params, seed, include_labels=True,
            actual_jam_type=intervention_meta['actual_jam_type']
        )
        
        intervention_gif_path = os.path.join(gifs_dir, f"intervention_{intervention_name}.gif")
        intervention_images = [Image.fromarray(frame) for frame in intervention_gif_frames]
        intervention_images[0].save(
            intervention_gif_path,
            save_all=True,
            append_images=intervention_images[1:],
            duration=int(1000/gif_fps),
            loop=0
        )
    
    return comparison_data

def main():
    parser = argparse.ArgumentParser(description="Generate intervention pairs for causal analysis")
    parser.add_argument("--num_pairs", type=int, default=20,
                       help="Number of baseline-intervention pairs to generate")
    parser.add_argument("--out", type=str, default="data/intervention_pairs",
                       help="Output directory for intervention pairs")
    parser.add_argument("--num_frames", type=int, default=60,
                       help="Number of frames per video")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of parallel workers")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                       help="Ratio of training pairs (default: 0.8)")
    parser.add_argument("--export_gif", action="store_true", default=True,
                       help="Export GIF files for visualization")
    parser.add_argument("--no_gif", action="store_true",
                       help="Disable GIF export")
    parser.add_argument("--gif_fps", type=int, default=10,
                       help="Frame rate for GIF files")
    parser.add_argument("--intervention_type", type=str, 
                       choices=['wind_strength_high', 'wind_strength_low', 'wind_direction_flip',
                               'hole_larger', 'hole_smaller', 'hole_offset',
                               'more_circles', 'fewer_circles', 'larger_circles', 'smaller_circles',
                               'spawn_faster', 'spawn_slower', 'all'],
                       default='all',
                       help="Type of intervention to generate (default: all)")
    
    args = parser.parse_args()
    
    # Handle GIF export
    export_gif = args.export_gif and not args.no_gif
    
    # Create output directories
    os.makedirs(args.out, exist_ok=True)
    train_dir = os.path.join(args.out, "train")
    test_dir = os.path.join(args.out, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Split pairs into train/test
    num_train_pairs = int(args.num_pairs * args.train_ratio)
    num_test_pairs = args.num_pairs - num_train_pairs
    
    print(f"🎯 Generating {args.num_pairs} causal intervention pairs")
    print(f"   📚 Train pairs: {num_train_pairs}")
    print(f"   🧪 Test pairs: {num_test_pairs}")
    
    # Generate baseline parameters and interventions
    tasks = []
    all_comparison_data = []
    
    for pair_idx in range(args.num_pairs):
        split = "train" if pair_idx < num_train_pairs else "test"
        is_train = (split == "train")
        
        # Generate baseline parameters
        baseline_params = sample_video_params(args.num_frames, is_train=is_train)
        
        # Generate intervention variants
        interventions = create_intervention_variants(baseline_params)
        
        # Select intervention type
        if args.intervention_type == 'all':
            # Cycle through different intervention types
            intervention_idx = pair_idx % len(interventions)
            intervention_name, intervention_params = interventions[intervention_idx]
        else:
            # Use specified intervention type
            matching_interventions = [iv for iv in interventions if iv[0] == args.intervention_type]
            if not matching_interventions:
                print(f"❌ Error: Intervention type '{args.intervention_type}' not found")
                return
            intervention_name, intervention_params = matching_interventions[0]
        
        # Use same seed for both baseline and intervention
        seed = random.randint(0, 2_000_000_000)
        
        tasks.append((pair_idx, baseline_params, args.out, split, intervention_name, 
                     intervention_params, seed, export_gif, args.gif_fps))
    
    # Generate pairs in parallel
    print(f"\\n🔄 Generating intervention pairs...")
    results = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(generate_intervention_pair_worker, *task) for task in tasks]
        
        for future in tqdm(as_completed(futures), total=args.num_pairs, desc="Generating pairs"):
            results.append(future.result())
    
    # Analyze causal effects
    print(f"\\n📊 Analyzing causal effects...")
    
    effect_summary = {}
    jam_transitions = {}
    
    for result in results:
        intervention_type = result['intervention_type']
        
        if intervention_type not in effect_summary:
            effect_summary[intervention_type] = {
                'total_pairs': 0,
                'jam_type_changes': 0,
                'avg_exit_ratio_change': 0,
                'exit_ratio_changes': []
            }
        
        effect_summary[intervention_type]['total_pairs'] += 1
        
        if result['causal_effect']['jam_type_changed']:
            effect_summary[intervention_type]['jam_type_changes'] += 1
            
            # Track jam type transitions
            baseline_jam = result['baseline']['jam_type']
            intervention_jam = result['intervention']['jam_type']
            transition = f"{baseline_jam} → {intervention_jam}"
            
            if intervention_type not in jam_transitions:
                jam_transitions[intervention_type] = {}
            jam_transitions[intervention_type][transition] = jam_transitions[intervention_type].get(transition, 0) + 1
        
        effect_summary[intervention_type]['exit_ratio_changes'].append(
            result['causal_effect']['exit_ratio_change']
        )
    
    # Calculate averages
    for intervention_type in effect_summary:
        changes = effect_summary[intervention_type]['exit_ratio_changes']
        effect_summary[intervention_type]['avg_exit_ratio_change'] = sum(changes) / len(changes) if changes else 0
        effect_summary[intervention_type]['effect_magnitude'] = sum(abs(c) for c in changes) / len(changes) if changes else 0
    
    # Save dataset manifest
    manifest_data = {
        'total_pairs': args.num_pairs,
        'train_pairs': num_train_pairs,
        'test_pairs': num_test_pairs,
        'intervention_types': list(effect_summary.keys()),
        'causal_effects_summary': effect_summary,
        'jam_type_transitions': jam_transitions,
        'pairs': sorted(results, key=lambda x: x['pair_idx'])
    }
    
    manifest_path = os.path.join(args.out, "intervention_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest_data, f, indent=2)
    
    # Print summary
    gif_status = f"with GIFs (fps={args.gif_fps})" if export_gif else "without GIFs"
    
    print(f"\\n✅ Generated {args.num_pairs} intervention pairs")
    print(f"📁 Output directory: {args.out}")
    print(f"🎨 Format: PNG frames {gif_status}")
    print(f"📄 Manifest: {manifest_path}")
    
    print(f"\\n🎯 CAUSAL EFFECTS SUMMARY:")
    for intervention_type, summary in effect_summary.items():
        print(f"\\n{intervention_type}:")
        print(f"  • Total pairs: {summary['total_pairs']}")
        print(f"  • Jam type changes: {summary['jam_type_changes']} ({summary['jam_type_changes']/summary['total_pairs']*100:.1f}%)")
        print(f"  • Avg exit ratio change: {summary['avg_exit_ratio_change']:+.3f}")
        print(f"  • Effect magnitude: {summary['effect_magnitude']:.3f}")
        
        if intervention_type in jam_transitions:
            print(f"  • Transitions: {jam_transitions[intervention_type]}")

if __name__ == "__main__":
    main()