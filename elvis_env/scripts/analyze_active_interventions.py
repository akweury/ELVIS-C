#!/usr/bin/env python3
"""
Analyze Active Intervention Results

Compare performance of different active controllers for causal discovery.
Analyzes intervention selection patterns, discovery efficiency, and learning curves.

Usage:
    python analyze_active_interventions.py --data_dir results/active_comparison --output_dir analysis/
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def load_active_intervention_data(data_dir: str) -> Dict[str, Any]:
    """Load active intervention dataset and extract key metrics"""
    
    manifest_path = os.path.join(data_dir, "active_intervention_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    return manifest

def compute_discovery_efficiency(controller_decisions: List[Dict]) -> Dict[str, float]:
    """Compute causal discovery efficiency metrics"""
    
    if not controller_decisions:
        return {}
    
    # Track cumulative effect discovery
    cumulative_effects = []
    cumulative_high_effects = []
    intervention_diversity = []
    
    seen_interventions = set()
    total_effect = 0
    high_effect_count = 0
    
    for i, decision in enumerate(controller_decisions):
        effect = decision.get('effect_magnitude', 0)
        intervention = decision.get('intervention_chosen', '')
        
        total_effect += effect
        if effect > 0.3:  # High effect threshold
            high_effect_count += 1
        
        seen_interventions.add(intervention)
        
        cumulative_effects.append(total_effect / (i + 1))  # Mean effect so far
        cumulative_high_effects.append(high_effect_count / (i + 1))  # High effect rate
        intervention_diversity.append(len(seen_interventions) / (i + 1))  # Diversity
    
    return {
        'mean_effect_discovery': cumulative_effects,
        'high_effect_discovery_rate': cumulative_high_effects,
        'intervention_diversity': intervention_diversity,
        'final_mean_effect': cumulative_effects[-1] if cumulative_effects else 0,
        'final_high_effect_rate': cumulative_high_effects[-1] if cumulative_high_effects else 0,
        'final_diversity': intervention_diversity[-1] if intervention_diversity else 0
    }

def analyze_intervention_selection_patterns(controller_decisions: List[Dict]) -> Dict[str, Any]:
    """Analyze how controllers select interventions over time"""
    
    intervention_timeline = []
    effect_timeline = []
    
    for decision in controller_decisions:
        intervention_timeline.append(decision.get('intervention_chosen', ''))
        effect_timeline.append(decision.get('effect_magnitude', 0))
    
    # Compute selection frequency over time windows
    window_size = 10
    selection_evolution = []
    
    for i in range(0, len(intervention_timeline), window_size):
        window = intervention_timeline[i:i+window_size]
        window_counts = {}
        for intervention in window:
            window_counts[intervention] = window_counts.get(intervention, 0) + 1
        selection_evolution.append(window_counts)
    
    # Identify exploitation vs exploration phases
    exploitation_phases = []
    for i, window in enumerate(selection_evolution):
        if window:
            max_count = max(window.values())
            total_count = sum(window.values())
            exploitation_ratio = max_count / total_count if total_count > 0 else 0
            exploitation_phases.append({
                'window': i,
                'exploitation_ratio': exploitation_ratio,
                'dominant_intervention': max(window, key=window.get) if window else None
            })
    
    return {
        'intervention_timeline': intervention_timeline,
        'effect_timeline': effect_timeline,
        'selection_evolution': selection_evolution,
        'exploitation_phases': exploitation_phases
    }

def compare_controller_performance(controller_results: Dict[str, Dict]) -> pd.DataFrame:
    """Compare performance metrics across different controllers"""
    
    comparison_data = []
    
    for controller_name, results in controller_results.items():
        manifest = results['manifest']
        decisions = manifest.get('controller_decisions', [])
        
        if not decisions:
            continue
        
        efficiency = compute_discovery_efficiency(decisions)
        patterns = analyze_intervention_selection_patterns(decisions)
        
        # Extract final statistics
        final_stats = manifest.get('controller_final_stats', {})
        effect_stats = final_stats.get('effect_statistics', {})
        
        # Compute aggregate metrics
        mean_effects = []
        success_rates = []
        for intervention, stats in effect_stats.items():
            mean_effects.append(stats.get('mean_effect', 0))
            success_rates.append(stats.get('success_rate', 0))
        
        comparison_data.append({
            'controller': controller_name,
            'total_interventions': final_stats.get('total_interventions', 0),
            'unique_interventions': len(effect_stats),
            'final_mean_effect': efficiency.get('final_mean_effect', 0),
            'final_high_effect_rate': efficiency.get('final_high_effect_rate', 0),
            'final_diversity': efficiency.get('final_diversity', 0),
            'overall_mean_effect': np.mean(mean_effects) if mean_effects else 0,
            'overall_success_rate': np.mean(success_rates) if success_rates else 0,
            'max_effect_discovered': max(mean_effects) if mean_effects else 0,
            'convergence_speed': _compute_convergence_speed(decisions),
            'exploration_efficiency': _compute_exploration_efficiency(patterns)
        })
    
    return pd.DataFrame(comparison_data)

def _compute_convergence_speed(decisions: List[Dict]) -> float:
    """Compute how quickly controller converges to good interventions"""
    
    effects = [d.get('effect_magnitude', 0) for d in decisions]
    if len(effects) < 10:
        return 0
    
    # Find when effects stabilize (reach 80% of final performance)
    final_mean = np.mean(effects[-10:])  # Last 10 decisions
    target = 0.8 * final_mean
    
    for i in range(10, len(effects)):
        window_mean = np.mean(effects[max(0, i-9):i+1])
        if window_mean >= target:
            return 1.0 - (i / len(effects))  # Earlier convergence = higher score
    
    return 0  # Never converged

def _compute_exploration_efficiency(patterns: Dict[str, Any]) -> float:
    """Compute how efficiently controller explores intervention space"""
    
    exploitation_phases = patterns.get('exploitation_phases', [])
    if not exploitation_phases:
        return 0
    
    # Good exploration balances exploitation and exploration
    exploitation_ratios = [p['exploitation_ratio'] for p in exploitation_phases]
    
    # Ideal exploitation ratio is around 0.6-0.8 (focused but not too narrow)
    ideal_range = (0.6, 0.8)
    efficiency_scores = []
    
    for ratio in exploitation_ratios:
        if ideal_range[0] <= ratio <= ideal_range[1]:
            efficiency_scores.append(1.0)
        elif ratio < ideal_range[0]:
            efficiency_scores.append(ratio / ideal_range[0])
        else:
            efficiency_scores.append(ideal_range[1] / ratio)
    
    return np.mean(efficiency_scores)

def plot_discovery_curves(controller_results: Dict[str, Dict], output_dir: str):
    """Plot learning curves for different controllers"""
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Mean Effect Discovery
    plt.subplot(2, 3, 1)
    for controller_name, results in controller_results.items():
        decisions = results['manifest'].get('controller_decisions', [])
        if decisions:
            efficiency = compute_discovery_efficiency(decisions)
            plt.plot(efficiency['mean_effect_discovery'], label=controller_name, linewidth=2)
    
    plt.xlabel('Intervention Number')
    plt.ylabel('Cumulative Mean Effect')
    plt.title('Effect Discovery Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: High Effect Discovery Rate
    plt.subplot(2, 3, 2)
    for controller_name, results in controller_results.items():
        decisions = results['manifest'].get('controller_decisions', [])
        if decisions:
            efficiency = compute_discovery_efficiency(decisions)
            plt.plot(efficiency['high_effect_discovery_rate'], label=controller_name, linewidth=2)
    
    plt.xlabel('Intervention Number')
    plt.ylabel('High Effect Discovery Rate')
    plt.title('High-Impact Discovery Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Intervention Diversity
    plt.subplot(2, 3, 3)
    for controller_name, results in controller_results.items():
        decisions = results['manifest'].get('controller_decisions', [])
        if decisions:
            efficiency = compute_discovery_efficiency(decisions)
            plt.plot(efficiency['intervention_diversity'], label=controller_name, linewidth=2)
    
    plt.xlabel('Intervention Number')
    plt.ylabel('Intervention Diversity')
    plt.title('Exploration Diversity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Effect Timeline
    plt.subplot(2, 3, 4)
    for controller_name, results in controller_results.items():
        decisions = results['manifest'].get('controller_decisions', [])
        if decisions:
            patterns = analyze_intervention_selection_patterns(decisions)
            plt.plot(patterns['effect_timeline'], label=controller_name, alpha=0.7)
    
    plt.xlabel('Intervention Number')
    plt.ylabel('Effect Magnitude')
    plt.title('Effect Discovery Timeline')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Controller Performance Comparison
    plt.subplot(2, 3, 5)
    comparison_df = compare_controller_performance(controller_results)
    
    metrics = ['final_mean_effect', 'final_high_effect_rate', 'convergence_speed', 'exploration_efficiency']
    x_pos = np.arange(len(comparison_df))
    
    width = 0.2
    for i, metric in enumerate(metrics):
        plt.bar(x_pos + i * width, comparison_df[metric], width, label=metric.replace('_', ' ').title())
    
    plt.xlabel('Controller')
    plt.ylabel('Performance Score')
    plt.title('Controller Performance Comparison')
    plt.xticks(x_pos + width * 1.5, comparison_df['controller'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Intervention Selection Heatmap
    plt.subplot(2, 3, 6)
    
    # Create heatmap of intervention selections over time
    all_interventions = set()
    for controller_name, results in controller_results.items():
        decisions = results['manifest'].get('controller_decisions', [])
        for decision in decisions:
            all_interventions.add(decision.get('intervention_chosen', ''))
    
    all_interventions = sorted(list(all_interventions))
    
    # For first controller, create timeline heatmap
    if controller_results:
        first_controller = list(controller_results.keys())[0]
        decisions = controller_results[first_controller]['manifest'].get('controller_decisions', [])
        patterns = analyze_intervention_selection_patterns(decisions)
        
        # Create selection matrix
        selection_matrix = []
        for window_data in patterns['selection_evolution']:
            row = [window_data.get(intervention, 0) for intervention in all_interventions]
            selection_matrix.append(row)
        
        if selection_matrix:
            sns.heatmap(np.array(selection_matrix).T, 
                       xticklabels=range(len(selection_matrix)),
                       yticklabels=all_interventions,
                       cmap='viridis', cbar_kws={'label': 'Selection Count'})
            plt.xlabel('Time Window')
            plt.ylabel('Intervention Type')
            plt.title(f'Intervention Selection Pattern ({first_controller})')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'active_intervention_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_analysis_report(controller_results: Dict[str, Dict], output_dir: str):
    """Generate comprehensive analysis report"""
    
    comparison_df = compare_controller_performance(controller_results)
    
    report_lines = [
        "# Active Intervention Controller Analysis Report\n",
        f"Generated: {pd.Timestamp.now()}\n",
        f"Controllers analyzed: {len(controller_results)}\n\n",
        
        "## Performance Summary\n\n",
        comparison_df.to_string(index=False),
        "\n\n",
        
        "## Key Findings\n\n"
    ]
    
    # Identify best performers
    best_overall = comparison_df.loc[comparison_df['overall_mean_effect'].idxmax()]
    best_convergence = comparison_df.loc[comparison_df['convergence_speed'].idxmax()]
    best_exploration = comparison_df.loc[comparison_df['exploration_efficiency'].idxmax()]
    
    report_lines.extend([
        f"🏆 **Best Overall Performance**: {best_overall['controller']} "
        f"(mean effect: {best_overall['overall_mean_effect']:.3f})\n\n",
        
        f"⚡ **Fastest Convergence**: {best_convergence['controller']} "
        f"(convergence speed: {best_convergence['convergence_speed']:.3f})\n\n",
        
        f"🔍 **Most Efficient Explorer**: {best_exploration['controller']} "
        f"(exploration efficiency: {best_exploration['exploration_efficiency']:.3f})\n\n",
    ])
    
    # Detailed analysis for each controller
    report_lines.append("## Detailed Controller Analysis\n\n")
    
    for controller_name, results in controller_results.items():
        manifest = results['manifest']
        decisions = manifest.get('controller_decisions', [])
        
        if not decisions:
            continue
        
        efficiency = compute_discovery_efficiency(decisions)
        patterns = analyze_intervention_selection_patterns(decisions)
        
        report_lines.extend([
            f"### {controller_name}\n\n",
            f"- **Total Interventions**: {len(decisions)}\n",
            f"- **Final Mean Effect**: {efficiency.get('final_mean_effect', 0):.3f}\n",
            f"- **High Effect Rate**: {efficiency.get('final_high_effect_rate', 0):.2%}\n",
            f"- **Intervention Diversity**: {efficiency.get('final_diversity', 0):.3f}\n",
        ])
        
        # Most selected interventions
        intervention_counts = {}
        for decision in decisions:
            intervention = decision.get('intervention_chosen', '')
            intervention_counts[intervention] = intervention_counts.get(intervention, 0) + 1
        
        top_interventions = sorted(intervention_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        report_lines.append(f"- **Top Interventions**: ")
        for intervention, count in top_interventions:
            report_lines.append(f"{intervention} ({count}), ")
        report_lines.append("\n\n")
    
    # Save report
    report_path = os.path.join(output_dir, 'active_intervention_analysis_report.txt')
    with open(report_path, 'w') as f:
        f.writelines(report_lines)
    
    print(f"Analysis report saved: {report_path}")
    
    # Save detailed comparison data
    comparison_path = os.path.join(output_dir, 'controller_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Comparison data saved: {comparison_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze active intervention results")
    parser.add_argument('--data_dirs', nargs='+', required=True,
                       help='Directories containing active intervention results')
    parser.add_argument('--controller_names', nargs='+',
                       help='Names for controllers (default: use directory names)')
    parser.add_argument('--output_dir', type=str, default='analysis/active_interventions',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data from all controllers
    controller_results = {}
    
    for i, data_dir in enumerate(args.data_dirs):
        if args.controller_names and i < len(args.controller_names):
            controller_name = args.controller_names[i]
        else:
            controller_name = os.path.basename(data_dir.rstrip('/'))
        
        try:
            manifest = load_active_intervention_data(data_dir)
            controller_results[controller_name] = {
                'manifest': manifest,
                'data_dir': data_dir
            }
            print(f"Loaded data for {controller_name}: {len(manifest.get('controller_decisions', []))} interventions")
        except Exception as e:
            print(f"Error loading data from {data_dir}: {e}")
    
    if not controller_results:
        print("No valid controller data found!")
        return
    
    # Generate analysis
    print(f"\nAnalyzing {len(controller_results)} controllers...")
    
    # Plot discovery curves
    plot_discovery_curves(controller_results, args.output_dir)
    print(f"Discovery curves saved: {args.output_dir}/active_intervention_analysis.png")
    
    # Generate report
    generate_analysis_report(controller_results, args.output_dir)
    
    print(f"\nAnalysis complete! Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main()