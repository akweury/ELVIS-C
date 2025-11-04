#!/usr/bin/env python3
"""
Analyze Causal Effects from Intervention Pairs Dataset

This script analyzes the causal effects discovered in intervention pairs,
providing statistical analysis and visualizations of causal relationships.

Usage:
    python analyze_causal_effects.py --manifest_path intervention_manifest.json --output_dir results/
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_intervention_data(manifest_path):
    """Load intervention pairs data from manifest"""
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Convert pairs data to DataFrame
    pairs_data = []
    for pair in manifest['pairs']:
        baseline = pair['baseline']
        intervention = pair['intervention']
        effect = pair['causal_effect']
        
        row = {
            'pair_idx': pair['pair_idx'],
            'intervention_type': pair['intervention_type'],
            'seed': pair['seed'],
            
            # Baseline
            'baseline_jam_type': baseline['jam_type'],
            'baseline_exit_ratio': baseline['exit_statistics']['exit_ratio'],
            'baseline_total_exited': baseline['exit_statistics']['total_exited'],
            'baseline_total_spawned': baseline['exit_statistics']['total_spawned'],
            
            # Intervention
            'intervention_jam_type': intervention['jam_type'],
            'intervention_exit_ratio': intervention['exit_statistics']['exit_ratio'],
            'intervention_total_exited': intervention['exit_statistics']['total_exited'],
            'intervention_total_spawned': intervention['exit_statistics']['total_spawned'],
            
            # Causal Effects
            'jam_type_changed': effect['jam_type_changed'],
            'exit_ratio_change': effect['exit_ratio_change'],
            'total_exited_change': effect['total_exited_change'],
            
            # Parameter changes (extract key changed parameters)
            'baseline_wind_strength': baseline['params']['wind_strength'],
            'intervention_wind_strength': intervention['params']['wind_strength'],
            'baseline_hole_diameter': baseline['params']['hole_diameter'],
            'intervention_hole_diameter': intervention['params']['hole_diameter'],
            'baseline_num_circles': baseline['params']['num_circles'],
            'intervention_num_circles': intervention['params']['num_circles'],
            'baseline_spawn_rate': baseline['params']['spawn_rate'],
            'intervention_spawn_rate': intervention['params']['spawn_rate'],
        }
        
        pairs_data.append(row)
    
    return pd.DataFrame(pairs_data), manifest

def analyze_causal_effects(df, output_dir):
    """Analyze causal effects and generate statistics"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Overall causal effect statistics
    total_pairs = len(df)
    jam_changes = df['jam_type_changed'].sum()
    avg_exit_change = df['exit_ratio_change'].mean()
    
    print(f"📊 CAUSAL EFFECTS ANALYSIS")
    print(f"   Total intervention pairs: {total_pairs}")
    print(f"   Jam type changes: {jam_changes} ({jam_changes/total_pairs*100:.1f}%)")
    print(f"   Average exit ratio change: {avg_exit_change:+.3f}")
    
    # 2. Effect by intervention type
    intervention_stats = df.groupby('intervention_type').agg({
        'jam_type_changed': ['count', 'sum', 'mean'],
        'exit_ratio_change': ['mean', 'std', 'min', 'max'],
        'total_exited_change': ['mean', 'std']
    }).round(3)
    
    print(f"\\n🎯 INTERVENTION TYPE ANALYSIS:")
    for intervention in df['intervention_type'].unique():
        subset = df[df['intervention_type'] == intervention]
        changes = subset['jam_type_changed'].sum()
        total = len(subset)
        avg_change = subset['exit_ratio_change'].mean()
        print(f"   {intervention}: {changes}/{total} changes ({changes/total*100:.1f}%), avg Δ={avg_change:+.3f}")
    
    # 3. Statistical significance tests
    print(f"\\n📈 STATISTICAL SIGNIFICANCE:")
    
    # Test if interventions have significant effects
    for intervention in df['intervention_type'].unique():
        subset = df[df['intervention_type'] == intervention]
        if len(subset) >= 3:  # Need sufficient samples
            # Test if exit ratio changes are significantly different from zero
            t_stat, p_value = stats.ttest_1samp(subset['exit_ratio_change'], 0)
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"   {intervention}: t={t_stat:.3f}, p={p_value:.3f} {significance}")
    
    # 4. Generate visualizations
    create_causal_effect_plots(df, output_dir)
    
    # 5. Save detailed analysis
    analysis_path = os.path.join(output_dir, 'causal_analysis_report.txt')
    with open(analysis_path, 'w') as f:
        f.write("CAUSAL EFFECTS ANALYSIS REPORT\\n")
        f.write("=" * 50 + "\\n\\n")
        
        f.write(f"Dataset Summary:\\n")
        f.write(f"- Total intervention pairs: {total_pairs}\\n")
        f.write(f"- Jam type changes observed: {jam_changes} ({jam_changes/total_pairs*100:.1f}%)\\n")
        f.write(f"- Average exit ratio change: {avg_exit_change:+.3f}\\n\\n")
        
        f.write("Intervention Effects by Type:\\n")
        f.write("-" * 30 + "\\n")
        for intervention in df['intervention_type'].unique():
            subset = df[df['intervention_type'] == intervention]
            f.write(f"\\n{intervention}:\\n")
            f.write(f"  Sample size: {len(subset)}\\n")
            f.write(f"  Jam type changes: {subset['jam_type_changed'].sum()} ({subset['jam_type_changed'].mean()*100:.1f}%)\\n")
            f.write(f"  Exit ratio change: {subset['exit_ratio_change'].mean():+.3f} ± {subset['exit_ratio_change'].std():.3f}\\n")
            f.write(f"  Range: [{subset['exit_ratio_change'].min():+.3f}, {subset['exit_ratio_change'].max():+.3f}]\\n")
            
            # Most effective cases
            if subset['jam_type_changed'].any():
                changed_cases = subset[subset['jam_type_changed']]
                f.write(f"  Transitions observed:\\n")
                for _, case in changed_cases.iterrows():
                    f.write(f"    {case['baseline_jam_type']} → {case['intervention_jam_type']} (Δ={case['exit_ratio_change']:+.3f})\\n")
        
        f.write("\\nKey Findings:\\n")
        f.write("-" * 15 + "\\n")
        
        # Find most effective interventions
        effect_sizes = df.groupby('intervention_type')['exit_ratio_change'].apply(lambda x: np.abs(x).mean())
        most_effective = effect_sizes.sort_values(ascending=False).head(3)
        
        f.write("Most effective interventions (by average effect magnitude):\\n")
        for intervention, effect in most_effective.items():
            f.write(f"  1. {intervention}: {effect:.3f}\\n")
        
        # Find interventions that cause jam type changes
        jam_change_rates = df.groupby('intervention_type')['jam_type_changed'].mean()
        highest_change_rate = jam_change_rates.sort_values(ascending=False).head(3)
        
        f.write("\\nInterventions most likely to change jam type:\\n")
        for intervention, rate in highest_change_rate.items():
            f.write(f"  1. {intervention}: {rate*100:.1f}%\\n")
    
    print(f"✅ Detailed analysis saved: {analysis_path}")
    
    return intervention_stats

def create_causal_effect_plots(df, output_dir):
    """Create visualizations of causal effects"""
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Effect magnitude by intervention type
    plt.figure(figsize=(12, 8))
    
    # Calculate effect magnitude for each intervention type
    effect_data = []
    for intervention in df['intervention_type'].unique():
        subset = df[df['intervention_type'] == intervention]
        for _, row in subset.iterrows():
            effect_data.append({
                'intervention': intervention,
                'exit_ratio_change': row['exit_ratio_change'],
                'jam_changed': 'Yes' if row['jam_type_changed'] else 'No'
            })
    
    effect_df = pd.DataFrame(effect_data)
    
    # Create box plot
    plt.subplot(2, 2, 1)
    sns.boxplot(data=effect_df, x='intervention', y='exit_ratio_change')
    plt.xticks(rotation=45, ha='right')
    plt.title('Exit Ratio Change by Intervention Type', fontweight='bold')
    plt.ylabel('Exit Ratio Change')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # 2. Jam type change frequency
    plt.subplot(2, 2, 2)
    jam_change_freq = df.groupby('intervention_type')['jam_type_changed'].agg(['sum', 'count'])
    jam_change_freq['rate'] = jam_change_freq['sum'] / jam_change_freq['count']
    
    bars = plt.bar(range(len(jam_change_freq)), jam_change_freq['rate'])
    plt.xticks(range(len(jam_change_freq)), jam_change_freq.index, rotation=45, ha='right')
    plt.title('Jam Type Change Rate by Intervention', fontweight='bold')
    plt.ylabel('Fraction of Pairs with Jam Type Change')
    
    # Color bars by effect magnitude
    for i, bar in enumerate(bars):
        if jam_change_freq['rate'].iloc[i] > 0.5:
            bar.set_color('red')
        elif jam_change_freq['rate'].iloc[i] > 0.2:
            bar.set_color('orange')
        else:
            bar.set_color('lightblue')
    
    # 3. Effect magnitude distribution
    plt.subplot(2, 2, 3)
    plt.hist(df['exit_ratio_change'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    plt.title('Distribution of Exit Ratio Changes', fontweight='bold')
    plt.xlabel('Exit Ratio Change')
    plt.ylabel('Frequency')
    
    # 4. Scatter plot: baseline vs intervention outcomes
    plt.subplot(2, 2, 4)
    colors = {'no_jam': 'green', 'partial_jam': 'orange', 'full_jam': 'red'}
    
    for jam_type in colors:
        subset = df[df['baseline_jam_type'] == jam_type]
        plt.scatter(subset['baseline_exit_ratio'], subset['intervention_exit_ratio'], 
                   c=colors[jam_type], label=f'Baseline: {jam_type}', alpha=0.7, s=50)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='No change line')
    plt.xlabel('Baseline Exit Ratio')
    plt.ylabel('Intervention Exit Ratio')
    plt.title('Baseline vs Intervention Outcomes', fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'causal_effects_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Causal effects plots saved: {plot_path}")
    
    # 5. Intervention transition matrix
    create_transition_matrix(df, output_dir)

def create_transition_matrix(df, output_dir):
    """Create jam type transition matrix heatmap"""
    
    # Create transition matrix
    jam_types = ['no_jam', 'partial_jam', 'full_jam']
    transition_matrix = np.zeros((len(jam_types), len(jam_types)))
    transition_counts = np.zeros((len(jam_types), len(jam_types)))
    
    for _, row in df.iterrows():
        baseline_idx = jam_types.index(row['baseline_jam_type'])
        intervention_idx = jam_types.index(row['intervention_jam_type'])
        transition_counts[baseline_idx, intervention_idx] += 1
        transition_matrix[baseline_idx, intervention_idx] += row['exit_ratio_change']
    
    # Normalize by counts (average effect per transition)
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_transition_matrix = np.divide(transition_matrix, transition_counts, 
                                        out=np.zeros_like(transition_matrix), 
                                        where=transition_counts!=0)
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    
    # Plot transition counts
    plt.subplot(1, 2, 1)
    sns.heatmap(transition_counts, 
                xticklabels=jam_types, 
                yticklabels=jam_types,
                annot=True, 
                fmt='g',
                cmap='Blues',
                cbar_kws={'label': 'Number of Transitions'})
    plt.title('Jam Type Transition Counts', fontweight='bold')
    plt.xlabel('Intervention Outcome')
    plt.ylabel('Baseline Outcome')
    
    # Plot average effects
    plt.subplot(1, 2, 2)
    sns.heatmap(avg_transition_matrix, 
                xticklabels=jam_types, 
                yticklabels=jam_types,
                annot=True, 
                fmt='.3f',
                cmap='RdBu_r',
                center=0,
                cbar_kws={'label': 'Average Exit Ratio Change'})
    plt.title('Average Causal Effects by Transition', fontweight='bold')
    plt.xlabel('Intervention Outcome')
    plt.ylabel('Baseline Outcome')
    
    plt.tight_layout()
    transition_path = os.path.join(output_dir, 'jam_type_transitions.png')
    plt.savefig(transition_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Transition matrix saved: {transition_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze causal effects from intervention pairs")
    parser.add_argument("--manifest_path", type=str, required=True,
                       help="Path to intervention_manifest.json file")
    parser.add_argument("--output_dir", type=str, default="causal_analysis",
                       help="Output directory for analysis results")
    
    args = parser.parse_args()
    
    print(f"🔍 Loading intervention pairs from: {args.manifest_path}")
    
    try:
        df, manifest = load_intervention_data(args.manifest_path)
        print(f"✅ Loaded {len(df)} intervention pairs")
        
        # Run analysis
        stats = analyze_causal_effects(df, args.output_dir)
        
        print(f"\\n🎉 Causal analysis complete!")
        print(f"📁 Results saved in: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()