#!/usr/bin/env python3
"""
Parameter Independence Analysis for Falling Circles Dataset

Purpose:
Verify that all physics parameters in the generated dataset are independently sampled,
ensuring unbiased causal modeling.

Usage:
    python analyze_parameter_independence.py --manifest_path dataset_manifest.json --output_dir results/
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_dataset_parameters(manifest_path):
    """
    Load dataset parameters from manifest file
    
    Returns:
        pd.DataFrame: DataFrame with all physics parameters and outcomes
    """
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    data_rows = []
    
    for video in manifest['videos']:
        params = video['params']
        
        # Extract relevant physics parameters
        row = {
            # Core physics parameters
            'wind_strength': params['wind_strength'],
            'wind_direction': params['wind_direction'],
            'hole_diameter': params['hole_diameter'],
            'hole_x_position': params['hole_x_position'],
            'circle_size_min': params['circle_size_min'],
            'circle_size_max': params['circle_size_max'],
            'gravity': params['gravity'],
            'spawn_rate': params['spawn_rate'],
            'num_circles': params['num_circles'],
            'noise_level': params['noise_level'],
            
            # Derived parameters
            'circle_radius_mean': (params['circle_size_min'] + params['circle_size_max']) / 2,
            'circle_radius_range': params['circle_size_max'] - params['circle_size_min'],
            'circle_radius_std': (params['circle_size_max'] - params['circle_size_min']) / 6,  # Approximate for uniform distribution
            
            # Note: slope_angle is not directly stored but could be computed from funnel geometry
            # For this analysis, we'll use hole_x_position as proxy for geometric variation
            
            # Outcome variable
            'jam_label': video['actual_jam_type'],
            
            # Additional metadata
            'video_idx': video['video_idx'],
            'split': video['split'],
            'seed': video['seed']
        }
        
        # Add exit statistics if available
        if 'exit_statistics' in video:
            row.update({
                'total_spawned': video['exit_statistics']['total_spawned'],
                'total_exited': video['exit_statistics']['total_exited'],
                'exit_ratio': video['exit_statistics']['exit_ratio'],
                'final_stuck_count': video['exit_statistics']['final_stuck_count']
            })
        
        data_rows.append(row)
    
    return pd.DataFrame(data_rows)

def compute_correlation_matrix(df, continuous_params):
    """
    Compute pairwise Pearson correlation coefficients for continuous parameters
    """
    correlation_matrix = df[continuous_params].corr(method='pearson')
    
    # Get correlation values with p-values
    correlation_details = {}
    n = len(continuous_params)
    
    for i in range(n):
        for j in range(i+1, n):
            param1, param2 = continuous_params[i], continuous_params[j]
            corr, p_value = pearsonr(df[param1], df[param2])
            correlation_details[(param1, param2)] = {
                'correlation': corr,
                'p_value': p_value,
                'abs_correlation': abs(corr)
            }
    
    return correlation_matrix, correlation_details

def compute_mutual_information_matrix(df, continuous_params, categorical_params):
    """
    Compute mutual information between all parameter pairs
    """
    all_params = continuous_params + categorical_params
    n_params = len(all_params)
    
    mi_matrix = np.zeros((n_params, n_params))
    mi_details = {}
    
    # Encode categorical variables
    label_encoders = {}
    df_encoded = df.copy()
    
    for param in categorical_params:
        le = LabelEncoder()
        df_encoded[param] = le.fit_transform(df[param].astype(str))
        label_encoders[param] = le
    
    for i in range(n_params):
        for j in range(n_params):
            if i == j:
                mi_matrix[i, j] = 1.0  # Perfect mutual information with self
                continue
                
            param1, param2 = all_params[i], all_params[j]
            
            # Compute mutual information
            if param2 in categorical_params:
                # Target is categorical
                mi_score = mutual_info_classif(
                    df_encoded[[param1]].values, 
                    df_encoded[param2].values,
                    random_state=42
                )[0]
            else:
                # Target is continuous
                mi_score = mutual_info_regression(
                    df_encoded[[param1]].values, 
                    df_encoded[param2].values,
                    random_state=42
                )[0]
            
            mi_matrix[i, j] = mi_score
            
            # Store details for pairs (avoid duplicates)
            if i < j:
                mi_details[(param1, param2)] = {
                    'mutual_information': mi_score,
                    'param1_type': 'categorical' if param1 in categorical_params else 'continuous',
                    'param2_type': 'categorical' if param2 in categorical_params else 'continuous'
                }
    
    # Create DataFrame for better visualization
    mi_df = pd.DataFrame(mi_matrix, index=all_params, columns=all_params)
    
    return mi_df, mi_details, label_encoders

def generate_heatmaps(correlation_matrix, mi_df, output_dir):
    """
    Generate correlation and mutual information heatmaps
    """
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("coolwarm")
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Correlation heatmap
    mask_corr = np.triu(np.ones_like(correlation_matrix.corr(), dtype=bool))
    sns.heatmap(correlation_matrix, 
                mask=mask_corr,
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                fmt='.3f',
                cbar_kws={"shrink": .8},
                ax=ax1)
    ax1.set_title('Pearson Correlation Matrix\\n(Continuous Parameters Only)', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.tick_params(axis='y', rotation=0)
    
    # Mutual Information heatmap
    mask_mi = np.triu(np.ones_like(mi_df, dtype=bool))
    sns.heatmap(mi_df, 
                mask=mask_mi,
                annot=True, 
                cmap='viridis', 
                square=True,
                fmt='.3f',
                cbar_kws={"shrink": .8},
                ax=ax2)
    ax2.set_title('Mutual Information Matrix\\n(All Parameters)', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    
    # Save the plot
    heatmap_path = os.path.join(output_dir, 'independence_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Heatmap saved to: {heatmap_path}")
    
    return heatmap_path

def analyze_independence(df, correlation_threshold=0.1, mi_threshold=0.05, output_dir="results"):
    """
    Main analysis function to check parameter independence
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define parameter categories
    continuous_params = [
        'wind_strength', 'hole_diameter', 'hole_x_position',
        'circle_size_min', 'circle_size_max', 'circle_radius_mean', 
        'circle_radius_range', 'circle_radius_std',
        'gravity', 'spawn_rate', 'noise_level', 'num_circles'
    ]
    
    categorical_params = ['wind_direction', 'jam_label', 'split']
    
    # Filter parameters that exist in the dataset
    continuous_params = [p for p in continuous_params if p in df.columns]
    categorical_params = [p for p in categorical_params if p in df.columns]
    
    print(f"📊 Analyzing {len(df)} videos with {len(continuous_params)} continuous and {len(categorical_params)} categorical parameters\\n")
    
    # 1. Compute correlation matrix
    print("1. Computing pairwise Pearson correlations...")
    correlation_matrix, correlation_details = compute_correlation_matrix(df, continuous_params)
    
    # 2. Compute mutual information matrix
    print("2. Computing mutual information matrix...")
    mi_df, mi_details, label_encoders = compute_mutual_information_matrix(df, continuous_params, categorical_params)
    
    # 3. Generate heatmaps
    print("3. Generating heatmaps...")
    heatmap_path = generate_heatmaps(correlation_matrix, mi_df, output_dir)
    
    # 4. Check for violations
    print("4. Checking for independence violations...")
    violations = []
    
    # Check correlation violations
    for (param1, param2), details in correlation_details.items():
        if details['abs_correlation'] > correlation_threshold:
            violations.append({
                'type': 'correlation',
                'param1': param1,
                'param2': param2,
                'value': details['correlation'],
                'abs_value': details['abs_correlation'],
                'p_value': details['p_value'],
                'threshold': correlation_threshold
            })
    
    # Check mutual information violations
    for (param1, param2), details in mi_details.items():
        if details['mutual_information'] > mi_threshold:
            violations.append({
                'type': 'mutual_information',
                'param1': param1,
                'param2': param2,
                'value': details['mutual_information'],
                'abs_value': details['mutual_information'],
                'threshold': mi_threshold,
                'param1_type': details['param1_type'],
                'param2_type': details['param2_type']
            })
    
    # 5. Save results
    print("5. Saving results...")
    
    # Save correlation matrix
    correlation_path = os.path.join(output_dir, 'independence_matrix.csv')
    
    # Combine correlation and MI data for comprehensive matrix
    all_params = continuous_params + categorical_params
    combined_matrix = pd.DataFrame(index=all_params, columns=all_params)
    
    # Fill with correlation data where available
    for param1 in continuous_params:
        for param2 in continuous_params:
            if param1 in correlation_matrix.index and param2 in correlation_matrix.columns:
                combined_matrix.loc[param1, param2] = correlation_matrix.loc[param1, param2]
    
    # Fill with MI data for remaining pairs
    for param1 in all_params:
        for param2 in all_params:
            if pd.isna(combined_matrix.loc[param1, param2]):
                combined_matrix.loc[param1, param2] = mi_df.loc[param1, param2]
    
    combined_matrix.to_csv(correlation_path)
    print(f"✅ Independence matrix saved to: {correlation_path}")
    
    # Generate detailed report
    report_path = os.path.join(output_dir, 'independence_report.txt')
    with open(report_path, 'w') as f:
        f.write("PHYSICS PARAMETER INDEPENDENCE ANALYSIS REPORT\\n")
        f.write("=" * 60 + "\\n\\n")
        
        f.write(f"Dataset Summary:\\n")
        f.write(f"- Total videos analyzed: {len(df)}\\n")
        f.write(f"- Continuous parameters: {len(continuous_params)}\\n")
        f.write(f"- Categorical parameters: {len(categorical_params)}\\n")
        f.write(f"- Correlation threshold: |r| > {correlation_threshold}\\n")
        f.write(f"- Mutual information threshold: MI > {mi_threshold}\\n\\n")
        
        f.write(f"Parameters analyzed:\\n")
        f.write(f"Continuous: {', '.join(continuous_params)}\\n")
        f.write(f"Categorical: {', '.join(categorical_params)}\\n\\n")
        
        # Summary statistics
        f.write("DATASET STATISTICS:\\n")
        f.write("-" * 30 + "\\n")
        for param in continuous_params:
            stats = df[param].describe()
            f.write(f"{param}:\\n")
            f.write(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}\\n")
            f.write(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]\\n")
        
        f.write("\\nCategorical distribution:\\n")
        for param in categorical_params:
            counts = df[param].value_counts()
            f.write(f"{param}: {dict(counts)}\\n")
        
        f.write("\\n" + "=" * 60 + "\\n")
        f.write("INDEPENDENCE ANALYSIS RESULTS\\n")
        f.write("=" * 60 + "\\n\\n")
        
        if violations:
            f.write(f"⚠️  FOUND {len(violations)} INDEPENDENCE VIOLATIONS:\\n\\n")
            
            # Group by type
            corr_violations = [v for v in violations if v['type'] == 'correlation']
            mi_violations = [v for v in violations if v['type'] == 'mutual_information']
            
            if corr_violations:
                f.write(f"CORRELATION VIOLATIONS (|r| > {correlation_threshold}):\\n")
                f.write("-" * 40 + "\\n")
                for v in sorted(corr_violations, key=lambda x: x['abs_value'], reverse=True):
                    f.write(f"• {v['param1']} ↔ {v['param2']}: r = {v['value']:.4f} (p = {v['p_value']:.4f})\\n")
                f.write("\\n")
            
            if mi_violations:
                f.write(f"MUTUAL INFORMATION VIOLATIONS (MI > {mi_threshold}):\\n")
                f.write("-" * 40 + "\\n")
                for v in sorted(mi_violations, key=lambda x: x['abs_value'], reverse=True):
                    f.write(f"• {v['param1']} ({v['param1_type']}) ↔ {v['param2']} ({v['param2_type']}): MI = {v['value']:.4f}\\n")
                f.write("\\n")
            
        else:
            f.write("✅ NO INDEPENDENCE VIOLATIONS FOUND\\n\\n")
            f.write("All parameter pairs satisfy independence criteria:\\n")
            f.write(f"- All |correlations| ≤ {correlation_threshold}\\n")
            f.write(f"- All mutual information ≤ {mi_threshold}\\n\\n")
        
        # Detailed statistics
        f.write("DETAILED STATISTICS:\\n")
        f.write("-" * 30 + "\\n\\n")
        
        f.write("Correlation Statistics (continuous parameters only):\\n")
        corr_values = [details['abs_correlation'] for details in correlation_details.values()]
        if corr_values:
            f.write(f"- Max |correlation|: {max(corr_values):.4f}\\n")
            f.write(f"- Mean |correlation|: {np.mean(corr_values):.4f}\\n")
            f.write(f"- Std |correlation|: {np.std(corr_values):.4f}\\n")
        
        f.write("\\nMutual Information Statistics (all parameter pairs):\\n")
        mi_values = [details['mutual_information'] for details in mi_details.values()]
        if mi_values:
            f.write(f"- Max MI: {max(mi_values):.4f}\\n")
            f.write(f"- Mean MI: {np.mean(mi_values):.4f}\\n")
            f.write(f"- Std MI: {np.std(mi_values):.4f}\\n")
        
        f.write(f"\\nAnalysis completed successfully.\\n")
        f.write(f"Generated files:\\n")
        f.write(f"- {correlation_path}\\n")
        f.write(f"- {heatmap_path}\\n")
        f.write(f"- {report_path}\\n")
    
    print(f"✅ Detailed report saved to: {report_path}")
    
    # Print warnings for violations
    if violations:
        print(f"\\n⚠️  WARNING: Found {len(violations)} independence violations!")
        for v in violations[:5]:  # Show first 5
            if v['type'] == 'correlation':
                print(f"   • {v['param1']} ↔ {v['param2']}: |r| = {v['abs_value']:.4f} > {v['threshold']}")
            else:
                print(f"   • {v['param1']} ↔ {v['param2']}: MI = {v['abs_value']:.4f} > {v['threshold']}")
        if len(violations) > 5:
            print(f"   ... and {len(violations) - 5} more (see report for details)")
    else:
        print("\\n✅ SUCCESS: All parameters appear to be independently sampled!")
    
    return {
        'correlation_matrix': correlation_matrix,
        'mi_matrix': mi_df,
        'violations': violations,
        'n_videos': len(df),
        'continuous_params': continuous_params,
        'categorical_params': categorical_params
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze parameter independence in falling circles dataset")
    parser.add_argument("--manifest_path", type=str, required=True,
                       help="Path to dataset_manifest.json file")
    parser.add_argument("--output_dir", type=str, default="independence_analysis",
                       help="Output directory for results")
    parser.add_argument("--correlation_threshold", type=float, default=0.1,
                       help="Threshold for correlation violations (default: 0.1)")
    parser.add_argument("--mi_threshold", type=float, default=0.05,
                       help="Threshold for mutual information violations (default: 0.05)")
    
    args = parser.parse_args()
    
    print(f"🔍 Loading dataset from: {args.manifest_path}")
    
    # Load dataset
    try:
        df = load_dataset_parameters(args.manifest_path)
        print(f"✅ Successfully loaded {len(df)} videos")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Run analysis
    try:
        results = analyze_independence(
            df, 
            correlation_threshold=args.correlation_threshold,
            mi_threshold=args.mi_threshold,
            output_dir=args.output_dir
        )
        print(f"\\n🎉 Analysis complete! Results saved in: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()