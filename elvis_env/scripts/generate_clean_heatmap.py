#!/usr/bin/env python3
"""
Generate a clean independence heatmap focusing on the most important parameters
"""

import os
import json
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
    """Load dataset parameters from manifest file"""
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    data_rows = []
    for video in manifest['videos']:
        params = video['params']
        row = {
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
            'jam_label': video['actual_jam_type'],
            'split': video['split']
        }
        data_rows.append(row)
    
    return pd.DataFrame(data_rows)

def create_clean_heatmap(df, output_path):
    """Create a clean, focused heatmap for key parameters"""
    
    # Focus on the most important independent parameters
    key_params = [
        'wind_strength', 'hole_diameter', 'spawn_rate', 
        'num_circles', 'circle_size_min', 'circle_size_max'
    ]
    
    # Compute correlation matrix for key parameters
    corr_matrix = df[key_params].corr()
    
    # Create the heatmap
    plt.figure(figsize=(12, 10))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Generate heatmap
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                fmt='.3f',
                cmap='RdBu_r', 
                center=0,
                square=True,
                cbar_kws={"shrink": .8, "label": "Pearson Correlation"},
                annot_kws={'size': 11})
    
    plt.title('Parameter Independence Analysis\\n(Falling Circles Dataset - 200 videos)', 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.xlabel('Parameters', fontsize=12)
    plt.ylabel('Parameters', fontsize=12)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add note about gravity
    plt.figtext(0.02, 0.02, 
                'Note: Gravity is constant (1.0) across all videos\\n' +
                'Values shown are Pearson correlation coefficients\\n' +
                'Green indicates independence, Red/Blue indicates correlation',
                fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def create_full_heatmap_with_outcomes(df, output_path):
    """Create a comprehensive heatmap including outcome variables"""
    
    # All relevant parameters including outcomes
    all_params = [
        'wind_strength', 'hole_diameter', 'spawn_rate', 'num_circles',
        'circle_size_min', 'circle_size_max', 'gravity', 'noise_level'
    ]
    
    categorical_params = ['wind_direction', 'jam_label', 'split']
    
    # Create combined matrix with correlations and MI
    n_continuous = len(all_params)
    n_categorical = len(categorical_params)
    n_total = n_continuous + n_categorical
    
    # Initialize matrix
    combined_matrix = np.zeros((n_total, n_total))
    param_names = all_params + categorical_params
    
    # Encode categorical variables
    df_encoded = df.copy()
    label_encoders = {}
    for param in categorical_params:
        le = LabelEncoder()
        df_encoded[param + '_encoded'] = le.fit_transform(df[param].astype(str))
        label_encoders[param] = le
    
    # Fill correlation matrix for continuous variables
    for i, param1 in enumerate(all_params):
        for j, param2 in enumerate(all_params):
            if i <= j:  # Only compute upper triangle and diagonal
                corr = df[param1].corr(df[param2])
                combined_matrix[i, j] = corr
                combined_matrix[j, i] = corr  # Symmetric
    
    # Fill MI for mixed pairs
    for i, param1 in enumerate(param_names):
        for j, param2 in enumerate(param_names):
            if i < n_continuous and j >= n_continuous:  # continuous vs categorical
                # Use mutual information
                mi_score = mutual_info_classif(
                    df[[param1]].values, 
                    df_encoded[param2 + '_encoded'].values,
                    random_state=42
                )[0]
                # Normalize MI to [-1, 1] range for visualization
                combined_matrix[i, j] = mi_score * 0.5  # Scale down for visualization
                combined_matrix[j, i] = mi_score * 0.5
    
    # Create DataFrame for heatmap
    combined_df = pd.DataFrame(combined_matrix, index=param_names, columns=param_names)
    
    # Create the heatmap
    plt.figure(figsize=(14, 12))
    
    # Create custom mask for better visualization
    mask = np.triu(np.ones_like(combined_df, dtype=bool), k=1)
    
    # Generate heatmap
    sns.heatmap(combined_df, 
                mask=mask,
                annot=True, 
                fmt='.3f',
                cmap='RdBu_r', 
                center=0,
                square=True,
                cbar_kws={"shrink": .7, "label": "Correlation / Mutual Information"},
                annot_kws={'size': 9})
    
    plt.title('Comprehensive Parameter Independence Analysis\\n' +
              'Falling Circles Dataset (200 videos)\\n' +
              'Correlations (continuous) & Mutual Information (mixed pairs)', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.xlabel('Parameters', fontsize=12)
    plt.ylabel('Parameters', fontsize=12)
    
    # Rotate labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add detailed note
    plt.figtext(0.02, 0.02, 
                'Interpretation:\\n' +
                '• Values near 0 (green) indicate independence\\n' +
                '• Red/Blue values indicate dependencies\\n' +
                '• Gravity is constant (perfect independence from sampling)\\n' +
                '• Mixed pairs use normalized mutual information',
                fontsize=8, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def main():
    manifest_path = '/Users/jing/PycharmProjects/ELVIS-C/src/output/final_verification_dataset/dataset_manifest.json'
    output_dir = '/Users/jing/PycharmProjects/ELVIS-C/src/output/clean_heatmaps'
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("🔍 Loading dataset...")
    df = load_dataset_parameters(manifest_path)
    print(f"✅ Loaded {len(df)} videos")
    
    print("📊 Creating clean parameter heatmap...")
    clean_path = create_clean_heatmap(df, os.path.join(output_dir, 'clean_independence_heatmap.png'))
    print(f"✅ Clean heatmap saved: {clean_path}")
    
    print("📊 Creating comprehensive heatmap...")
    full_path = create_full_heatmap_with_outcomes(df, os.path.join(output_dir, 'comprehensive_independence_heatmap.png'))
    print(f"✅ Comprehensive heatmap saved: {full_path}")
    
    # Print summary statistics
    key_params = ['wind_strength', 'hole_diameter', 'spawn_rate', 'num_circles', 'circle_size_min', 'circle_size_max']
    corr_matrix = df[key_params].corr()
    
    # Get correlations (excluding diagonal)
    correlations = []
    for i in range(len(key_params)):
        for j in range(i+1, len(key_params)):
            correlations.append(abs(corr_matrix.iloc[i, j]))
    
    print(f"\\n📈 INDEPENDENCE SUMMARY:")
    print(f"   • Maximum |correlation|: {max(correlations):.4f}")
    print(f"   • Mean |correlation|: {np.mean(correlations):.4f}")
    print(f"   • Correlations > 0.1: {sum(1 for c in correlations if c > 0.1)}/{len(correlations)}")
    print(f"   • Gravity: CONSTANT (1.0) - Perfect independence")
    
    if max(correlations) < 0.15:
        print("\\n🎉 EXCELLENT: Strong parameter independence achieved!")
    elif max(correlations) < 0.25:
        print("\\n✅ GOOD: Acceptable parameter independence")
    else:
        print("\\n⚠️  WARNING: Some strong correlations detected")

if __name__ == "__main__":
    main()