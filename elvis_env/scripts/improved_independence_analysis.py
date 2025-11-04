#!/usr/bin/env python3
"""
Improved Parameter Independence Analysis for Falling Circles Dataset

This version accounts for expected dependencies and focuses on problematic ones.
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

def analyze_parameter_generation_logic():
    """
    Analyze the parameter generation logic to identify expected vs unexpected dependencies
    """
    expected_dependencies = {
        # Derived parameters (by mathematical definition)
        ('circle_size_min', 'circle_size_max'): 'Design: max sampled based on min (min + 2 to 15)',
        ('circle_size_min', 'circle_radius_mean'): 'Mathematical: mean = (min + max) / 2',
        ('circle_size_max', 'circle_radius_mean'): 'Mathematical: mean = (min + max) / 2',
        ('circle_size_min', 'circle_radius_range'): 'Mathematical: range = max - min',
        ('circle_size_max', 'circle_radius_range'): 'Mathematical: range = max - min',
        ('circle_size_min', 'circle_radius_std'): 'Mathematical: std ≈ range / 6',
        ('circle_size_max', 'circle_radius_std'): 'Mathematical: std ≈ range / 6',
        ('circle_radius_range', 'circle_radius_std'): 'Mathematical: std ≈ range / 6',
        ('circle_radius_mean', 'circle_radius_range'): 'Mathematical: both derived from min/max',
        ('circle_radius_mean', 'circle_radius_std'): 'Mathematical: both derived from min/max',
        
        # Outcome dependencies (expected)
        ('hole_diameter', 'jam_label'): 'Expected: larger holes → less jamming',
        ('circle_size_min', 'jam_label'): 'Expected: larger circles → more jamming',
        ('circle_size_max', 'jam_label'): 'Expected: larger circles → more jamming',
        ('circle_radius_mean', 'jam_label'): 'Expected: larger circles → more jamming',
        ('circle_radius_range', 'jam_label'): 'Expected: size variation affects flow',
        ('circle_radius_std', 'jam_label'): 'Expected: size variation affects flow',
        ('wind_strength', 'jam_label'): 'Expected: wind affects outcomes',
        ('gravity', 'jam_label'): 'Expected: gravity affects flow dynamics',
        ('spawn_rate', 'jam_label'): 'Expected: timing affects accumulation',
        
        # Train/test split effects (expected)
        ('noise_level', 'split'): 'Design: different noise ranges for train/test',
        ('circle_color', 'split'): 'Design: different color schemes for train/test',
        ('background_color', 'split'): 'Design: different color schemes for train/test',
    }
    
    # Parameters that should be completely independent
    independent_params = {
        'gravity', 'spawn_rate', 'wind_strength', 'wind_direction', 
        'hole_diameter', 'hole_x_position', 'num_circles', 'noise_level'
    }
    
    return expected_dependencies, independent_params

def classify_dependencies(correlation_details, mi_details, expected_dependencies, independent_params):
    """
    Classify dependencies as expected vs problematic
    """
    problematic_correlations = []
    problematic_mi = []
    expected_found = []
    
    # Check correlations
    for (param1, param2), details in correlation_details.items():
        if details['abs_correlation'] > 0.1:
            dependency_key = (param1, param2) if (param1, param2) in expected_dependencies else (param2, param1)
            
            if dependency_key in expected_dependencies:
                expected_found.append({
                    'params': (param1, param2),
                    'type': 'correlation',
                    'value': details['correlation'],
                    'explanation': expected_dependencies[dependency_key]
                })
            elif param1 in independent_params and param2 in independent_params:
                problematic_correlations.append({
                    'params': (param1, param2),
                    'value': details['correlation'],
                    'abs_value': details['abs_correlation'],
                    'p_value': details['p_value']
                })
    
    # Check mutual information
    for (param1, param2), details in mi_details.items():
        if details['mutual_information'] > 0.05:
            dependency_key = (param1, param2) if (param1, param2) in expected_dependencies else (param2, param1)
            
            if dependency_key not in expected_dependencies:
                # Check if both parameters should be independent
                if ((param1 in independent_params and param2 in independent_params) or
                    (param1 in independent_params and param2 not in ['jam_label', 'split']) or
                    (param2 in independent_params and param1 not in ['jam_label', 'split'])):
                    problematic_mi.append({
                        'params': (param1, param2),
                        'value': details['mutual_information'],
                        'param1_type': details['param1_type'],
                        'param2_type': details['param2_type']
                    })
    
    return problematic_correlations, problematic_mi, expected_found

def generate_improved_report(df, problematic_correlations, problematic_mi, expected_found, output_dir):
    """
    Generate an improved analysis report focusing on actual problems
    """
    report_path = os.path.join(output_dir, 'improved_independence_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("IMPROVED PHYSICS PARAMETER INDEPENDENCE ANALYSIS\\n")
        f.write("=" * 70 + "\\n\\n")
        
        f.write(f"Dataset Summary:\\n")
        f.write(f"- Total videos analyzed: {len(df)}\\n")
        f.write(f"- Analysis focuses on unexpected dependencies\\n")
        f.write(f"- Expected dependencies are documented but not flagged as violations\\n\\n")
        
        # Summary of findings
        total_problematic = len(problematic_correlations) + len(problematic_mi)
        f.write("SUMMARY OF FINDINGS:\\n")
        f.write("-" * 30 + "\\n")
        f.write(f"✅ Expected dependencies found: {len(expected_found)}\\n")
        f.write(f"⚠️  Problematic dependencies: {total_problematic}\\n")
        f.write(f"   - Unexpected correlations: {len(problematic_correlations)}\\n")
        f.write(f"   - Unexpected mutual information: {len(problematic_mi)}\\n\\n")
        
        if total_problematic == 0:
            f.write("🎉 EXCELLENT: No unexpected dependencies found!\\n")
            f.write("The parameter generation appears to be working correctly.\\n")
            f.write("All detected dependencies are either expected by design or due to mathematical relationships.\\n\\n")
        else:
            f.write("🚨 ATTENTION NEEDED: Unexpected dependencies detected!\\n")
            f.write("These suggest issues with the parameter generation process.\\n\\n")
        
        # Detailed problematic dependencies
        if problematic_correlations:
            f.write("UNEXPECTED CORRELATIONS (NEED INVESTIGATION):\\n")
            f.write("=" * 50 + "\\n")
            for item in sorted(problematic_correlations, key=lambda x: x['abs_value'], reverse=True):
                f.write(f"⚠️  {item['params'][0]} ↔ {item['params'][1]}:\\n")
                f.write(f"    Correlation: {item['value']:.4f} (|r| = {item['abs_value']:.4f})\\n")
                f.write(f"    P-value: {item['p_value']:.4f}\\n")
                f.write(f"    Issue: These parameters should be independently sampled\\n\\n")
        
        if problematic_mi:
            f.write("UNEXPECTED MUTUAL INFORMATION (NEED INVESTIGATION):\\n")
            f.write("=" * 55 + "\\n")
            for item in sorted(problematic_mi, key=lambda x: x['value'], reverse=True):
                f.write(f"⚠️  {item['params'][0]} ({item['param1_type']}) ↔ {item['params'][1]} ({item['param2_type']}):\\n")
                f.write(f"    Mutual Information: {item['value']:.4f}\\n")
                f.write(f"    Issue: These parameters should be independently sampled\\n\\n")
        
        # Expected dependencies (documentation)
        f.write("EXPECTED DEPENDENCIES (BY DESIGN):\\n")
        f.write("=" * 40 + "\\n")
        f.write("The following dependencies are expected and indicate correct behavior:\\n\\n")
        
        expected_by_type = {}
        for item in expected_found:
            explanation = item['explanation'].split(':')[0]  # Get category
            if explanation not in expected_by_type:
                expected_by_type[explanation] = []
            expected_by_type[explanation].append(item)
        
        for category, items in expected_by_type.items():
            f.write(f"{category} Dependencies:\\n")
            f.write("-" * (len(category) + 15) + "\\n")
            for item in items:
                f.write(f"✅ {item['params'][0]} ↔ {item['params'][1]}: {item['value']:.4f}\\n")
                f.write(f"   {item['explanation']}\\n")
            f.write("\\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS:\\n")
        f.write("=" * 20 + "\\n")
        
        if total_problematic > 0:
            f.write("🔧 IMMEDIATE ACTIONS NEEDED:\\n")
            f.write("1. Investigate parameter generation code for unexpected dependencies\\n")
            f.write("2. Check for shared random seeds or sequential sampling issues\\n")
            f.write("3. Consider using independent random number generators for each parameter\\n")
            f.write("4. Verify that parameter sampling order doesn't create dependencies\\n\\n")
            
            # Specific recommendations for major issues
            major_issues = [item for item in problematic_correlations if item['abs_value'] > 0.3]
            if major_issues:
                f.write("🚨 CRITICAL ISSUES (|r| > 0.3):\\n")
                for item in major_issues:
                    f.write(f"   • {item['params'][0]} ↔ {item['params'][1]}: Investigate immediately\\n")
                f.write("\\n")
        
        f.write("🔄 ONGOING MONITORING:\\n")
        f.write("1. Run this analysis on larger datasets (500+ videos) for more robust statistics\\n")
        f.write("2. Monitor jam type distribution to ensure balanced outcomes\\n")
        f.write("3. Periodically re-run independence tests after code changes\\n\\n")
        
        f.write("✅ VALIDATION SUCCESS CRITERIA:\\n")
        f.write("- No correlations > 0.1 between independent parameters\\n")
        f.write("- No mutual information > 0.05 between independent parameters\\n")
        f.write("- Balanced distribution of jam types across parameter space\\n")
        f.write("- Mathematical relationships (e.g., mean = (min+max)/2) preserved\\n")
    
    return report_path

def main():
    parser = argparse.ArgumentParser(description="Improved parameter independence analysis")
    parser.add_argument("--manifest_path", type=str, required=True,
                       help="Path to dataset_manifest.json file")
    parser.add_argument("--output_dir", type=str, default="improved_independence_analysis",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Load the original analysis functions
    from analyze_parameter_independence import (
        load_dataset_parameters, compute_correlation_matrix, 
        compute_mutual_information_matrix
    )
    
    print(f"🔍 Loading dataset from: {args.manifest_path}")
    df = load_dataset_parameters(args.manifest_path)
    print(f"✅ Successfully loaded {len(df)} videos")
    
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
    
    # Get expected dependencies
    expected_dependencies, independent_params = analyze_parameter_generation_logic()
    
    # Compute correlations and mutual information
    print("📊 Computing statistical dependencies...")
    correlation_matrix, correlation_details = compute_correlation_matrix(df, continuous_params)
    mi_df, mi_details, _ = compute_mutual_information_matrix(df, continuous_params, categorical_params)
    
    # Classify dependencies
    print("🔍 Classifying dependencies...")
    problematic_correlations, problematic_mi, expected_found = classify_dependencies(
        correlation_details, mi_details, expected_dependencies, independent_params
    )
    
    # Generate improved report
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = generate_improved_report(
        df, problematic_correlations, problematic_mi, expected_found, args.output_dir
    )
    
    # Summary
    total_problematic = len(problematic_correlations) + len(problematic_mi)
    print(f"\\n📋 ANALYSIS COMPLETE:")
    print(f"   ✅ Expected dependencies: {len(expected_found)}")
    print(f"   ⚠️  Problematic dependencies: {total_problematic}")
    
    if total_problematic == 0:
        print("\\n🎉 SUCCESS: Parameter generation appears to be working correctly!")
        print("   All dependencies are either expected by design or mathematical relationships.")
    else:
        print(f"\\n🚨 ATTENTION: Found {total_problematic} unexpected dependencies!")
        print("   Review the detailed report for specific issues to investigate.")
    
    print(f"\\n📄 Detailed report: {report_path}")

if __name__ == "__main__":
    main()