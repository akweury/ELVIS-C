#!/usr/bin/env python3
"""
Comprehensive Audit Script for Falling Circles Datasets
========================================================

This script performs detailed analysis of falling circles datasets to assess:
- Parameter independence and correlations
- Jam type distribution and classification accuracy
- Dataset quality metrics and recommendations
- Label sensitivity analysis
- Visual heatmaps and statistics

Usage:
    python audit_dataset.py <dataset_path>
    python audit_dataset.py <dataset_path> --output custom_audit_dir
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from scipy.stats import f_oneway, pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(description='Comprehensive audit of falling circles dataset')
    parser.add_argument('dataset_path', help='Path to the falling circles dataset')
    parser.add_argument('--output', help='Output directory for audit results (default: dataset/audits)')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"❌ Dataset path does not exist: {dataset_path}")
        sys.exit(1)
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = dataset_path / 'audits'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run comprehensive audit
    auditor = DatasetAuditor(dataset_path, output_dir, not args.no_plots)
    auditor.run_comprehensive_audit()

class DatasetAuditor:
    def __init__(self, dataset_path: Path, output_dir: Path, generate_plots: bool = True):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.generate_plots = generate_plots
        
        # Load dataset
        self.df = self._load_dataset()
        if self.df is None:
            print("❌ Failed to load dataset")
            sys.exit(1)
    
    def _load_dataset(self) -> pd.DataFrame:
        """Load dataset from samples.csv"""
        index_file = self.dataset_path / 'index' / 'samples.csv'
        if not index_file.exists():
            print(f"❌ samples.csv not found at {index_file}")
            return None
        
        try:
            df = pd.read_csv(index_file)
            print(f"✅ Loaded dataset with {len(df)} samples")
            return df
        except Exception as e:
            print(f"❌ Error loading samples.csv: {e}")
            return None
    
    def run_comprehensive_audit(self):
        """Run all audit analyses"""
        print("🔍 Running Comprehensive Dataset Audit")
        print("=" * 50)
        
        self.audit_basic_stats()
        self.audit_parameter_independence()
        self.audit_jam_type_distribution()
        self.audit_parameter_distributions()
        
        if self.generate_plots:
            self.create_independence_heatmap()
            self.create_distribution_plots()
        
        self.audit_quality_metrics()
        self.create_label_sensitivity_report()
        
        print(f"\n✅ Comprehensive audit complete!")
        print(f"📁 Results saved in: {self.output_dir}")
        self._print_summary()
    
    def audit_basic_stats(self):
        """Generate basic dataset statistics"""
        print("📊 Analyzing basic statistics...")
        
        stats = {
            'audit_timestamp': datetime.now().isoformat(),
            'dataset_path': str(self.dataset_path),
            'total_samples': len(self.df),
            'jam_type_counts': self.df['jam_type'].value_counts().to_dict(),
            'jam_type_percentages': (self.df['jam_type'].value_counts() / len(self.df) * 100).to_dict(),
            'exit_ratio_stats': {
                'mean': float(self.df['exit_ratio'].mean()),
                'std': float(self.df['exit_ratio'].std()),
                'min': float(self.df['exit_ratio'].min()),
                'max': float(self.df['exit_ratio'].max()),
                'median': float(self.df['exit_ratio'].median())
            },
            'parameter_stats': {}
        }
        
        # Parameter statistics
        param_columns = self._get_numeric_params()
        for col in param_columns:
            stats['parameter_stats'][col] = {
                'mean': float(self.df[col].mean()),
                'std': float(self.df[col].std()),
                'min': float(self.df[col].min()),
                'max': float(self.df[col].max()),
                'median': float(self.df[col].median())
            }
        
        # Save basic stats
        with open(self.output_dir / 'basic_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
    
    def audit_parameter_independence(self):
        """Audit parameter independence using correlation analysis"""
        print("🔗 Analyzing parameter independence...")
        
        param_columns = self._get_numeric_params()
        if len(param_columns) < 2:
            print("⚠️  Not enough numeric parameters for correlation analysis")
            return
        
        param_data = self.df[param_columns]
        
        # Calculate correlations
        pearson_corr = param_data.corr(method='pearson')
        spearman_corr = param_data.corr(method='spearman')
        
        # Find high correlations
        high_corr_threshold = 0.3
        high_correlations = []
        
        for i, param1 in enumerate(param_columns):
            for j, param2 in enumerate(param_columns):
                if i < j:
                    pearson_val = abs(pearson_corr.loc[param1, param2])
                    spearman_val = abs(spearman_corr.loc[param1, param2])
                    
                    if pearson_val > high_corr_threshold or spearman_val > high_corr_threshold:
                        high_correlations.append({
                            'param1': param1,
                            'param2': param2,
                            'pearson_correlation': float(pearson_val),
                            'spearman_correlation': float(spearman_val)
                        })
        
        # Calculate independence score
        all_correlations = []
        for i, param1 in enumerate(param_columns):
            for j, param2 in enumerate(param_columns):
                if i < j:
                    all_correlations.append(abs(pearson_corr.loc[param1, param2]))
        
        avg_correlation = np.mean(all_correlations) if all_correlations else 0
        independence_score = max(0.0, 1.0 - avg_correlation * 2)
        
        # Assessment
        if independence_score > 0.8:
            assessment = "Excellent - Parameters are highly independent"
        elif independence_score > 0.6:
            assessment = "Good - Parameters show low correlation"
        elif independence_score > 0.4:
            assessment = "Moderate - Some parameter dependencies detected"
        else:
            assessment = "Poor - Significant parameter dependencies detected"
        
        results = {
            'parameter_correlations': {
                'pearson': pearson_corr.to_dict(),
                'spearman': spearman_corr.to_dict()
            },
            'high_correlations': high_correlations,
            'independence_score': float(independence_score),
            'independence_assessment': assessment,
            'avg_absolute_correlation': float(avg_correlation)
        }
        
        # Save results
        with open(self.output_dir / 'parameter_independence.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create CSV matrix
        independence_matrix_data = []
        for i, param1 in enumerate(param_columns):
            for j, param2 in enumerate(param_columns):
                if i <= j:
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
        import csv
        with open(self.output_dir / 'independence_matrix.csv', 'w', newline='') as f:
            if independence_matrix_data:
                writer = csv.DictWriter(f, fieldnames=independence_matrix_data[0].keys())
                writer.writeheader()
                writer.writerows(independence_matrix_data)
        
        print(f"   Independence Score: {independence_score:.3f} - {assessment}")
    
    def audit_jam_type_distribution(self):
        """Analyze jam type distribution and exit ratio patterns"""
        print("🎯 Analyzing jam type distribution...")
        
        jam_analysis = {
            'jam_type_distribution': self.df['jam_type'].value_counts().to_dict(),
            'exit_ratio_by_jam_type': {},
            'jam_type_thresholds_validation': {},
            'natural_distribution_score': 0.0
        }
        
        # Exit ratio statistics by jam type
        for jam_type in self.df['jam_type'].unique():
            jam_data = self.df[self.df['jam_type'] == jam_type]['exit_ratio']
            jam_analysis['exit_ratio_by_jam_type'][jam_type] = {
                'count': len(jam_data),
                'mean': float(jam_data.mean()),
                'std': float(jam_data.std()),
                'min': float(jam_data.min()),
                'max': float(jam_data.max()),
                'median': float(jam_data.median())
            }
        
        # Validate thresholds
        thresholds = {
            'no_jam': (0.9, 1.0),
            'partial_jam': (0.3, 0.9),
            'full_jam': (0.0, 0.3)
        }
        
        for jam_type, (min_thresh, max_thresh) in thresholds.items():
            jam_data = self.df[self.df['jam_type'] == jam_type]['exit_ratio']
            if len(jam_data) > 0:
                within_range = ((jam_data >= min_thresh) & (jam_data < max_thresh)).sum()
                accuracy = within_range / len(jam_data)
                jam_analysis['jam_type_thresholds_validation'][jam_type] = {
                    'total_samples': len(jam_data),
                    'within_threshold': int(within_range),
                    'accuracy': float(accuracy),
                    'expected_range': f"{min_thresh}-{max_thresh}"
                }
        
        # Calculate natural distribution score
        counts = self.df['jam_type'].value_counts()
        proportions = counts / len(self.df)
        
        if len(proportions) == 3:
            min_prop = proportions.min()
            max_prop = proportions.max()
            balance_score = 1.0 - (max_prop - min_prop)
            presence_score = 1.0 if min_prop > 0.05 else min_prop * 20
            natural_score = (balance_score + presence_score) / 2
            jam_analysis['natural_distribution_score'] = float(max(0.0, min(1.0, natural_score)))
        
        # Save analysis
        with open(self.output_dir / 'jam_type_analysis.json', 'w') as f:
            json.dump(jam_analysis, f, indent=2)
    
    def audit_parameter_distributions(self):
        """Analyze parameter distributions for uniformity and coverage"""
        print("📈 Analyzing parameter distributions...")
        
        param_columns = self._get_numeric_params()
        distribution_analysis = {
            'parameter_distributions': {},
            'uniformity_scores': {},
            'coverage_analysis': {}
        }
        
        for param in param_columns:
            data = self.df[param]
            
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
            
            # Uniformity analysis
            hist, _ = np.histogram(data, bins=10)
            expected_uniform = len(data) / 10
            chi_sq = np.sum((hist - expected_uniform) ** 2 / expected_uniform)
            uniformity_score = max(0.0, 1.0 - chi_sq / (len(data) * 2))
            
            distribution_analysis['uniformity_scores'][param] = {
                'score': float(uniformity_score),
                'interpretation': 'Highly uniform' if uniformity_score > 0.8 else
                               'Moderately uniform' if uniformity_score > 0.6 else
                               'Non-uniform distribution'
            }
            
            # Coverage analysis
            param_range = data.max() - data.min()
            if param_range > 0:
                coverage_bins = 20
                hist, _ = np.histogram(data, bins=coverage_bins)
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
        
        # Save analysis
        with open(self.output_dir / 'parameter_distributions.json', 'w') as f:
            json.dump(distribution_analysis, f, indent=2)
    
    def create_independence_heatmap(self):
        """Create comprehensive independence heatmap visualization"""
        print("🔥 Creating independence heatmap...")
        
        param_columns = self._get_numeric_params()
        if len(param_columns) < 2:
            print("⚠️  Not enough parameters for heatmap")
            return
        
        param_data = self.df[param_columns]
        
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Dataset Parameter Independence Analysis', fontsize=16, fontweight='bold')
        
        # 1. Pearson correlation heatmap
        pearson_corr = param_data.corr(method='pearson')
        sns.heatmap(pearson_corr, annot=True, cmap='RdBu_r', center=0, 
                   square=True, fmt='.2f', cbar_kws={'label': 'Pearson Correlation'},
                   ax=ax1)
        ax1.set_title('Pearson Correlations\\n(Linear Relationships)', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Spearman correlation heatmap
        spearman_corr = param_data.corr(method='spearman')
        sns.heatmap(spearman_corr, annot=True, cmap='RdBu_r', center=0,
                   square=True, fmt='.2f', cbar_kws={'label': 'Spearman Correlation'},
                   ax=ax2)
        ax2.set_title('Spearman Correlations\\n(Rank-based)', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Parameter distributions
        key_params = param_columns[:min(4, len(param_columns))]
        for param in key_params:
            ax3.hist(param_data[param], alpha=0.6, label=param, bins=20)
        ax3.set_title('Parameter Distributions', fontweight='bold')
        ax3.legend()
        
        # 4. Summary
        ax4.axis('off')
        
        # Calculate metrics
        all_correlations = []
        high_corr_pairs = []
        
        for i, param1 in enumerate(param_columns):
            for j, param2 in enumerate(param_columns):
                if i < j:
                    pearson_val = abs(pearson_corr.loc[param1, param2])
                    all_correlations.append(pearson_val)
                    
                    if pearson_val > 0.3:
                        high_corr_pairs.append(f"{param1} ↔ {param2}: {pearson_val:.2f}")
        
        avg_correlation = np.mean(all_correlations) if all_correlations else 0
        independence_score = max(0.0, 1.0 - avg_correlation * 2)
        
        summary_text = f"""Independence Analysis Summary

📊 Dataset Size: {len(self.df):,} samples
🔢 Parameters: {len(param_columns)}
📈 Avg Correlation: {avg_correlation:.3f}
🎯 Independence Score: {independence_score:.3f}/1.0

Assessment: {
    'Excellent' if independence_score > 0.8 else
    'Good' if independence_score > 0.6 else
    'Moderate' if independence_score > 0.4 else 'Poor'
}

High Correlations (>0.3):
{chr(10).join(high_corr_pairs[:5]) if high_corr_pairs else 'None detected'}
{"..." if len(high_corr_pairs) > 5 else ""}"""
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'independence_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_distribution_plots(self):
        """Create parameter distribution plots"""
        print("📊 Creating distribution plots...")
        
        param_columns = self._get_numeric_params()
        if not param_columns:
            return
        
        # Create distribution plots
        n_params = len(param_columns)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Parameter Distribution Analysis', fontsize=16, fontweight='bold')
        
        for i, param in enumerate(param_columns):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Histogram with KDE
            data = self.df[param]
            ax.hist(data, bins=20, alpha=0.7, density=True, label='Histogram')
            
            # Add KDE
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(data)
                x_range = np.linspace(data.min(), data.max(), 100)
                ax.plot(x_range, kde(x_range), 'r-', label='KDE')
            except:
                pass
            
            ax.set_title(f'{param}\\nMean: {data.mean():.3f}, Std: {data.std():.3f}')
            ax.set_xlabel(param)
            ax.set_ylabel('Density')
            ax.legend()
        
        # Hide empty subplots
        for i in range(n_params, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            if n_rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'parameter_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def audit_quality_metrics(self):
        """Generate overall dataset quality metrics"""
        print("⭐ Calculating quality metrics...")
        
        scores = {}
        recommendations = []
        
        # Sample size adequacy
        sample_count = len(self.df)
        if sample_count >= 1000:
            scores['sample_size'] = 1.0
        elif sample_count >= 100:
            scores['sample_size'] = 0.8
        else:
            scores['sample_size'] = 0.6
            recommendations.append("Consider increasing sample size for better statistical power")
        
        # Jam type balance
        jam_counts = self.df['jam_type'].value_counts()
        if len(jam_counts) > 1:
            balance_ratio = jam_counts.min() / jam_counts.max()
            scores['jam_type_balance'] = balance_ratio
            if balance_ratio < 0.2:
                recommendations.append("Improve jam type balance - some types underrepresented")
        else:
            scores['jam_type_balance'] = 0.0
            recommendations.append("Only one jam type present - increase diversity")
        
        # Parameter independence
        try:
            with open(self.output_dir / 'parameter_independence.json', 'r') as f:
                independence_data = json.load(f)
                scores['parameter_independence'] = independence_data.get('independence_score', 0.5)
        except:
            scores['parameter_independence'] = 0.5
        
        # Exit ratio validity
        valid_ratios = ((self.df['exit_ratio'] >= 0) & (self.df['exit_ratio'] <= 1)).sum()
        scores['exit_ratio_validity'] = valid_ratios / len(self.df)
        
        # Overall quality score
        weights = {
            'sample_size': 0.25,
            'jam_type_balance': 0.25,
            'parameter_independence': 0.3,
            'exit_ratio_validity': 0.2
        }
        
        overall_score = sum(scores[metric] * weights[metric] for metric in scores.keys())
        
        # Assessment
        if overall_score >= 0.85:
            assessment = "Excellent - High quality dataset"
        elif overall_score >= 0.7:
            assessment = "Good - Dataset meets quality standards"
        elif overall_score >= 0.5:
            assessment = "Acceptable - Some quality issues"
        else:
            assessment = "Poor - Significant quality issues"
            recommendations.append("Consider regenerating dataset with improved parameters")
        
        quality_metrics = {
            'dataset_quality_score': float(overall_score),
            'quality_breakdown': {k: float(v) for k, v in scores.items()},
            'recommendations': recommendations,
            'overall_assessment': assessment,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save quality metrics
        with open(self.output_dir / 'quality_metrics.json', 'w') as f:
            json.dump(quality_metrics, f, indent=2)
        
        print(f"   Overall Quality Score: {overall_score:.3f} - {assessment}")
    
    def create_label_sensitivity_report(self):
        """Create comprehensive label sensitivity analysis report"""
        print("📝 Creating label sensitivity report...")
        
        report_lines = []
        report_lines.append("Dataset Label Sensitivity Analysis")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated: {datetime.now().isoformat()}")
        report_lines.append(f"Dataset: {self.dataset_path}")
        report_lines.append(f"Total samples: {len(self.df)}")
        report_lines.append("")
        
        # Jam type distribution
        report_lines.append("JAM TYPE DISTRIBUTION:")
        jam_counts = self.df['jam_type'].value_counts()
        for jam_type, count in jam_counts.items():
            percentage = (count / len(self.df)) * 100
            report_lines.append(f"  {jam_type}: {count} samples ({percentage:.1f}%)")
        report_lines.append("")
        
        # Exit ratio validation
        report_lines.append("EXIT RATIO THRESHOLD VALIDATION:")
        thresholds = {
            'no_jam': (0.9, 1.0),
            'partial_jam': (0.3, 0.9),
            'full_jam': (0.0, 0.3)
        }
        
        for jam_type, (min_thresh, max_thresh) in thresholds.items():
            jam_data = self.df[self.df['jam_type'] == jam_type]['exit_ratio']
            if len(jam_data) > 0:
                within_range = ((jam_data >= min_thresh) & (jam_data < max_thresh)).sum()
                accuracy = within_range / len(jam_data) * 100
                report_lines.append(f"  {jam_type} ({min_thresh}-{max_thresh}):")
                report_lines.append(f"    Accuracy: {accuracy:.1f}% ({within_range}/{len(jam_data)})")
                report_lines.append(f"    Range: {jam_data.min():.3f} - {jam_data.max():.3f}")
                report_lines.append(f"    Mean: {jam_data.mean():.3f} ± {jam_data.std():.3f}")
        
        report_lines.append("")
        
        # Parameter sensitivity
        report_lines.append("PARAMETER SENSITIVITY TO JAM TYPE:")
        param_columns = self._get_numeric_params()
        
        for param in param_columns:
            report_lines.append(f"\\n  {param.upper()}:")
            
            for jam_type in self.df['jam_type'].unique():
                jam_data = self.df[self.df['jam_type'] == jam_type][param]
                if len(jam_data) > 0:
                    report_lines.append(f"    {jam_type}: {jam_data.mean():.3f} ± {jam_data.std():.3f}")
            
            # ANOVA test
            try:
                groups = [self.df[self.df['jam_type'] == jt][param].values 
                         for jt in self.df['jam_type'].unique()]
                f_stat, p_value = f_oneway(*groups)
                sensitivity = "High" if p_value < 0.01 else "Medium" if p_value < 0.05 else "Low"
                report_lines.append(f"    Sensitivity: {sensitivity} (F={f_stat:.2f}, p={p_value:.4f})")
            except:
                report_lines.append(f"    Sensitivity: Unable to calculate")
        
        # Classification stability
        report_lines.append("")
        report_lines.append("CLASSIFICATION STABILITY:")
        boundary_tolerance = 0.05
        near_boundaries = 0
        
        for _, row in self.df.iterrows():
            exit_ratio = row['exit_ratio']
            if (abs(exit_ratio - 0.3) < boundary_tolerance or 
                abs(exit_ratio - 0.9) < boundary_tolerance):
                near_boundaries += 1
        
        boundary_percentage = (near_boundaries / len(self.df)) * 100
        stability_score = 1.0 - (boundary_percentage / 100)
        
        report_lines.append(f"  Samples near boundaries (±5%): {near_boundaries} ({boundary_percentage:.1f}%)")
        report_lines.append(f"  Stability score: {stability_score:.3f}")
        
        # Save report
        report_path = self.output_dir / 'label_sensitivity_report.txt'
        with open(report_path, 'w') as f:
            f.write('\\n'.join(report_lines))
    
    def _get_numeric_params(self):
        """Get list of numeric parameter columns"""
        return [col for col in self.df.columns 
                if col not in ['sample_id', 'sample_type', 'jam_type'] 
                and self.df[col].dtype in ['int64', 'float64']]
    
    def _print_summary(self):
        """Print audit summary"""
        try:
            with open(self.output_dir / 'quality_metrics.json', 'r') as f:
                quality = json.load(f)
            
            with open(self.output_dir / 'parameter_independence.json', 'r') as f:
                independence = json.load(f)
            
            print("\\n" + "=" * 50)
            print("AUDIT SUMMARY")
            print("=" * 50)
            print(f"📊 Dataset Size: {len(self.df):,} samples")
            print(f"⭐ Quality Score: {quality['dataset_quality_score']:.3f}/1.0")
            print(f"🔗 Independence Score: {independence['independence_score']:.3f}/1.0")
            print(f"🎯 Assessment: {quality['overall_assessment']}")
            
            if quality['recommendations']:
                print("\\n📋 Recommendations:")
                for rec in quality['recommendations']:
                    print(f"  • {rec}")
                    
        except Exception as e:
            print(f"⚠️  Could not generate summary: {e}")

if __name__ == "__main__":
    main()