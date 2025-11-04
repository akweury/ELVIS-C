# ELVIS-C Core Source Files

This folder contains the essential files needed for the falling circles physics simulation and dataset generation project.

## Core Files

### 🎯 **falling_circles_env.py**
- **Purpose**: Core physics simulation engine
- **Key Classes**: 
  - `VideoParams`: Configuration parameters for videos
  - `Circle`: Individual circle physics object
  - `FallingCirclesEnvironment`: Main simulation engine
- **Usage**: 
  ```python
  from falling_circles_env import FallingCirclesEnvironment, VideoParams
  params = VideoParams(hole_diameter=40, wind_strength=2.0, num_circles=8)
  env = FallingCirclesEnvironment(params)
  frames, metadata = env.generate_video(seed=42)
  ```

### � Comprehensive Dataset Auditing
- **Parameter Independence Analysis**: Correlation matrices, independence scores, and heatmap visualizations
- **Jam Type Distribution**: Natural distribution analysis and classification accuracy validation  
- **Quality Metrics**: Overall dataset quality scoring with detailed breakdowns and recommendations
- **Label Sensitivity**: Analysis of how sensitive jam type labels are to parameter changes
- **Visual Analytics**: Independence heatmaps, parameter distribution plots, and correlation analysis

### 📊 Comprehensive Metadata
- **Purpose**: Large-scale dataset generation with parallel processing
- **Features**:
  - Independent parameter sampling
  - Automatic jam type classification
  - Parallel video generation
  - Comprehensive metadata collection
  - Train/validation/test splits
- **Usage**:
  ```bash
  # Use 8 parallel workers for faster generation
python generate_falling_circles_dataset.py --num_videos 1000 --workers 8
```

### Dataset Quality Auditing
```bash
# Comprehensive dataset quality audit with independence analysis
python audit_dataset.py path/to/dataset

# Custom audit output directory  
python audit_dataset.py path/to/dataset --output custom_audit_dir

# Skip plot generation (faster)
python audit_dataset.py path/to/dataset --no-plots
  ```

### 🤖 **ai_model_interface.py**
- **Purpose**: Simple interface for AI models to generate intervention videos
- **Key Functions**:
  - `generate_intervention_video()`: One-line intervention generation
  - `AIVideoInterface`: Advanced intervention capabilities
- **Usage**:
  ```python
  from ai_model_interface import generate_intervention_video
  result = generate_intervention_video(
      baseline_parameters=params,
      intervention_target="hole_diameter",
      intervention_value=70
  )
  ```

### Testing
- **`test_dataset_structure.py`** - Validation script to verify generated datasets match falling_circles_v1 format
- **`audit_dataset.py`** - Comprehensive dataset quality analysis and independence verification

## Quick Start

1. **Generate a single video**:
   ```python
   from falling_circles_env import FallingCirclesEnvironment, VideoParams
   params = VideoParams(num_frames=40, hole_diameter=35)
   env = FallingCirclesEnvironment(params)
   frames, metadata = env.generate_video()
   ```

2. **Generate dataset**:
   ```bash
   python generate_falling_circles_dataset.py --num_videos 100 --output my_dataset
   ```

3. **AI intervention**:
   ```python
   from ai_model_interface import generate_intervention_video
   result = generate_intervention_video(
       baseline_parameters={"hole_diameter": 30, "num_circles": 8},
       intervention_target="hole_diameter",
       intervention_value=60
   )
   ```

## Dependencies

- `numpy`: Numerical computations
- `PIL`: Image processing
- `pathlib`: File system operations
- `tqdm`: Progress bars
- `concurrent.futures`: Parallel processing

## Features

- ✅ **40-frame videos**: Optimized for fast completion
- ✅ **Independent parameter sampling**: Natural jam type distribution
- ✅ **Automatic jam classification**: Based on exit ratios
- ✅ **Parallel processing**: Multi-worker dataset generation
- ✅ **AI-friendly interface**: Simple intervention capabilities
- ✅ **Labeled GIFs**: Visual feedback with overlaid information
- ✅ **Comprehensive metadata**: Full simulation tracking

## Comprehensive Dataset Auditing

The enhanced audit system provides deep analysis of dataset quality and parameter independence:

### 📊 **Audit Output Files**
- `basic_statistics.json` - Dataset overview and parameter statistics
- `parameter_independence.json` - Correlation analysis and independence scoring  
- `jam_type_analysis.json` - Jam type distribution and threshold validation
- `quality_metrics.json` - Overall quality score and recommendations
- `label_sensitivity_report.txt` - Detailed sensitivity analysis report

### 📈 **Visual Analytics**
- `independence_heatmap.png` - Comprehensive parameter correlation heatmap
- `parameter_distributions.png` - Parameter distribution analysis plots
- `independence_matrix.csv` - Detailed correlation matrix data

### 🎯 **Key Quality Metrics**
- **Independence Score**: 0.0-1.0 scale measuring parameter independence
- **Quality Score**: 0.0-1.0 overall dataset quality assessment
- **Classification Stability**: How stable jam type classifications are near boundaries
- **Parameter Coverage**: How well parameters cover their design space
- **Jam Type Balance**: Distribution balance across different jam scenarios

### 🔍 **Independence Analysis**
The audit system specifically examines parameter independence to ensure:
- ✅ Parameters are sampled independently without bias
- ✅ No unexpected correlations between physics parameters  
- ✅ Natural emergence of jam types based on physics outcomes
- ✅ Robust classification boundaries and label stability
- ✅ Comprehensive coverage of parameter space

This analysis validates that the dataset generation produces truly independent parameter sampling, leading to natural and unbiased jam type distributions based purely on physics simulation outcomes.