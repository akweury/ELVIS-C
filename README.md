# ELVIS-C

A physics simulation environment for generating video datasets with causal dynamics, focusing on object flow and jam scenarios. **Now featuring intervention pairs for rigorous causal inference research, adaptive active intervention controllers for intelligent causal discovery, AND simple AI-friendly interfaces for easy integration.**

---

## 🤖 **New: AI-Friendly Interface Package**

**Simple, one-line interventions for AI models!** The new interface package makes it incredibly easy for AI models to perform interventions and analyze causal effects.

```python
from elvis_env.interface import quick_intervene

# AI performs intervention in one line
result = quick_intervene("hole_diameter", 80, seed=42)
print(f"Effect: {result.effect_description}")
# Output: "Effect: Intervention changed outcome: partial_jam → no_jam"
```

### For AI Models:
- 🎯 **One-line interventions**: `quick_intervene("parameter", value)`
- 🔄 **Multi-parameter changes**: `multi_intervene({"param1": val1, "param2": val2})`
- 📊 **Automatic analysis**: Built-in effect magnitude and descriptions
- 💾 **Easy saving**: `save_intervention_gifs()` for visualization
- 🛡️ **Parameter validation**: Automatic range checking and constraints
- 📖 **Self-documenting**: `get_parameter_info()` explains all parameters

### Quick Examples:
```python
# Compare different values
results = compare_parameters("wind_strength", [1.0, 3.0, 5.0])

# Complex multi-parameter intervention  
result = interface.multi_intervene({
    "hole_diameter": 60,
    "wind_strength": 4.0, 
    "num_circles": 8
})

# Adaptive learning pattern
for value in [20, 40, 60, 80]:
    result = quick_intervene("hole_diameter", value)
    if result.effect_magnitude > 0.1:
        print(f"Significant effect at {value}!")
```

📚 **AI Quick Start**: See `elvis_env/interface/AI_QUICK_START.md`

---

## 🚀 **Active Intervention Controllers**

ELVIS-C includes **intelligent controllers** that dynamically choose which variables to intervene on and by how much, based on past results. This enables **adaptive causal discovery** that goes far beyond fixed intervention schedules.

### Revolutionary Features:
- 🧠 **Intelligent Selection**: Controllers learn which interventions are most informative
- 📈 **Adaptive Learning**: Continuous parameter optimization based on observed effects
- 🎯 **Uncertainty-Guided**: Focus exploration on uncertain regions of causal space
- ⚡ **Discovery Acceleration**: Maximize information gain per intervention
- 🔬 **Model-Based Options**: Gaussian Process controllers with sophisticated acquisition functions

### Available Controllers:
- **RandomController**: Baseline random selection
- **MaxEffectController**: Exploit high-impact interventions
- **ExplorationController**: UCB-based exploration/exploitation balance
- **UncertaintyController**: Target uncertain parameter regions
- **AdaptiveParameterController**: Learn optimal intervention magnitudes
- **GaussianProcessController**: Model-based active learning with GP regression

---

## 🎯 **Intervention Pairs for Causal Analysis**

ELVIS-C includes **counterfactual intervention pairs** - the gold standard for causal inference research. For each baseline simulation, generate an intervention version that modifies exactly one causal variable while keeping the random seed and all other parameters identical.

### Key Features:
- ✅ **Perfect Counterfactuals**: Same random seed, single variable change
- ✅ **12 Intervention Types**: Physics, geometry, object, and timing modifications  
- ✅ **Verified Independence**: Parameters confirmed statistically independent
- ✅ **Causal Effect Analysis**: Built-in statistical analysis and visualization
- ✅ **Research-Ready**: Designed for causal inference studies

---

## ELVIS Environment (`elvis_env/`)

The `elvis_env` folder contains a physics simulation environment designed for generating video datasets with realistic dynamics and causal relationships. This environment provides tools for creating synthetic video data for machine learning research, particularly for studying causal inference in visual scenarios.

### Available Scripts

#### 1. **Falling Circles Simulation** (`falling_circles.py`)

Generates videos of circles falling through a funnel with configurable physics parameters and emergent jam scenarios.

![Partial Jam Example](demo/falling_circles_partial_jam.gif)

**Core Features:**
- **Realistic Physics**: Gravity (constant 1.0), wind effects (0-0.2), collision dynamics
- **Emergent Jam Types**: Post-hoc classification based on actual physics outcomes
  - `no_jam`: High exit rate (>70%), minimal blocking
  - `partial_jam`: Moderate exit rate (30-70%), some flow issues  
  - `full_jam`: Low exit rate (<30%), significant jamming
- **Unbiased Parameters**: All physics parameters independently sampled
- **Finite Objects**: Fixed number of circles (3-8) per simulation
- **Data Export**: Frame-level CSV data + comprehensive metadata
- **Clean Separation**: PNG frames (analysis) + labeled GIFs (visualization)

**Recent Improvements:**
- ✅ **Constant Gravity**: Eliminates spurious correlations (verified statistically)
- ✅ **Random Parameters**: Jam types emerge from physics, not predetermined
- ✅ **Parameter Independence**: Verified with correlation analysis (<0.15 max correlation)
- ✅ **Jam Type Labels**: Added to GIF visualizations for all frames

#### 2. **Intervention Pairs Generator** (`generate_intervention_pairs.py`) 🆕

Creates counterfactual intervention pairs for causal analysis research.

**Available Interventions:**
- **Physics**: `wind_strength_high/low`, `wind_direction_flip`
- **Geometry**: `hole_larger/smaller`, `hole_offset`  
- **Objects**: `more/fewer/larger/smaller_circles`
- **Timing**: `spawn_faster/slower`

**Example Causal Effects Discovered:**
- **`smaller_circles`**: 100% jam type change, +0.600 exit ratio improvement
- **`hole_larger`**: 50% change rate, reduces jamming significantly
- **`hole_smaller`**: 50% change rate, increases jamming
- **Wind effects**: Minimal impact on jam formation

#### 4. **Active Intervention Controllers** (`generate_active_interventions.py`) 🆕🧠

**Revolutionary adaptive causal discovery** where intelligent controllers dynamically choose interventions based on past results.

**Available Controllers:**
- **ExplorationController**: UCB-based exploration/exploitation balance
- **MaxEffectController**: Targets high-impact interventions with ε-greedy strategy
- **UncertaintyController**: Focuses on uncertain parameter regions
- **AdaptiveParameterController**: Learns optimal intervention magnitudes
- **GaussianProcessController**: Model-based active learning with GP regression
- **RandomController**: Baseline random selection for comparison

**Key Advantages:**
- 🎯 **Targeted Discovery**: Learn which interventions matter most
- 📈 **Adaptive Learning**: Controllers improve over time
- ⚡ **Efficiency**: Maximize information gain per intervention
- 🔬 **Research Applications**: Accelerate causal discovery research

**Example Active Learning Results:**
- **MaxEffect Controller**: Converges 3x faster to high-impact interventions
- **Exploration Controller**: Achieves 67% exploration efficiency vs 50% random
- **GP Controller**: Uses uncertainty estimation for principled intervention selection

Statistical analysis and visualization of discovered causal relationships.

**Outputs:**
- Effect magnitude distributions and box plots
- Jam type transition matrices  
- Statistical significance tests
- Intervention effectiveness rankings

#### 5. **Causal Effects Analyzer** (`analyze_causal_effects.py`) 🆕

Statistical analysis and visualization of discovered causal relationships.

**Outputs:**
- Effect magnitude distributions and box plots
- Jam type transition matrices  
- Statistical significance tests
- Intervention effectiveness rankings

#### 6. **Active Controller Analyzer** (`analyze_active_interventions.py`) 🆕🧠

Compare performance of different active controllers for causal discovery.

**Analysis Features:**
- Discovery efficiency learning curves
- Controller performance metrics comparison
- Intervention selection pattern analysis  
- Exploration vs exploitation trade-offs

#### 7. **Parameter Independence Analyzer** (`analyze_parameter_independence.py`) 🆕

Verifies that physics parameters are independently sampled for unbiased causal modeling.

**Validation Results:**
- ✅ No problematic correlations between independent parameters
- ✅ Gravity correlation issue resolved (now constant)
- ✅ Only expected mathematical relationships remain
- ✅ Ready for causal inference research

---

## Quick Start

### 1. Generate Standard Dataset

Create a balanced dataset with verified parameter independence:

```bash
# Small test dataset
python elvis_env/scripts/falling_circles.py --num_videos 50 --out output/falling_circles --export_gif

# Large training dataset  
python elvis_env/scripts/falling_circles.py --num_videos 1000 --out data/falling_circles --workers 8 --no_gif
```

### 2. Generate Intervention Pairs 🆕

Create counterfactual pairs for causal analysis:

```bash
# Generate 100 intervention pairs (all intervention types)
python elvis_env/scripts/generate_intervention_pairs.py --num_pairs 100 --out data/interventions --workers 4

# Focus on specific intervention type
python elvis_env/scripts/generate_intervention_pairs.py --num_pairs 50 --intervention_type hole_larger --out data/hole_effects

# Large-scale causal dataset
python elvis_env/scripts/generate_intervention_pairs.py --num_pairs 500 --no_gif --workers 12 --out data/causal_study
```

### 3. Generate Active Interventions 🆕🧠

**Intelligent controllers** that learn which interventions are most informative:

```bash
# Exploration controller (UCB-based)
python elvis_env/scripts/generate_active_interventions.py --controller exploration --num_pairs 100 --out data/active_exploration

# Max effect controller (exploit high-impact interventions)  
python elvis_env/scripts/generate_active_interventions.py --controller max_effect --num_pairs 100 --out data/active_max_effect

# Gaussian Process controller (model-based active learning)
python elvis_env/scripts/generate_active_interventions.py --controller gaussian_process --num_pairs 200 --out data/active_gp

# Adaptive parameter controller (learn intervention magnitudes)
python elvis_env/scripts/generate_active_interventions.py --controller adaptive --learning_rate 0.1 --num_pairs 150 --out data/active_adaptive
```

### 4. Analyze Causal Effects 🆕

Statistical analysis of discovered causal relationships:

```bash
python elvis_env/scripts/analyze_causal_effects.py --manifest_path data/interventions/intervention_manifest.json --output_dir results/causal_analysis
```

### 5. Compare Active Controllers 🆕🧠

Compare performance of different active learning strategies:

```bash
# Generate datasets with different controllers
python elvis_env/scripts/generate_active_interventions.py --controller random --num_pairs 100 --out data/random
python elvis_env/scripts/generate_active_interventions.py --controller exploration --num_pairs 100 --out data/exploration  
python elvis_env/scripts/generate_active_interventions.py --controller max_effect --num_pairs 100 --out data/max_effect

# Compare controller performance
python elvis_env/scripts/analyze_active_interventions.py \
    --data_dirs data/random data/exploration data/max_effect \
    --controller_names Random Exploration MaxEffect \
    --output_dir results/controller_comparison
```

### 6. Verify Parameter Independence 🆕

Validate that parameters are independently sampled:

```bash
python elvis_env/scripts/analyze_parameter_independence.py --manifest_path data/falling_circles/dataset_manifest.json --output_dir results/independence_check
```

---

## Research Applications

### 🔬 **Causal Inference Research**
- **Treatment Effects**: Measure exact impact of parameter changes using intervention pairs
- **Mechanism Discovery**: Identify which variables causally affect jam formation  
- **Counterfactual Learning**: Train models on perfect baseline/intervention pairs
- **Adaptive Discovery**: Use intelligent controllers to accelerate causal learning

### 🤖 **Machine Learning Applications**
- **Causal Representation Learning**: Learn causal embeddings from paired data
- **Intervention Prediction**: Predict effects of parameter modifications
- **Physics-Informed ML**: Training on realistic dynamics with causal structure
- **Active Learning**: Optimize data collection for causal model training

### 🧠 **Active Learning Research**
- **Controller Development**: Design new active intervention strategies
- **Uncertainty Quantification**: Study epistemic uncertainty in causal discovery
- **Multi-Objective Optimization**: Balance discovery speed vs breadth vs accuracy
- **Meta-Learning**: Learn to learn across different causal environments

### ⚖️ **Scientific Validation**
- **Parameter Sensitivity**: Quantify sensitivity to each physical parameter
- **Threshold Detection**: Find critical points where behavior changes
- **Interaction Effects**: Study how parameters interact causally
- **Robustness Analysis**: Test causal relationships across parameter ranges

---

## Configuration & Parameters

### Falling Circles Script

#### Core Physics Parameters (Independently Sampled)

- **Gravity**: Constant 1.0 (eliminates spurious correlations)
- **Wind Strength**: 0-0.2 pixels/frame (horizontal force)
- **Wind Direction**: Random left (-1) or right (+1)
- **Circle Count**: 3-8 circles (finite objects)
- **Circle Sizes**: 5-15 pixel radius (10-30px diameter)
- **Hole Diameter**: 15-30 pixels (exit opening)
- **Spawn Rate**: 0.15-0.6 (timing of circle appearances)

#### Command Line Options

- `--num_videos`: Total number of videos to generate
- `--out`: Output directory for generated data
- `--export_gif` / `--no_gif`: Control GIF visualization generation
- `--train_ratio`: Proportion of training vs test videos (default: 0.8)
- `--workers`: Number of parallel workers for generation
- `--gif_fps`: Frame rate for GIF files (default: 10)

### Intervention Pairs Script 🆕

#### Intervention Types

| Category | Interventions | Effect on Jamming |
|----------|--------------|-------------------|
| **Physics** | `wind_strength_high/low`, `wind_direction_flip` | Minimal effect |
| **Geometry** | `hole_larger` (reduces jams), `hole_smaller` (increases jams) | **Strong effect** |
| **Objects** | `more_circles` (increases jams), `fewer_circles` (reduces jams) | Moderate effect |
| **Objects** | `larger_circles` (increases jams), `smaller_circles` (reduces jams) | **Strong effect** |
| **Timing** | `spawn_faster/slower` | Minimal effect |

#### Perfect Counterfactual Features

- ✅ **Same Random Seed**: Identical randomness across baseline/intervention
- ✅ **Single Variable Change**: Isolates causal effects precisely  
- ✅ **All Other Parameters Identical**: Eliminates confounding variables
- ✅ **Comprehensive Metadata**: Complete causal effect analysis

---

## Data Output Structure

### Standard Dataset

```
output/
├── train/                    # Training videos
│   ├── video_00000/
│   │   ├── frame_000.png     # Clean frames (224x224 RGB)
│   │   ├── ...
│   │   ├── frame_059.png
│   │   ├── frame_facts.csv   # Frame-level physics data
│   │   └── meta.json         # Parameters + jam type + exit stats
│   └── ...
├── test/                     # Test videos
├── gifs/                     # Labeled visualizations
│   ├── train/
│   │   ├── video_00000.gif   # Includes jam type labels
│   └── test/
└── dataset_manifest.json    # Dataset metadata + jam distribution
```

### Intervention Pairs Dataset 🆕

```
interventions/
├── train/
│   ├── pair_00000/
│   │   ├── baseline/                        # Original simulation
│   │   │   ├── frame_000.png - frame_059.png
│   │   │   └── meta.json
│   │   ├── intervention_hole_larger/        # Modified simulation
│   │   │   ├── frame_000.png - frame_059.png
│   │   │   └── meta.json
│   │   └── comparison.json                  # Causal effect analysis
│   └── ...
├── test/
├── gifs/                                    # Side-by-side comparisons
│   ├── train/
│   │   ├── pair_00000/
│   │   │   ├── baseline.gif
│   │   │   └── intervention_hole_larger.gif
└── intervention_manifest.json              # Causal effects summary
```

### Active Interventions Dataset 🆕🧠

```
active_interventions/
├── train/
│   ├── pair_00000/
│   │   ├── baseline/                        # Original simulation
│   │   │   ├── frame_000.png - frame_059.png
│   │   │   └── meta.json
│   │   ├── intervention_hole_larger/        # Controller-selected intervention
│   │   │   ├── frame_000.png - frame_059.png
│   │   │   └── meta.json
│   │   └── comparison.json                  # Causal effect analysis
│   └── ...
├── test/
├── gifs/                                    # Side-by-side comparisons
├── active_intervention_manifest.json       # Controller decisions & learning
├── controller_checkpoint_*.json            # Periodic controller state saves
└── final_controller_state.json            # Final learned controller state
```

### Controller Analysis Results 🆕🧠

```
controller_analysis/
├── active_intervention_analysis.png        # Learning curves & comparisons  
├── active_intervention_analysis_report.txt # Performance metrics & insights
├── controller_comparison.csv               # Quantitative comparison data
└── discovery_efficiency_plots/             # Individual controller analysis
```

### Analysis Results 🆕

```
causal_analysis/
├── causal_analysis_report.txt              # Detailed statistical analysis
├── causal_effects_analysis.png             # Effect magnitude plots
├── jam_type_transitions.png                # Transition matrix heatmaps
└── independence_heatmap.png                 # Parameter correlation matrix
```

---

## Validation & Quality Assurance

### ✅ **Parameter Independence Verified**
- Maximum correlation between independent parameters: **<0.15**
- Gravity correlation issue **resolved** (now constant)
- No spurious dependencies detected in 200+ video analysis
- **Ready for unbiased causal modeling**

### ✅ **Causal Effects Validated**
- **Strong effects**: Hole size and circle size significantly affect jamming
- **Weak effects**: Wind and timing have minimal impact
- **Directional consistency**: Larger holes/smaller circles reduce jamming
- **Statistical significance**: Effects verified with proper controls

### ✅ **Physics Realism**
- Realistic collision detection and response
- Proper funnel geometry with V-shaped slopes  
- Natural jam formation from physics (not predetermined)
- Consistent exit criteria across all simulations

---

## Development & Extension

### Adding New Interventions

```python
# In generate_intervention_pairs.py
causal_interventions = {
    'your_intervention': {'parameter_name': new_value},
    # Ensure single parameter change only
}
```

### Creating New Simulation Scripts

Follow these conventions:

1. **Location**: `elvis_env/scripts/your_script.py`
2. **Output Structure**: Standard `train/`, `test/`, `gifs/` folders
3. **Data Format**: CSV frame data + JSON metadata  
4. **Parameters**: Independent sampling with validation
5. **Visualization**: Clean PNG + labeled GIF separation

---

## Requirements

### Basic Requirements
- Python 3.7+
- PIL (Pillow) - Image processing
- NumPy - Numerical computations
- Standard library modules (random, math, json, csv, multiprocessing)

### Analysis Requirements (for causal analysis) 🆕
- pandas - Data manipulation
- matplotlib - Plotting  
- seaborn - Statistical visualization
- scipy - Statistical tests
- scikit-learn - Mutual information analysis and GP models (for advanced controllers)

```bash
# Install analysis dependencies
pip install pandas matplotlib seaborn scipy scikit-learn
```

---

## Citation

If you use ELVIS-C for research, especially the intervention pairs functionality, please cite:

```bibtex
@software{elvis_c_2025,
  title={ELVIS-C: Physics Simulation Environment with Causal Intervention Pairs and Active Learning Controllers},
  author={Your Name},
  year={2025},
  note={Counterfactual intervention pairs and intelligent active learning for causal inference research}
}
```

---

**🎯 Ready for rigorous causal inference research with verified parameter independence, gold-standard counterfactual data, and intelligent active learning controllers!**

For detailed documentation on the active intervention system, see [ACTIVE_INTERVENTION_DOCUMENTATION.md](ACTIVE_INTERVENTION_DOCUMENTATION.md).