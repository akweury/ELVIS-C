# Active Intervention System Documentation

## Overview

The Active Intervention System extends ELVIS-C's static intervention pairs with **intelligent controllers** that dynamically choose which variables to intervene on and by how much, based on accumulated results from previous interventions. This enables sophisticated **adaptive causal discovery** that goes far beyond fixed intervention schedules.

## Key Innovation: Adaptive Causal Discovery

Traditional causal discovery approaches use fixed experimental designs. Our active intervention system enables:

- **Dynamic Intervention Selection**: Controllers learn which interventions are most informative
- **Adaptive Parameter Tuning**: Continuous parameter modifications based on observed effects
- **Uncertainty-Guided Exploration**: Focus exploration on uncertain regions of the causal space
- **Efficient Causal Discovery**: Maximize information gain per intervention

## Controller Architecture

### Base Controller Interface

All controllers inherit from `ActiveController` and implement:

```python
@abstractmethod
def decide_intervention(self, baseline_params, available_interventions: List[str]) -> Tuple[str, Dict[str, Any]]:
    """Decide which intervention to perform and with what parameters"""
    pass

def update_from_results(self, result: InterventionResult):
    """Update controller state based on intervention results"""
    pass
```

### Intervention History Tracking

Controllers maintain comprehensive history with:
- **Effect Magnitudes**: Quantitative impact of each intervention
- **Success Rates**: Proportion of interventions causing jam type changes
- **Parameter Ranges**: Bounds of parameter space explored
- **Statistical Confidence**: Running statistics for decision-making

## Available Controllers

### 1. RandomController (Baseline)
- **Strategy**: Random intervention selection
- **Use Case**: Baseline comparison, unbiased exploration
- **Characteristics**: No learning, uniform exploration

```bash
python generate_active_interventions.py --controller random --num_pairs 100
```

### 2. MaxEffectController 
- **Strategy**: Exploit interventions with highest observed effects
- **Use Case**: Maximize immediate causal effect discovery
- **Characteristics**: ε-greedy exploration (default 20%)

```bash
python generate_active_interventions.py --controller max_effect --exploration_rate 0.2 --num_pairs 100
```

### 3. ExplorationController (UCB-based)
- **Strategy**: Upper Confidence Bound for balancing exploration/exploitation
- **Use Case**: Systematic causal space exploration with confidence bounds
- **Characteristics**: Principled uncertainty-guided selection

```bash
python generate_active_interventions.py --controller exploration --exploration_constant 1.4 --num_pairs 100
```

### 4. UncertaintyController
- **Strategy**: Target interventions with highest uncertainty (standard deviation)
- **Use Case**: Reduce uncertainty about causal effects
- **Characteristics**: Minimize epistemic uncertainty

```bash
python generate_active_interventions.py --controller uncertainty --min_samples 3 --num_pairs 100
```

### 5. AdaptiveParameterController
- **Strategy**: Learn optimal intervention magnitudes using gradient updates
- **Use Case**: Fine-tune intervention parameters for maximum effect
- **Characteristics**: Continuous parameter optimization

```bash
python generate_active_interventions.py --controller adaptive --learning_rate 0.1 --num_pairs 100
```

### 6. GaussianProcessController (Model-Based)
- **Strategy**: Learn causal effect functions with Gaussian Process regression
- **Use Case**: Model-based active learning with uncertainty estimation
- **Characteristics**: Sophisticated acquisition functions (UCB, EI, PI)

```bash
# Requires: pip install scikit-learn
python generate_active_interventions.py --controller gaussian_process --num_pairs 100
```

## Continuous Parameter Support

Controllers can specify exact parameter values for fine-grained control:

### Example: Adaptive Hole Size
```python
# Controller decides: hole_diameter = baseline + learned_delta
intervention_params = {
    'hole_diameter': min(50, baseline_params.hole_diameter + learned_delta)
}
```

### Supported Continuous Parameters:
- **hole_diameter**: Exit hole size (10-50 pixels)
- **wind_strength**: Wind force magnitude (0-0.4 px/frame)
- **circle_size_min/max**: Object size ranges (2-20 pixels)
- **spawn_rate**: Spawning frequency (0.05-1.0)

## Performance Analysis

### Controller Comparison Metrics

1. **Discovery Efficiency**: Cumulative mean effect over time
2. **High-Impact Discovery Rate**: Proportion discovering strong effects (>0.3)
3. **Intervention Diversity**: Breadth of causal space explored  
4. **Convergence Speed**: How quickly controller focuses on effective interventions
5. **Exploration Efficiency**: Balance between exploration and exploitation

### Example Performance Results

From our analysis of 3 controllers on 10-15 interventions each:

| Controller | Mean Effect | High Effect Rate | Diversity | Convergence | Exploration Efficiency |
|------------|-------------|------------------|-----------|-------------|----------------------|
| **Random** | 0.345 | 50.0% | 0.60 | 0.000 | 0.500 |
| **MaxEffect** | 0.165 | 26.7% | 0.53 | **0.333** | 0.500 |
| **Exploration** | 0.133 | 20.0% | **0.80** | 0.000 | **0.667** |

**Key Insights:**
- **Random**: Surprisingly effective due to lucky high-impact discoveries
- **MaxEffect**: Fastest convergence but lower diversity
- **Exploration**: Best exploration efficiency and diversity

## Research Applications

### 1. Adaptive Experimental Design
```python
# Design experiments that adapt based on results
controller = ExplorationController()
for trial in range(num_trials):
    intervention = controller.decide_intervention(baseline, available)
    result = run_experiment(baseline, intervention)
    controller.update_from_results(result)
```

### 2. Causal Discovery Acceleration
- **Focus on Informative Regions**: Target parameter ranges with high uncertainty
- **Avoid Redundant Experiments**: Skip interventions with well-known effects
- **Optimize Resource Allocation**: Maximize information gain per experiment

### 3. Active Learning for Causal Models
```python
# Train causal models with actively selected data
gp_controller = GaussianProcessController()
while not converged:
    # Select most informative intervention
    intervention = gp_controller.decide_intervention(baseline, available)
    result = collect_data(intervention)
    gp_controller.update_from_results(result)
    
    # Update causal model with new data
    causal_model.update(result)
```

### 4. Uncertainty Quantification
- **Effect Confidence Intervals**: Track uncertainty about causal effects
- **Parameter Sensitivity Analysis**: Identify which parameters matter most
- **Robustness Testing**: Test causal relationships across parameter ranges

## Usage Examples

### Basic Active Intervention Generation

```bash
# Generate 50 pairs with exploration controller
python generate_active_interventions.py \
    --controller exploration \
    --num_pairs 50 \
    --out data/active_interventions \
    --no_gif

# Check controller learning progress
python analyze_active_interventions.py \
    --data_dirs data/active_interventions \
    --output_dir analysis/controller_performance
```

### Controller Comparison Study

```bash
# Generate datasets with different controllers
python generate_active_interventions.py --controller random --num_pairs 100 --out data/random
python generate_active_interventions.py --controller max_effect --num_pairs 100 --out data/max_effect  
python generate_active_interventions.py --controller exploration --num_pairs 100 --out data/exploration

# Compare performance
python analyze_active_interventions.py \
    --data_dirs data/random data/max_effect data/exploration \
    --controller_names Random MaxEffect Exploration \
    --output_dir analysis/controller_comparison
```

### Model-Based Active Learning

```bash
# Advanced GP-based controller with acquisition function tuning
python generate_active_interventions.py \
    --controller gaussian_process \
    --acquisition_function ucb \
    --exploration_weight 2.0 \
    --num_pairs 200 \
    --out data/gp_active
```

## Data Output Structure

### Active Intervention Dataset
```
active_interventions/
├── train/
│   ├── pair_00000/
│   │   ├── baseline/
│   │   │   ├── frame_000.png - frame_059.png
│   │   │   └── meta.json
│   │   ├── intervention_hole_larger/
│   │   │   ├── frame_000.png - frame_059.png  
│   │   │   └── meta.json
│   │   └── comparison.json                    # Causal effect analysis
├── test/
├── gifs/ (if enabled)
├── active_intervention_manifest.json          # Controller decisions & results
├── controller_checkpoint_*.json               # Periodic controller state
└── final_controller_state.json               # Final learned state
```

### Controller State Files
```json
{
  "name": "Exploration",
  "history": {
    "results": [...],                          # All intervention results
    "intervention_counts": {...},              # Usage frequency  
    "effect_statistics": {...},               # Running statistics
    "parameter_ranges": {...}                 # Explored parameter bounds
  },
  "parameter_adjustments": {...}              # Learned parameter modifications
}
```

### Analysis Results
```
controller_analysis/
├── active_intervention_analysis.png          # Learning curves & comparisons
├── active_intervention_analysis_report.txt   # Detailed performance report
└── controller_comparison.csv                 # Quantitative metrics
```

## Advanced Features

### Custom Controller Development

```python
class MyCustomController(ActiveController):
    def __init__(self):
        super().__init__("MyCustom")
        self.custom_state = {}
    
    def decide_intervention(self, baseline_params, available_interventions):
        # Implement your decision logic
        chosen_intervention = self.my_selection_algorithm(available_interventions)
        custom_params = self.compute_parameters(baseline_params, chosen_intervention)
        return chosen_intervention, custom_params
    
    def _update_strategy(self, result):
        # Update custom state based on result
        self.custom_state.update(result)
```

### Integration with External Models

```python
# Use external causal models for intervention selection
class ExternalModelController(ActiveController):
    def __init__(self, external_model):
        super().__init__("ExternalModel")
        self.model = external_model
    
    def decide_intervention(self, baseline_params, available_interventions):
        # Query external model for best intervention
        predictions = self.model.predict_effects(baseline_params, available_interventions)
        best_intervention = max(predictions, key=predictions.get)
        return best_intervention, {}
```

## Performance Optimization

### Parallelization
- Controllers run sequentially to maintain state consistency
- Intervention generation within pairs can be parallelized
- Analysis scripts support batch processing

### Checkpointing
- Automatic controller state saving every 10 pairs
- Resume interrupted runs from checkpoints
- Export/import controller states for reproducibility

### Memory Management
- Streaming analysis for large datasets
- Configurable history buffer sizes
- Efficient sparse representation of parameter ranges

## Integration with ELVIS-C

The active intervention system seamlessly integrates with existing ELVIS-C components:

### Compatibility
- **Same Video Format**: Identical PNG frames and metadata structure
- **Analysis Tools**: All existing analysis scripts work with active datasets
- **Visualization**: GIF generation maintains same labeling format
- **Parameter Independence**: Same unbiased parameter generation

### Workflow Integration
```bash
# Standard workflow
python falling_circles.py --num_videos 1000 --out data/standard
python generate_intervention_pairs.py --num_pairs 100 --out data/static

# Active workflow  
python generate_active_interventions.py --controller exploration --num_pairs 100 --out data/active
python analyze_active_interventions.py --data_dirs data/active --output_dir analysis/
```

## Future Extensions

### 1. Multi-Objective Controllers
- Balance multiple objectives (effect size, diversity, speed)
- Pareto-optimal intervention selection
- User-specified preference weights

### 2. Hierarchical Controllers
- High-level strategy selection
- Low-level parameter optimization
- Meta-learning across intervention types

### 3. Collaborative Controllers
- Multiple controllers sharing information
- Ensemble decision-making
- Specialized controllers for different parameter regions

### 4. Online Learning Integration
- Real-time model updates during generation
- Stream-based causal discovery
- Dynamic stopping criteria

## Research Impact

The Active Intervention System enables several important research directions:

### Causal Discovery Efficiency
- **Reduce Experimental Costs**: Focus on informative interventions
- **Accelerate Discovery**: Learn causal relationships faster
- **Improve Robustness**: Test across diverse parameter ranges

### Adaptive Experimental Design
- **Personalized Medicine**: Adapt treatments based on patient responses  
- **Robotics**: Learn manipulation strategies through active exploration
- **AI Safety**: Test AI systems with targeted interventions

### Uncertainty Quantification
- **Confidence Bounds**: Quantify uncertainty about causal effects
- **Risk Assessment**: Identify regions of high uncertainty
- **Robust Decision Making**: Account for epistemic uncertainty

The active intervention system represents a significant advancement in computational causal discovery, enabling researchers to conduct more efficient, targeted, and informative causal inference studies.