#!/usr/bin/env python3
"""
Active Controllers for Dynamic Intervention Selection

Controllers that can adaptively choose which variables to intervene on and by how much,
based on past results and accumulated knowledge about causal effects.

This enables sophisticated causal discovery strategies that go beyond fixed intervention sets.
"""

import numpy as np
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import logging

@dataclass
class InterventionResult:
    """Results from a single intervention pair"""
    intervention_name: str
    intervention_params: Dict[str, Any]
    baseline_outcome: Dict[str, float]  # jam_type, exit_ratio, etc.
    intervention_outcome: Dict[str, float]
    effect_magnitude: float  # Overall effect size
    jam_type_changed: bool
    exit_ratio_delta: float
    confidence: float = 1.0  # Confidence in this result

@dataclass  
class InterventionHistory:
    """Accumulated history of all intervention results"""
    results: List[InterventionResult]
    intervention_counts: Dict[str, int]
    effect_statistics: Dict[str, Dict[str, float]]  # intervention -> {mean_effect, std, success_rate}
    parameter_ranges: Dict[str, Tuple[float, float]]  # Parameter bounds explored
    
    def __post_init__(self):
        if not hasattr(self, 'results'):
            self.results = []
        if not hasattr(self, 'intervention_counts'):
            self.intervention_counts = {}
        if not hasattr(self, 'effect_statistics'):
            self.effect_statistics = {}
        if not hasattr(self, 'parameter_ranges'):
            self.parameter_ranges = {}

    def add_result(self, result: InterventionResult):
        """Add a new intervention result and update statistics"""
        self.results.append(result)
        
        # Update counts
        if result.intervention_name not in self.intervention_counts:
            self.intervention_counts[result.intervention_name] = 0
        self.intervention_counts[result.intervention_name] += 1
        
        # Update effect statistics
        self._update_effect_statistics(result)
        self._update_parameter_ranges(result)
    
    def _update_effect_statistics(self, result: InterventionResult):
        """Update running statistics for intervention effects"""
        name = result.intervention_name
        if name not in self.effect_statistics:
            self.effect_statistics[name] = {
                'effects': [],
                'mean_effect': 0.0,
                'std_effect': 0.0,
                'success_rate': 0.0,
                'count': 0
            }
        
        stats = self.effect_statistics[name]
        stats['effects'].append(result.effect_magnitude)
        stats['count'] += 1
        
        # Update running statistics
        effects = stats['effects']
        stats['mean_effect'] = np.mean(effects)
        stats['std_effect'] = np.std(effects) if len(effects) > 1 else 0.0
        
        # Success rate (proportion with jam type change)
        successes = sum(1 for r in self.results if r.intervention_name == name and r.jam_type_changed)
        stats['success_rate'] = successes / stats['count']
    
    def _update_parameter_ranges(self, result: InterventionResult):
        """Track parameter ranges that have been explored"""
        for param_name, param_value in result.intervention_params.items():
            if isinstance(param_value, (int, float)):
                if param_name not in self.parameter_ranges:
                    self.parameter_ranges[param_name] = (param_value, param_value)
                else:
                    min_val, max_val = self.parameter_ranges[param_name]
                    self.parameter_ranges[param_name] = (
                        min(min_val, param_value),
                        max(max_val, param_value)
                    )

class ActiveController(ABC):
    """Abstract base class for active intervention controllers"""
    
    def __init__(self, name: str):
        self.name = name
        self.history = InterventionHistory([], {}, {}, {})
    
    @abstractmethod
    def decide_intervention(self, baseline_params, available_interventions: List[str]) -> Tuple[str, Dict[str, Any]]:
        """
        Decide which intervention to perform and with what parameters
        
        Args:
            baseline_params: The baseline video parameters
            available_interventions: List of possible intervention types
            
        Returns:
            Tuple of (intervention_name, intervention_parameters)
        """
        pass
    
    def update_from_results(self, result: InterventionResult):
        """Update controller state based on intervention results"""
        self.history.add_result(result)
        self._update_strategy(result)
    
    def _update_strategy(self, result: InterventionResult):
        """Override in subclasses to update strategy based on results"""
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return current controller statistics"""
        return {
            'name': self.name,
            'total_interventions': len(self.history.results),
            'intervention_counts': self.history.intervention_counts,
            'effect_statistics': self.history.effect_statistics,
            'parameter_ranges': self.history.parameter_ranges
        }

class RandomController(ActiveController):
    """Baseline controller that randomly selects interventions"""
    
    def __init__(self):
        super().__init__("Random")
    
    def decide_intervention(self, baseline_params, available_interventions: List[str]) -> Tuple[str, Dict[str, Any]]:
        """Randomly select an intervention type"""
        intervention_name = random.choice(available_interventions)
        
        # Use default intervention parameters (will be computed by intervention system)
        return intervention_name, {}

class MaxEffectController(ActiveController):
    """Controller that targets interventions with highest observed effects"""
    
    def __init__(self, exploration_rate: float = 0.2):
        super().__init__("MaxEffect")
        self.exploration_rate = exploration_rate
    
    def decide_intervention(self, baseline_params, available_interventions: List[str]) -> Tuple[str, Dict[str, Any]]:
        """Select intervention with highest expected effect, with some exploration"""
        
        # Exploration: try untested interventions
        untested = [name for name in available_interventions 
                   if name not in self.history.intervention_counts]
        
        if untested and random.random() < self.exploration_rate:
            return random.choice(untested), {}
        
        # Exploitation: choose intervention with highest mean effect
        if not self.history.effect_statistics:
            return random.choice(available_interventions), {}
        
        best_intervention = max(
            available_interventions,
            key=lambda name: self.history.effect_statistics.get(name, {}).get('mean_effect', 0.0)
        )
        
        return best_intervention, {}

class UncertaintyController(ActiveController):
    """Controller that targets interventions with highest uncertainty"""
    
    def __init__(self, min_samples: int = 3):
        super().__init__("Uncertainty")
        self.min_samples = min_samples
    
    def decide_intervention(self, baseline_params, available_interventions: List[str]) -> Tuple[str, Dict[str, Any]]:
        """Select intervention with highest uncertainty (standard deviation)"""
        
        # First, ensure minimum samples for all interventions
        for intervention in available_interventions:
            count = self.history.intervention_counts.get(intervention, 0)
            if count < self.min_samples:
                return intervention, {}
        
        # Then target highest uncertainty
        if not self.history.effect_statistics:
            return random.choice(available_interventions), {}
        
        def get_uncertainty(name):
            stats = self.history.effect_statistics.get(name, {})
            return stats.get('std_effect', float('inf'))
        
        most_uncertain = max(available_interventions, key=get_uncertainty)
        return most_uncertain, {}

class ExplorationController(ActiveController):
    """Controller that balances exploration and exploitation using UCB-like strategy"""
    
    def __init__(self, exploration_constant: float = 1.4):
        super().__init__("Exploration")
        self.exploration_constant = exploration_constant
    
    def decide_intervention(self, baseline_params, available_interventions: List[str]) -> Tuple[str, Dict[str, Any]]:
        """Select intervention using Upper Confidence Bound strategy"""
        
        total_trials = len(self.history.results)
        if total_trials == 0:
            return random.choice(available_interventions), {}
        
        def ucb_score(intervention_name):
            stats = self.history.effect_statistics.get(intervention_name, {})
            mean_effect = stats.get('mean_effect', 0.0)
            count = stats.get('count', 0)
            
            if count == 0:
                return float('inf')  # Prioritize untested interventions
            
            # UCB formula: mean + sqrt(2 * log(total_trials) / trials_for_this_intervention)
            confidence_bonus = self.exploration_constant * np.sqrt(2 * np.log(total_trials) / count)
            return mean_effect + confidence_bonus
        
        best_intervention = max(available_interventions, key=ucb_score)
        return best_intervention, {}

class AdaptiveParameterController(ActiveController):
    """Controller that can adaptively modify intervention magnitudes"""
    
    def __init__(self, learning_rate: float = 0.1):
        super().__init__("AdaptiveParameter")
        self.learning_rate = learning_rate
        self.parameter_adjustments = {}  # intervention -> parameter adjustments learned
    
    def decide_intervention(self, baseline_params, available_interventions: List[str]) -> Tuple[str, Dict[str, Any]]:
        """Select intervention and adaptively choose parameter values"""
        
        # Use exploration strategy to choose intervention type
        intervention_name = self._choose_intervention_type(available_interventions)
        
        # Adaptively choose parameter values
        intervention_params = self._choose_intervention_parameters(intervention_name, baseline_params)
        
        return intervention_name, intervention_params
    
    def _choose_intervention_type(self, available_interventions: List[str]) -> str:
        """Choose intervention type using UCB strategy"""
        total_trials = len(self.history.results)
        if total_trials == 0:
            return random.choice(available_interventions)
        
        def ucb_score(intervention_name):
            stats = self.history.effect_statistics.get(intervention_name, {})
            mean_effect = stats.get('mean_effect', 0.0)
            count = stats.get('count', 0)
            
            if count == 0:
                return float('inf')
            
            confidence_bonus = 1.4 * np.sqrt(2 * np.log(total_trials) / count)
            return mean_effect + confidence_bonus
        
        return max(available_interventions, key=ucb_score)
    
    def _choose_intervention_parameters(self, intervention_name: str, baseline_params) -> Dict[str, Any]:
        """Adaptively choose intervention parameter values"""
        
        # Get learned adjustments for this intervention type
        adjustments = self.parameter_adjustments.get(intervention_name, {})
        
        # Default intervention parameters with adaptive modifications
        if intervention_name == 'hole_larger':
            base_increase = 10
            learned_adjustment = adjustments.get('hole_diameter_delta', 0)
            total_increase = base_increase + learned_adjustment
            return {'hole_diameter': min(50, baseline_params.hole_diameter + total_increase)}
            
        elif intervention_name == 'hole_smaller':
            base_decrease = 10
            learned_adjustment = adjustments.get('hole_diameter_delta', 0)
            total_decrease = base_decrease + learned_adjustment
            return {'hole_diameter': max(10, baseline_params.hole_diameter - total_decrease)}
            
        elif intervention_name == 'wind_strength_high':
            base_multiplier = 2.0
            learned_adjustment = adjustments.get('wind_multiplier_delta', 0)
            total_multiplier = base_multiplier + learned_adjustment
            return {'wind_strength': min(0.4, baseline_params.wind_strength * total_multiplier)}
        
        # For other interventions, use default parameters
        return {}
    
    def _update_strategy(self, result: InterventionResult):
        """Update parameter adjustments based on intervention results"""
        intervention_name = result.intervention_name
        effect_magnitude = result.effect_magnitude
        
        if intervention_name not in self.parameter_adjustments:
            self.parameter_adjustments[intervention_name] = {}
        
        # Simple gradient-based update
        for param_name, param_value in result.intervention_params.items():
            if param_name == 'hole_diameter' and 'hole' in intervention_name:
                delta_key = 'hole_diameter_delta'
                current_delta = self.parameter_adjustments[intervention_name].get(delta_key, 0)
                
                # Update delta based on effect magnitude
                if effect_magnitude > 0.3:  # Strong effect - good direction
                    adjustment = self.learning_rate * 2  # Increase magnitude
                elif effect_magnitude < 0.1:  # Weak effect - adjust
                    adjustment = -self.learning_rate * 1  # Decrease magnitude
                else:
                    adjustment = 0  # Maintain current setting
                
                self.parameter_adjustments[intervention_name][delta_key] = current_delta + adjustment

# Helper function to create controllers
def create_controller(controller_type: str, **kwargs) -> ActiveController:
    """Factory function to create controllers"""
    
    controllers = {
        'random': RandomController,
        'max_effect': MaxEffectController,
        'uncertainty': UncertaintyController,
        'exploration': ExplorationController,
        'adaptive': AdaptiveParameterController
    }
    
    if controller_type not in controllers:
        raise ValueError(f"Unknown controller type: {controller_type}. Available: {list(controllers.keys())}")
    
    return controllers[controller_type](**kwargs)

def save_controller_state(controller: ActiveController, filepath: str):
    """Save controller state to file"""
    state = {
        'name': controller.name,
        'history': {
            'results': [
                {
                    'intervention_name': r.intervention_name,
                    'intervention_params': r.intervention_params,
                    'baseline_outcome': r.baseline_outcome,
                    'intervention_outcome': r.intervention_outcome,
                    'effect_magnitude': r.effect_magnitude,
                    'jam_type_changed': r.jam_type_changed,
                    'exit_ratio_delta': r.exit_ratio_delta,
                    'confidence': r.confidence
                }
                for r in controller.history.results
            ],
            'intervention_counts': controller.history.intervention_counts,
            'effect_statistics': controller.history.effect_statistics,
            'parameter_ranges': controller.history.parameter_ranges
        }
    }
    
    # Add controller-specific state
    if hasattr(controller, 'parameter_adjustments'):
        state['parameter_adjustments'] = controller.parameter_adjustments
    
    with open(filepath, 'w') as f:
        json.dump(state, f, indent=2)

def load_controller_state(filepath: str, controller_type: str) -> ActiveController:
    """Load controller state from file"""
    with open(filepath, 'r') as f:
        state = json.load(f)
    
    # Create controller
    controller = create_controller(controller_type)
    
    # Restore history
    history_data = state['history']
    controller.history = InterventionHistory([], {}, {}, {})
    
    # Restore results
    for result_data in history_data['results']:
        result = InterventionResult(**result_data)
        controller.history.add_result(result)
    
    # Restore controller-specific state
    if 'parameter_adjustments' in state and hasattr(controller, 'parameter_adjustments'):
        controller.parameter_adjustments = state['parameter_adjustments']
    
    return controller