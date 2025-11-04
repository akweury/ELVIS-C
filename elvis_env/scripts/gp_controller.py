#!/usr/bin/env python3
"""
Gaussian Process Controller for Model-Based Active Learning

Uses Gaussian Process regression to learn causal effect functions and selects
interventions based on uncertainty estimation and expected information gain.

This enables sophisticated model-based active causal discovery.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from active_controllers import ActiveController, InterventionResult

class GaussianProcessController(ActiveController):
    """
    Model-based controller using Gaussian Process for uncertainty-guided exploration
    
    This controller learns a probabilistic model of the causal effect function
    and selects interventions to maximize information gain.
    """
    
    def __init__(self, 
                 acquisition_function: str = 'ucb',
                 exploration_weight: float = 2.0,
                 min_samples_per_intervention: int = 2):
        """
        Initialize GP controller
        
        Args:
            acquisition_function: 'ucb', 'ei', or 'pi' for acquisition strategy
            exploration_weight: Weight for exploration vs exploitation
            min_samples_per_intervention: Minimum samples before using GP
        """
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for GaussianProcessController. "
                            "Install with: pip install scikit-learn")
        
        super().__init__("GaussianProcess")
        self.acquisition_function = acquisition_function
        self.exploration_weight = exploration_weight
        self.min_samples_per_intervention = min_samples_per_intervention
        
        # GP models for different intervention types
        self.gp_models = {}
        self.feature_scalers = {}
        self.target_scalers = {}
        
        # Training data cache
        self.intervention_features = {}  # intervention_type -> list of feature vectors
        self.intervention_targets = {}   # intervention_type -> list of effect magnitudes
        
    def decide_intervention(self, baseline_params, available_interventions: List[str]) -> Tuple[str, Dict[str, Any]]:
        """Select intervention using GP-guided acquisition function"""
        
        # Check if we need more samples for any intervention type
        for intervention in available_interventions:
            count = self.history.intervention_counts.get(intervention, 0)
            if count < self.min_samples_per_intervention:
                # Use default parameters for initial exploration
                return intervention, {}
        
        # Use GP-based selection if we have enough data
        if len(self.history.results) >= len(available_interventions) * self.min_samples_per_intervention:
            return self._gp_based_selection(baseline_params, available_interventions)
        else:
            # Fall back to random selection for initial data collection
            return np.random.choice(available_interventions), {}
    
    def _gp_based_selection(self, baseline_params, available_interventions: List[str]) -> Tuple[str, Dict[str, Any]]:
        """Select intervention using trained GP models"""
        
        best_score = -np.inf
        best_intervention = None
        best_params = {}
        
        for intervention in available_interventions:
            # Generate candidate parameter values
            candidate_params = self._generate_candidate_parameters(intervention, baseline_params)
            
            for params in candidate_params:
                # Convert to feature vector
                features = self._params_to_features(baseline_params, intervention, params)
                
                # Compute acquisition score
                score = self._compute_acquisition_score(intervention, features)
                
                if score > best_score:
                    best_score = score
                    best_intervention = intervention
                    best_params = params
        
        return best_intervention, best_params
    
    def _generate_candidate_parameters(self, intervention: str, baseline_params) -> List[Dict[str, Any]]:
        """Generate candidate parameter values for given intervention type"""
        
        candidates = []
        
        if intervention == 'hole_larger':
            # Try different hole size increases
            for delta in [5, 10, 15, 20]:
                new_diameter = min(50, baseline_params.hole_diameter + delta)
                candidates.append({'hole_diameter': new_diameter})
                
        elif intervention == 'hole_smaller':
            # Try different hole size decreases
            for delta in [5, 10, 15, 20]:
                new_diameter = max(10, baseline_params.hole_diameter - delta)
                candidates.append({'hole_diameter': new_diameter})
                
        elif intervention == 'wind_strength_high':
            # Try different wind strength multipliers
            for multiplier in [1.5, 2.0, 2.5, 3.0]:
                new_strength = min(0.4, baseline_params.wind_strength * multiplier)
                candidates.append({'wind_strength': new_strength})
                
        elif intervention == 'larger_circles':
            # Try different circle size increases
            for delta in [2, 4, 6, 8]:
                new_min = min(15, baseline_params.circle_size_min + delta)
                new_max = min(20, baseline_params.circle_size_max + delta)
                candidates.append({'circle_size_min': new_min, 'circle_size_max': new_max})
                
        elif intervention == 'smaller_circles':
            # Try different circle size decreases
            for delta in [2, 4, 6, 8]:
                new_min = max(2, baseline_params.circle_size_min - delta)
                new_max = max(3, baseline_params.circle_size_max - delta)
                candidates.append({'circle_size_min': new_min, 'circle_size_max': new_max})
        
        # If no specific candidates, use empty dict for default behavior
        if not candidates:
            candidates = [{}]
            
        return candidates
    
    def _params_to_features(self, baseline_params, intervention: str, intervention_params: Dict[str, Any]) -> np.ndarray:
        """Convert parameters to feature vector for GP input"""
        
        # Create feature vector from baseline parameters and intervention
        features = [
            baseline_params.hole_diameter,
            baseline_params.wind_strength,
            baseline_params.wind_direction,
            baseline_params.num_circles,
            (baseline_params.circle_size_min + baseline_params.circle_size_max) / 2,  # Average circle size
            baseline_params.spawn_rate
        ]
        
        # Add intervention-specific features
        if intervention_params:
            for param_name, param_value in intervention_params.items():
                if param_name == 'hole_diameter':
                    features.append(param_value - baseline_params.hole_diameter)  # Delta
                elif param_name == 'wind_strength':
                    features.append(param_value / max(baseline_params.wind_strength, 0.001))  # Ratio
                elif param_name in ['circle_size_min', 'circle_size_max']:
                    baseline_avg = (baseline_params.circle_size_min + baseline_params.circle_size_max) / 2
                    features.append(param_value - baseline_avg)  # Delta from baseline average
                else:
                    features.append(param_value)
        else:
            # Use default intervention features
            features.extend([0, 0, 0])  # Placeholder features
        
        return np.array(features)
    
    def _compute_acquisition_score(self, intervention: str, features: np.ndarray) -> float:
        """Compute acquisition function score for given features"""
        
        if intervention not in self.gp_models:
            return np.random.random()  # Random score if no model yet
        
        gp_model = self.gp_models[intervention]
        scaler = self.feature_scalers[intervention]
        
        # Scale features
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # Predict mean and variance
        mean, std = gp_model.predict(features_scaled, return_std=True)
        mean = mean[0]
        std = std[0]
        
        # Compute acquisition score
        if self.acquisition_function == 'ucb':
            # Upper Confidence Bound
            score = mean + self.exploration_weight * std
        elif self.acquisition_function == 'ei':
            # Expected Improvement
            best_value = np.max(self.intervention_targets.get(intervention, [0]))
            z = (mean - best_value) / (std + 1e-9)
            score = (mean - best_value) * self._normal_cdf(z) + std * self._normal_pdf(z)
        elif self.acquisition_function == 'pi':
            # Probability of Improvement
            best_value = np.max(self.intervention_targets.get(intervention, [0]))
            z = (mean - best_value) / (std + 1e-9)
            score = self._normal_cdf(z)
        else:
            score = mean + std  # Default UCB-like
        
        return score
    
    def _normal_cdf(self, x):
        """Standard normal CDF approximation"""
        return 0.5 * (1 + np.tanh(x * np.sqrt(2 / np.pi)))
    
    def _normal_pdf(self, x):
        """Standard normal PDF"""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    def _update_strategy(self, result: InterventionResult):
        """Update GP models with new intervention result"""
        
        intervention = result.intervention_name
        
        # Initialize storage for this intervention type
        if intervention not in self.intervention_features:
            self.intervention_features[intervention] = []
            self.intervention_targets[intervention] = []
        
        # Extract features from result
        # This is a simplified feature extraction - in practice, you'd want more sophisticated features
        baseline_params_dict = {}  # We'd need to store baseline params in the result
        
        # For now, use intervention parameters as proxy features
        feature_vector = []
        for param_name in ['hole_diameter', 'wind_strength', 'circle_size_min', 'num_circles']:
            if param_name in ['circle_size_min', 'circle_size_max']:
                # Use average for circle size features
                min_size = result.intervention_params.get('circle_size_min', 0)
                max_size = result.intervention_params.get('circle_size_max', 0)
                feature_vector.append((min_size + max_size) / 2 if min_size and max_size else 0)
            else:
                feature_vector.append(result.intervention_params.get(param_name, 0))
        
        # Add baseline outcome features
        feature_vector.extend([
            result.baseline_outcome.get('exit_ratio', 0),
            1.0 if result.baseline_outcome.get('jam_type') == 'no_jam' else 0.0,
            1.0 if result.baseline_outcome.get('jam_type') == 'partial_jam' else 0.0
        ])
        
        features = np.array(feature_vector)
        target = result.effect_magnitude
        
        # Store training data
        self.intervention_features[intervention].append(features)
        self.intervention_targets[intervention].append(target)
        
        # Retrain GP model if we have enough data
        if len(self.intervention_features[intervention]) >= 3:
            self._train_gp_model(intervention)
    
    def _train_gp_model(self, intervention: str):
        """Train GP model for specific intervention type"""
        
        X = np.array(self.intervention_features[intervention])
        y = np.array(self.intervention_targets[intervention])
        
        # Initialize scalers if needed
        if intervention not in self.feature_scalers:
            self.feature_scalers[intervention] = StandardScaler()
            self.target_scalers[intervention] = StandardScaler()
        
        # Scale features and targets
        X_scaled = self.feature_scalers[intervention].fit_transform(X)
        y_scaled = self.target_scalers[intervention].fit_transform(y.reshape(-1, 1)).ravel()
        
        # Create and train GP model
        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=False,
            n_restarts_optimizer=2
        )
        
        try:
            gp.fit(X_scaled, y_scaled)
            self.gp_models[intervention] = gp
        except Exception as e:
            print(f"Warning: Failed to train GP model for {intervention}: {e}")
    
    def get_model_predictions(self, intervention: str, test_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get GP model predictions for test features"""
        
        if intervention not in self.gp_models:
            return None, None
        
        gp_model = self.gp_models[intervention]
        scaler = self.feature_scalers[intervention]
        target_scaler = self.target_scalers[intervention]
        
        # Scale features
        test_features_scaled = scaler.transform(test_features.reshape(1, -1))
        
        # Predict
        mean_scaled, std_scaled = gp_model.predict(test_features_scaled, return_std=True)
        
        # Inverse transform targets
        mean = target_scaler.inverse_transform(mean_scaled.reshape(-1, 1)).ravel()
        std = std_scaled * target_scaler.scale_  # Approximate inverse transform for std
        
        return mean, std
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return GP-specific statistics"""
        base_stats = super().get_statistics()
        
        gp_stats = {
            'gp_models_trained': list(self.gp_models.keys()),
            'acquisition_function': self.acquisition_function,
            'exploration_weight': self.exploration_weight,
            'training_data_sizes': {
                intervention: len(features) 
                for intervention, features in self.intervention_features.items()
            }
        }
        
        base_stats.update(gp_stats)
        return base_stats

def create_gp_controller(**kwargs) -> GaussianProcessController:
    """Factory function for GP controller"""
    return GaussianProcessController(**kwargs)