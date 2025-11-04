"""
Intervention Interface for AI Models

High-level interface specifically designed for AI models to perform interventions
on the falling circles environment. Provides simple methods with intelligent defaults.
"""

import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scripts.falling_circles import VideoParams
from .video_interface import VideoInterface, InterventionResult


@dataclass
class InterventionRequest:
    """
    Simple intervention request that AI models can easily construct
    """
    baseline: Dict[str, Any]  # Baseline parameters
    intervention: Dict[str, Any]  # What to change
    seed: Optional[int] = None
    description: Optional[str] = None


class InterventionInterface:
    """
    High-level interface for AI models to perform interventions
    
    This interface is designed to be extremely simple for AI models to use:
    1. Specify baseline parameters
    2. Specify what to intervene on
    3. Get back the intervention videos and analysis
    
    The interface handles all the complexity of parameter validation,
    video generation, and effect analysis.
    """
    
    def __init__(self, default_seed: Optional[int] = 42):
        """
        Initialize intervention interface
        
        Args:
            default_seed: Default seed for reproducible results
        """
        self.video_interface = VideoInterface(default_seed=default_seed)
        self.default_params = self._get_default_params()
    
    def intervene(
        self,
        baseline_params: Optional[Dict] = None,
        intervention_target: str = "hole_diameter",
        intervention_value: Union[float, int] = None,
        intervention_change: Optional[float] = None,
        seed: Optional[int] = None,
        description: Optional[str] = None
    ) -> InterventionResult:
        """
        Perform a simple intervention on a single parameter
        
        Args:
            baseline_params: Base parameters (uses defaults if None)
            intervention_target: Parameter to intervene on
            intervention_value: Absolute value to set (if provided)
            intervention_change: Relative change to apply (if intervention_value not provided)
            seed: Random seed
            description: Human description of intervention
            
        Returns:
            InterventionResult with videos and analysis
            
        Example:
            # Increase hole diameter by 20
            result = interface.intervene(
                intervention_target="hole_diameter",
                intervention_change=20
            )
            
            # Set wind strength to 5.0
            result = interface.intervene(
                intervention_target="wind_strength", 
                intervention_value=5.0
            )
        """
        # Use default params if none provided
        if baseline_params is None:
            baseline_params = self.default_params.copy()
        else:
            # Merge with defaults for any missing parameters
            full_params = self.default_params.copy()
            full_params.update(baseline_params)
            baseline_params = full_params
        
        # Create intervention parameters
        intervention_params = baseline_params.copy()
        
        if intervention_value is not None:
            # Set absolute value
            intervention_params[intervention_target] = intervention_value
        elif intervention_change is not None:
            # Apply relative change
            current_value = baseline_params[intervention_target]
            intervention_params[intervention_target] = current_value + intervention_change
        else:
            raise ValueError("Must provide either intervention_value or intervention_change")
        
        # Validate parameters
        intervention_params = self._validate_params(intervention_params)
        
        # Generate intervention
        result = self.video_interface.create_intervention(
            baseline_params=baseline_params,
            intervention_params=intervention_params,
            seed=seed
        )
        
        # Add custom description if provided
        if description:
            result.effect_description = f"{description}. {result.effect_description}"
        
        return result
    
    def multi_intervene(
        self,
        baseline_params: Optional[Dict] = None,
        interventions: Dict[str, Union[float, int]] = None,
        seed: Optional[int] = None,
        description: Optional[str] = None
    ) -> InterventionResult:
        """
        Perform interventions on multiple parameters simultaneously
        
        Args:
            baseline_params: Base parameters (uses defaults if None)
            interventions: Dict mapping parameter names to new values
            seed: Random seed
            description: Human description of intervention
            
        Returns:
            InterventionResult with videos and analysis
            
        Example:
            result = interface.multi_intervene(
                interventions={
                    "hole_diameter": 60,
                    "wind_strength": 4.0,
                    "num_circles": 8
                },
                description="Increase hole size, wind, and circle count"
            )
        """
        # Use default params if none provided
        if baseline_params is None:
            baseline_params = self.default_params.copy()
        else:
            full_params = self.default_params.copy()
            full_params.update(baseline_params)
            baseline_params = full_params
        
        if interventions is None:
            raise ValueError("Must provide interventions dict")
        
        # Create intervention parameters
        intervention_params = baseline_params.copy()
        intervention_params.update(interventions)
        
        # Validate parameters
        intervention_params = self._validate_params(intervention_params)
        
        # Generate intervention
        result = self.video_interface.create_intervention(
            baseline_params=baseline_params,
            intervention_params=intervention_params,
            seed=seed
        )
        
        # Add custom description if provided
        if description:
            result.effect_description = f"{description}. {result.effect_description}"
        
        return result
    
    def batch_intervene(
        self,
        intervention_requests: List[InterventionRequest]
    ) -> List[InterventionResult]:
        """
        Perform multiple interventions in batch
        
        Args:
            intervention_requests: List of InterventionRequest objects
            
        Returns:
            List of InterventionResult objects
            
        Example:
            requests = [
                InterventionRequest(
                    baseline={"hole_diameter": 40},
                    intervention={"hole_diameter": 60},
                    description="Increase hole size"
                ),
                InterventionRequest(
                    baseline={"wind_strength": 2.0},
                    intervention={"wind_strength": 5.0},
                    description="Increase wind"
                )
            ]
            results = interface.batch_intervene(requests)
        """
        results = []
        for request in intervention_requests:
            result = self.video_interface.create_intervention(
                baseline_params=request.baseline,
                intervention_params=request.intervention,
                seed=request.seed
            )
            if request.description:
                result.effect_description = f"{request.description}. {result.effect_description}"
            results.append(result)
        return results
    
    def explore_parameter(
        self,
        parameter_name: str,
        values: List[Union[float, int]],
        baseline_params: Optional[Dict] = None,
        seed: Optional[int] = None
    ) -> List[InterventionResult]:
        """
        Explore different values of a single parameter
        
        Args:
            parameter_name: Name of parameter to explore
            values: List of values to try
            baseline_params: Base parameters
            seed: Random seed
            
        Returns:
            List of InterventionResult objects, one for each value
            
        Example:
            # Explore different hole sizes
            results = interface.explore_parameter(
                parameter_name="hole_diameter",
                values=[20, 40, 60, 80, 100]
            )
        """
        if baseline_params is None:
            baseline_params = self.default_params.copy()
        
        results = []
        for value in values:
            intervention_params = baseline_params.copy()
            intervention_params[parameter_name] = value
            
            result = self.video_interface.create_intervention(
                baseline_params=baseline_params,
                intervention_params=intervention_params,
                seed=seed
            )
            results.append(result)
        
        return results
    
    def get_parameter_info(self) -> Dict[str, Dict]:
        """
        Get information about available parameters and their valid ranges
        
        Returns:
            Dict with parameter info including descriptions and valid ranges
        """
        return {
            "hole_diameter": {
                "description": "Size of the exit hole at bottom",
                "type": "int",
                "min": 10,
                "max": 200,
                "default": 40,
                "effect": "Larger holes allow more circles to exit"
            },
            "wind_strength": {
                "description": "Horizontal wind force",
                "type": "float",
                "min": 0.0,
                "max": 10.0,
                "default": 2.0,
                "effect": "Higher wind pushes circles sideways"
            },
            "num_circles": {
                "description": "Maximum number of circles to spawn",
                "type": "int", 
                "min": 1,
                "max": 50,
                "default": 5,
                "effect": "More circles can create traffic jams"
            },
            "circle_size_min": {
                "description": "Minimum circle radius",
                "type": "int",
                "min": 3,
                "max": 30,
                "default": 8,
                "effect": "Larger circles have harder time fitting through hole"
            },
            "circle_size_max": {
                "description": "Maximum circle radius", 
                "type": "int",
                "min": 3,
                "max": 30,
                "default": 12,
                "effect": "Larger circles have harder time fitting through hole"
            },
            "spawn_rate": {
                "description": "Probability of spawning new circles per frame",
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "default": 0.3,
                "effect": "Higher values spawn circles more frequently, causing congestion"
            },
            "hole_x_position": {
                "description": "Horizontal position of hole (fraction of width)",
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "default": 0.5,
                "effect": "Changes where circles need to move to exit"
            },
            "wind_direction": {
                "description": "Wind direction (1 = left to right, -1 = right to left)",
                "type": "int",
                "min": -1,
                "max": 1,
                "default": 1,
                "effect": "Changes which direction wind pushes circles"
            },
            "gravity": {
                "description": "Gravitational acceleration",
                "type": "float",
                "min": 0.1,
                "max": 2.0,
                "default": 0.8,
                "effect": "Higher gravity makes circles fall faster"
            },
            "noise_level": {
                "description": "Amount of visual noise in frames",
                "type": "float",
                "min": 0.0,
                "max": 0.2,
                "default": 0.0,
                "effect": "Higher values add more noise to video frames"
            }
        }
    
    def _get_default_params(self) -> Dict:
        """Get default parameters as dictionary"""
        default_video_params = VideoInterface.create_default_params()
        return default_video_params.to_dict()
    
    def _validate_params(self, params: Dict) -> Dict:
        """
        Validate and constrain parameters to valid ranges
        
        Args:
            params: Parameter dictionary
            
        Returns:
            Validated parameter dictionary
        """
        param_info = self.get_parameter_info()
        validated = params.copy()
        
        for param_name, value in params.items():
            if param_name in param_info:
                info = param_info[param_name]
                # Constrain to valid range
                if "min" in info and value < info["min"]:
                    validated[param_name] = info["min"]
                    print(f"Warning: {param_name}={value} below minimum, using {info['min']}")
                elif "max" in info and value > info["max"]:
                    validated[param_name] = info["max"]
                    print(f"Warning: {param_name}={value} above maximum, using {info['max']}")
        
        return validated


# Convenience functions for quick access
def quick_intervene(
    parameter: str, 
    value: Union[float, int], 
    seed: Optional[int] = None
) -> InterventionResult:
    """
    Quick intervention on a single parameter
    
    Args:
        parameter: Parameter name to change
        value: New value for parameter
        seed: Random seed
        
    Returns:
        InterventionResult
        
    Example:
        result = quick_intervene("hole_diameter", 80)
    """
    interface = InterventionInterface(default_seed=seed)
    return interface.intervene(
        intervention_target=parameter,
        intervention_value=value,
        seed=seed
    )


def compare_parameters(
    parameter: str,
    values: List[Union[float, int]],
    seed: Optional[int] = None
) -> List[InterventionResult]:
    """
    Compare multiple values of a parameter
    
    Args:
        parameter: Parameter name
        values: List of values to compare
        seed: Random seed
        
    Returns:
        List of InterventionResult objects
        
    Example:
        results = compare_parameters("wind_strength", [1.0, 3.0, 5.0])
    """
    interface = InterventionInterface(default_seed=seed)
    return interface.explore_parameter(parameter, values, seed=seed)