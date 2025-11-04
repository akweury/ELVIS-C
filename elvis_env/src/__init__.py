"""
ELVIS-C Core Source Package

Essential falling circles physics simulation and dataset generation tools.

Key modules:
- falling_circles_env: Core physics simulation engine
- generate_falling_circles_dataset: Large-scale dataset generation
- ai_model_interface: AI intervention interface
- test_interface: Usage examples and testing

Quick Start:
    from falling_circles_env import FallingCirclesEnvironment, VideoParams
    from ai_model_interface import generate_intervention_video
    
    # Generate a single video
    params = VideoParams(hole_diameter=40, num_circles=8)
    env = FallingCirclesEnvironment(params)
    frames, metadata = env.generate_video()
    
    # AI intervention
    result = generate_intervention_video(
        baseline_parameters={"hole_diameter": 30},
        intervention_target="hole_diameter", 
        intervention_value=60
    )
"""

__version__ = "1.0.0"
__author__ = "ELVIS-C Project"

# Import main classes and functions for easy access
from .falling_circles_env import FallingCirclesEnvironment, VideoParams
from .ai_model_interface import generate_intervention_video, AIVideoInterface

__all__ = [
    'FallingCirclesEnvironment',
    'VideoParams', 
    'generate_intervention_video',
    'AIVideoInterface'
]