"""
ELVIS-C Interface Package

Provides simple, AI-friendly interfaces for video generation and intervention tasks.
"""

from .video_interface import VideoInterface, InterventionResult
from .intervention_interface import InterventionInterface, quick_intervene, compare_parameters
from .ai_model_interface import AIVideoInterface, VideoGenerationResult, generate_intervention_video

__all__ = [
    'VideoInterface', 'InterventionInterface', 'InterventionResult', 
    'quick_intervene', 'compare_parameters',
    'AIVideoInterface', 'VideoGenerationResult', 'generate_intervention_video'
]