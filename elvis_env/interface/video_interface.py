"""
Video Interface for AI Models

Provides a simple, clean interface for AI models to generate falling circles videos
with optional interventions.
"""

import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from PIL import Image

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scripts.falling_circles_env import VideoParams, generate_falling_circles_video


@dataclass
class InterventionResult:
    """
    Result from an intervention operation
    """
    baseline_frames: List[np.ndarray]
    intervention_frames: List[np.ndarray]
    baseline_metadata: Dict
    intervention_metadata: Dict
    effect_magnitude: float
    effect_description: str


class VideoInterface:
    """
    Simple interface for AI models to generate falling circles videos
    
    This class provides clean methods for:
    1. Generating single videos with given parameters
    2. Creating intervention pairs (baseline vs intervention)
    3. Converting between different parameter formats
    4. Saving results in various formats
    """
    
    def __init__(self, default_seed: Optional[int] = None):
        """
        Initialize the video interface
        
        Args:
            default_seed: Default random seed for reproducible results
        """
        self.default_seed = default_seed
    
    def generate_video(
        self, 
        params: Union[Dict, VideoParams],
        seed: Optional[int] = None,
        include_labels: bool = True
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        Generate a single falling circles video
        
        Args:
            params: Video parameters (dict or VideoParams object)
            seed: Random seed (uses default_seed if None)
            include_labels: Whether to include visual labels in frames
            
        Returns:
            Tuple of (frames, metadata)
            - frames: List of numpy arrays representing video frames
            - metadata: Dictionary with generation metadata
        """
        # Convert dict to VideoParams if needed
        if isinstance(params, dict):
            params = VideoParams(**params)
        
        # Use provided seed or default
        if seed is None:
            seed = self.default_seed
            
        # Generate video
        frames, metadata = generate_falling_circles_video(
            params=params,
            seed=seed,
            include_labels=include_labels
        )
        
        return frames, metadata
    
    def create_intervention(
        self,
        baseline_params: Union[Dict, VideoParams],
        intervention_params: Union[Dict, VideoParams],
        seed: Optional[int] = None,
        include_labels: bool = True
    ) -> InterventionResult:
        """
        Create an intervention pair: baseline vs intervention videos
        
        Args:
            baseline_params: Parameters for baseline video
            intervention_params: Parameters for intervention video
            seed: Random seed (same for both videos for fair comparison)
            include_labels: Whether to include visual labels
            
        Returns:
            InterventionResult object containing both videos and analysis
        """
        # Convert dicts to VideoParams if needed
        if isinstance(baseline_params, dict):
            baseline_params = VideoParams(**baseline_params)
        if isinstance(intervention_params, dict):
            intervention_params = VideoParams(**intervention_params)
            
        # Use provided seed or default
        if seed is None:
            seed = self.default_seed
            
        # Generate baseline video
        baseline_frames, baseline_meta = self.generate_video(
            params=baseline_params,
            seed=seed,
            include_labels=include_labels
        )
        
        # Generate intervention video (same seed for fair comparison)
        intervention_frames, intervention_meta = self.generate_video(
            params=intervention_params,
            seed=seed,
            include_labels=include_labels
        )
        
        # Compute effect magnitude and description
        effect_magnitude, effect_description = self._analyze_intervention_effect(
            baseline_meta, intervention_meta, baseline_params, intervention_params
        )
        
        return InterventionResult(
            baseline_frames=baseline_frames,
            intervention_frames=intervention_frames,
            baseline_metadata=baseline_meta,
            intervention_metadata=intervention_meta,
            effect_magnitude=effect_magnitude,
            effect_description=effect_description
        )
    
    def _analyze_intervention_effect(
        self, 
        baseline_meta: Dict, 
        intervention_meta: Dict,
        baseline_params: VideoParams,
        intervention_params: VideoParams
    ) -> Tuple[float, str]:
        """
        Analyze the effect of an intervention
        
        Returns:
            Tuple of (effect_magnitude, effect_description)
        """
        # Get final outcomes
        baseline_jam = baseline_meta.get('actual_jam_type', 'unknown')
        intervention_jam = intervention_meta.get('actual_jam_type', 'unknown')
        
        # Find changed parameters
        changed_params = []
        for attr in ['hole_diameter', 'wind_strength', 'num_circles', 'circle_size_min', 
                     'circle_size_max', 'spawn_rate', 'hole_x_position', 'wind_direction', 
                     'gravity', 'noise_level']:
            baseline_val = getattr(baseline_params, attr)
            intervention_val = getattr(intervention_params, attr)
            if baseline_val != intervention_val:
                changed_params.append(f"{attr}: {baseline_val} → {intervention_val}")
        
        # Determine effect magnitude
        if baseline_jam != intervention_jam:
            # Outcome changed
            effect_magnitude = 1.0
            effect_desc = f"Intervention changed outcome: {baseline_jam} → {intervention_jam}"
        else:
            # Same outcome, check quantitative measures
            baseline_exits = baseline_meta.get('final_stats', {}).get('total_exited', 0)
            intervention_exits = intervention_meta.get('final_stats', {}).get('total_exited', 0)
            
            if baseline_exits > 0:
                effect_magnitude = abs(intervention_exits - baseline_exits) / baseline_exits
            else:
                effect_magnitude = 0.0 if intervention_exits == 0 else 1.0
                
            effect_desc = f"Exit count change: {baseline_exits} → {intervention_exits}"
        
        if changed_params:
            effect_desc += f". Changed: {', '.join(changed_params)}"
            
        return effect_magnitude, effect_desc
    
    def save_frames_as_gif(
        self, 
        frames: List[np.ndarray], 
        output_path: str,
        duration: int = 100,
        loop: int = 0
    ) -> None:
        """
        Save video frames as an animated GIF
        
        Args:
            frames: List of numpy array frames
            output_path: Path to save GIF file
            duration: Duration between frames in milliseconds
            loop: Number of loops (0 = infinite)
        """
        # Convert numpy arrays to PIL Images
        pil_frames = [Image.fromarray(frame.astype(np.uint8)) for frame in frames]
        
        # Save as GIF
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration,
            loop=loop
        )
    
    def save_intervention_gifs(
        self, 
        result: InterventionResult, 
        baseline_path: str, 
        intervention_path: str,
        duration: int = 100
    ) -> None:
        """
        Save intervention result as two GIF files
        
        Args:
            result: InterventionResult object
            baseline_path: Path for baseline GIF
            intervention_path: Path for intervention GIF
            duration: Frame duration in milliseconds
        """
        self.save_frames_as_gif(result.baseline_frames, baseline_path, duration)
        self.save_frames_as_gif(result.intervention_frames, intervention_path, duration)
    
    @staticmethod
    def create_default_params() -> VideoParams:
        """
        Create default video parameters
        
        Returns:
            VideoParams object with sensible defaults
        """
        return VideoParams(
            width=400,
            height=400,
            num_frames=100,
            hole_diameter=40,
            wind_strength=2.0,
            num_circles=5,
            circle_size_min=8,
            circle_size_max=12,
            spawn_rate=0.3,
            hole_x_position=0.5,
            noise_level=0.0
        )
    
    @staticmethod
    def params_to_dict(params: VideoParams) -> Dict:
        """Convert VideoParams to dictionary"""
        return params.to_dict()
    
    @staticmethod
    def dict_to_params(params_dict: Dict) -> VideoParams:
        """Convert dictionary to VideoParams"""
        return VideoParams(**params_dict)


# Convenience function for quick usage
def generate_intervention_pair(
    baseline_params: Union[Dict, VideoParams],
    intervention_params: Union[Dict, VideoParams],
    seed: Optional[int] = None
) -> InterventionResult:
    """
    Convenience function to quickly generate an intervention pair
    
    Args:
        baseline_params: Baseline video parameters
        intervention_params: Intervention video parameters  
        seed: Random seed for reproducibility
        
    Returns:
        InterventionResult with both videos and analysis
    """
    interface = VideoInterface(default_seed=seed)
    return interface.create_intervention(baseline_params, intervention_params, seed)