"""
AI Model Interface for ELVIS-C

Simple interface designed for AI models that:
1. Perceive parameters from existing videos
2. Specify desired interventions 
3. Generate intervention videos with those exact parameters

This interface focuses on video generation given specific parameters,
not on providing defaults or parameter exploration.
"""

import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.falling_circles_env import VideoParams, generate_falling_circles_video, FallingCirclesEnvironment


@dataclass
class VideoGenerationResult:
    """
    Result from video generation
    """
    frames: List[np.ndarray]
    metadata: Dict
    parameters: Dict
    success: bool
    message: str = ""


@dataclass 
class InterventionResult:
    """
    Result from an intervention request
    """
    baseline_result: VideoGenerationResult
    intervention_result: VideoGenerationResult
    parameters_changed: Dict[str, Tuple]  # param_name -> (old_value, new_value)
    effect_description: str
    success: bool


class AIVideoInterface:
    """
    Simple interface for AI models to generate videos with specific parameters
    
    Workflow:
    1. AI perceives parameters from existing video
    2. AI specifies baseline parameters and desired intervention
    3. Interface generates both baseline and intervention videos
    4. AI receives videos for analysis
    """
    
    def __init__(self):
        """Initialize the AI video interface"""
        pass
    
    def generate_video(
        self,
        parameters: Dict,
        seed: Optional[int] = None,
        include_labels: bool = False
    ) -> VideoGenerationResult:
        """
        Generate a single video with specified parameters
        
        Args:
            parameters: Dictionary of video parameters
            seed: Random seed for reproducibility
            include_labels: Whether to include visual labels
            
        Returns:
            VideoGenerationResult with frames and metadata
            
        Example:
            params = {
                "hole_diameter": 40,
                "wind_strength": 2.0,
                "num_circles": 5,
                "width": 400,
                "height": 400,
                "num_frames": 100
            }
            result = interface.generate_video(params, seed=42)
        """
        try:
            # Validate and create VideoParams
            video_params = self._dict_to_video_params(parameters)
            
            # Generate video
            frames, metadata = generate_falling_circles_video(
                params=video_params,
                seed=seed,
                include_labels=include_labels
            )
            
            return VideoGenerationResult(
                frames=frames,
                metadata=metadata,
                parameters=parameters.copy(),
                success=True,
                message="Video generated successfully"
            )
            
        except Exception as e:
            return VideoGenerationResult(
                frames=[],
                metadata={},
                parameters=parameters.copy(),
                success=False,
                message=f"Error generating video: {str(e)}"
            )
    
    def create_intervention_pair(
        self,
        baseline_parameters: Dict,
        intervention_parameters: Dict,
        seed: Optional[int] = None,
        include_labels: bool = False
    ) -> InterventionResult:
        """
        Generate intervention pair: baseline and intervention videos
        
        Args:
            baseline_parameters: Parameters for baseline video (as perceived by AI)
            intervention_parameters: Parameters for intervention video
            seed: Random seed (same for both videos for fair comparison)
            include_labels: Whether to include visual labels
            
        Returns:
            InterventionResult with both videos and analysis
            
        Example:
            # AI perceived these parameters from original video
            baseline = {
                "hole_diameter": 40,
                "wind_strength": 2.0,
                "num_circles": 5,
                "circle_size_min": 8,
                "circle_size_max": 12,
                "width": 400,
                "height": 400,
                "num_frames": 100
            }
            
            # AI wants to intervene by increasing hole size
            intervention = baseline.copy()
            intervention["hole_diameter"] = 80
            
            result = interface.create_intervention_pair(baseline, intervention, seed=42)
        """
        # Generate baseline video
        baseline_result = self.generate_video(
            parameters=baseline_parameters,
            seed=seed,
            include_labels=include_labels
        )
        
        # Generate intervention video (same seed for comparison)
        intervention_result = self.generate_video(
            parameters=intervention_parameters,
            seed=seed,
            include_labels=include_labels
        )
        
        # Analyze what changed
        parameters_changed = {}
        for param, baseline_value in baseline_parameters.items():
            intervention_value = intervention_parameters.get(param)
            if intervention_value is not None and baseline_value != intervention_value:
                parameters_changed[param] = (baseline_value, intervention_value)
        
        # Create effect description
        if parameters_changed:
            changes = [f"{param}: {old} → {new}" for param, (old, new) in parameters_changed.items()]
            effect_description = f"Changed parameters: {', '.join(changes)}"
        else:
            effect_description = "No parameters changed"
        
        # Add outcome comparison if available
        if baseline_result.success and intervention_result.success:
            baseline_outcome = baseline_result.metadata.get('actual_jam_type', 'unknown')
            intervention_outcome = intervention_result.metadata.get('actual_jam_type', 'unknown')
            if baseline_outcome != intervention_outcome:
                effect_description += f". Outcome: {baseline_outcome} → {intervention_outcome}"
        
        success = baseline_result.success and intervention_result.success
        
        return InterventionResult(
            baseline_result=baseline_result,
            intervention_result=intervention_result,
            parameters_changed=parameters_changed,
            effect_description=effect_description,
            success=success
        )
    
    def quick_intervention(
        self,
        baseline_parameters: Dict,
        intervention_target: str,
        intervention_value: Union[float, int],
        seed: Optional[int] = None,
        include_labels: bool = False
    ) -> InterventionResult:
        """
        Quick intervention on a single parameter
        
        Args:
            baseline_parameters: Base parameters perceived by AI
            intervention_target: Parameter to change
            intervention_value: New value for the parameter
            seed: Random seed
            include_labels: Whether to include labels
            
        Returns:
            InterventionResult
            
        Example:
            # AI perceived these parameters
            baseline = {"hole_diameter": 40, "wind_strength": 2.0, ...}
            
            # AI wants to double the hole size
            result = interface.quick_intervention(
                baseline_parameters=baseline,
                intervention_target="hole_diameter", 
                intervention_value=80,
                seed=42
            )
        """
        # Create intervention parameters
        intervention_parameters = baseline_parameters.copy()
        intervention_parameters[intervention_target] = intervention_value
        
        return self.create_intervention_pair(
            baseline_parameters=baseline_parameters,
            intervention_parameters=intervention_parameters,
            seed=seed,
            include_labels=include_labels
        )
    
    def save_videos(
        self, 
        result: Union[VideoGenerationResult, InterventionResult], 
        output_dir: str = ".",
        prefix: str = "video"
    ) -> Dict[str, str]:
        """
        Save video frames as GIF files
        
        Args:
            result: VideoGenerationResult or InterventionResult
            output_dir: Directory to save files
            prefix: Filename prefix
            
        Returns:
            Dictionary mapping video type to saved file path
            
        Example:
            files = interface.save_videos(result, output_dir="./outputs", prefix="experiment1")
            # Returns: {"baseline": "./outputs/experiment1_baseline.gif", 
            #          "intervention": "./outputs/experiment1_intervention.gif"}
        """
        from PIL import Image
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}
        
        if isinstance(result, VideoGenerationResult):
            # Single video
            if result.success and result.frames:
                filepath = os.path.join(output_dir, f"{prefix}.gif")
                self._save_frames_as_gif(result.frames, filepath)
                saved_files["video"] = filepath
                
        elif isinstance(result, InterventionResult):
            # Intervention pair
            if result.baseline_result.success and result.baseline_result.frames:
                baseline_path = os.path.join(output_dir, f"{prefix}_baseline.gif")
                self._save_frames_as_gif(result.baseline_result.frames, baseline_path)
                saved_files["baseline"] = baseline_path
                
            if result.intervention_result.success and result.intervention_result.frames:
                intervention_path = os.path.join(output_dir, f"{prefix}_intervention.gif") 
                self._save_frames_as_gif(result.intervention_result.frames, intervention_path)
                saved_files["intervention"] = intervention_path
        
        return saved_files
    
    def _save_frames_as_gif(
        self, 
        frames: List[np.ndarray], 
        output_path: str,
        duration: int = 100
    ) -> None:
        """Save frames as animated GIF"""
        from PIL import Image
        
        # Convert numpy arrays to PIL Images
        pil_frames = [Image.fromarray(frame.astype(np.uint8)) for frame in frames]
        
        # Save as GIF
        if pil_frames:
            pil_frames[0].save(
                output_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=duration,
                loop=0
            )
    
    def _dict_to_video_params(self, params_dict: Dict) -> VideoParams:
        """
        Convert parameter dictionary to VideoParams object
        
        Provides defaults for missing parameters to ensure valid video generation
        """
        # Default values based on VideoParams class
        defaults = {
            "num_frames": 60,
            "width": 224, 
            "height": 224,
            "num_circles": 15,
            "circle_size_min": 4,
            "circle_size_max": 8,
            "hole_diameter": 20,
            "hole_x_position": 0.5,
            "wind_strength": 0.0,
            "wind_direction": 1,
            "gravity": 0.8,
            "spawn_rate": 0.3,
            "circle_color": None,
            "background_color": None,
            "noise_level": 0.0
        }
        
        # Merge provided parameters with defaults
        final_params = defaults.copy()
        final_params.update(params_dict)
        
        # Create VideoParams object
        return VideoParams(**final_params)
    
    def get_parameter_template(self) -> Dict:
        """
        Get a template of all available parameters with their default values
        
        Returns:
            Dictionary with parameter names and default values
            
        This helps AI models understand what parameters are available
        and their expected ranges/types.
        """
        return {
            "num_frames": 60,          # Number of frames to generate
            "width": 224,              # Video width in pixels
            "height": 224,             # Video height in pixels
            "num_circles": 15,         # Maximum number of circles
            "circle_size_min": 4,      # Minimum circle radius
            "circle_size_max": 8,      # Maximum circle radius
            "hole_diameter": 20,       # Size of exit hole
            "hole_x_position": 0.5,    # Hole position (0.0=left, 1.0=right)
            "wind_strength": 0.0,      # Wind force (pixels per frame)
            "wind_direction": 1,       # Wind direction (1=right, -1=left)
            "gravity": 0.8,            # Gravitational acceleration
            "spawn_rate": 0.3,         # Probability of spawning per frame
            "noise_level": 0.0         # Visual noise level (0.0-0.2)
        }
    
    def validate_parameters(self, parameters: Dict) -> Tuple[bool, str]:
        """
        Validate parameter dictionary
        
        Args:
            parameters: Parameter dictionary to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Try to create VideoParams - this will catch type/value errors
            self._dict_to_video_params(parameters)
            return True, "Parameters are valid"
        except Exception as e:
            return False, f"Invalid parameters: {str(e)}"


# Convenience function for quick access
def generate_intervention_video(
    baseline_parameters: Dict,
    intervention_target: str,
    intervention_value: Union[float, int],
    seed: Optional[int] = None,
    include_labels: bool = False
) -> InterventionResult:
    """
    Convenience function for AI models to quickly generate intervention videos
    
    Args:
        baseline_parameters: Parameters perceived by AI from original video
        intervention_target: Parameter to intervene on
        intervention_value: New value for intervention
        seed: Random seed for reproducibility
        
    Returns:
        InterventionResult with baseline and intervention videos
        
    Example:
        # AI perceived parameters from video
        baseline_params = {
            "hole_diameter": 40,
            "wind_strength": 2.0,
            "num_circles": 5,
            "width": 400,
            "height": 400
        }
        
        # Generate intervention video with larger hole
        result = generate_intervention_video(
            baseline_parameters=baseline_params,
            intervention_target="hole_diameter",
            intervention_value=80,
            seed=42
        )
        
        # Access the videos
        baseline_video = result.baseline_result.frames
        intervention_video = result.intervention_result.frames
    """
    interface = AIVideoInterface()
    return interface.quick_intervention(
        baseline_parameters=baseline_parameters,
        intervention_target=intervention_target,
        intervention_value=intervention_value,
        seed=seed,
        include_labels=include_labels
    )