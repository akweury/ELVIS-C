import cv2
import random
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

class TwoPartsEnv:
    def __init__(self, left_objects=5, right_objects=5, width=800, height=600, 
                 intervention_params=None):
        """
        Environment with two parts: left side objects move down, right side objects move up
        
        Args:
            left_objects: Number of objects on the left side
            right_objects: Number of objects on the right side
            width: Screen width
            height: Screen height
            intervention_params: Optional dict with intervention settings for reversed movement
        """
        self.width = width
        self.height = height
        self.left_objects = left_objects
        self.right_objects = right_objects
        
        # Intervention parameters
        self.intervention_params = intervention_params or {}
        self.is_intervention = self.intervention_params.get('is_intervention', False)
        self.left_intervention_indices = self.intervention_params.get('left_intervention_indices', [])
        self.right_intervention_indices = self.intervention_params.get('right_intervention_indices', [])
        
        # Colors (BGR format for OpenCV)
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (0, 0, 255)
        self.BLUE = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        
        # Object properties
        self.object_radius = 15
        self.velocity = 2
        
        # Entity-role mapping: consistent colors for each entity across videos
        self.entity_colors = self._generate_entity_color_mapping()
        
        self.reset()
    
    def _check_overlap(self, new_x: float, new_y: float, existing_objects: List[Dict]) -> bool:
        """
        Check if a new object at (new_x, new_y) would overlap with any existing objects
        
        Args:
            new_x: X coordinate of new object
            new_y: Y coordinate of new object
            existing_objects: List of existing objects to check against
            
        Returns:
            True if overlap detected, False otherwise
        """
        min_distance = 2 * self.object_radius + 2  # Add small buffer for safety
        
        for obj in existing_objects:
            distance = np.sqrt((new_x - obj['x'])**2 + (new_y - obj['y'])**2)
            if distance < min_distance:
                return True
        return False
    
    def _generate_entity_color_mapping(self) -> Dict[str, Tuple[int, int, int]]:
        """
        Generate consistent color mapping for entities across videos
        
        Returns:
            Dictionary mapping entity IDs to BGR colors
        """
        # Predefined distinct colors for consistency (BGR format)
        base_colors = [
            (100, 150, 255),  # Light orange
            (255, 150, 100),  # Light blue  
            (150, 255, 100),  # Light green
            (255, 100, 150),  # Light purple
            (150, 100, 255),  # Light red
            (100, 255, 150),  # Light cyan
            (200, 200, 100),  # Light yellow
            (150, 150, 255),  # Light pink
            (100, 200, 200),  # Light teal
            (200, 150, 150),  # Light brown
        ]
        
        entity_colors = {}
        color_idx = 0
        
        # Assign colors to left entities
        for i in range(self.left_objects):
            entity_id = f"left_{i+1}"
            entity_colors[entity_id] = base_colors[color_idx % len(base_colors)]
            color_idx += 1
            
        # Assign colors to right entities  
        for i in range(self.right_objects):
            entity_id = f"right_{i+1}"
            entity_colors[entity_id] = base_colors[color_idx % len(base_colors)]
            color_idx += 1
            
        return entity_colors
    
    def _generate_non_overlapping_position(self, side: str, existing_objects: List[Dict], 
                                          max_attempts: int = 100) -> Tuple[int, int]:
        """
        Generate a non-overlapping position for an object
        
        Args:
            side: 'left' or 'right' to determine the area
            existing_objects: List of existing objects to avoid
            max_attempts: Maximum number of attempts to find a position
            
        Returns:
            Tuple of (x, y) coordinates
        """
        for _ in range(max_attempts):
            if side == 'left':
                x = random.randint(self.object_radius, self.width // 2 - self.object_radius)
                y = random.randint(self.object_radius, self.height // 3)
            else:  # right side
                x = random.randint(self.width // 2 + self.object_radius, self.width - self.object_radius)
                y = random.randint(2 * self.height // 3, self.height - self.object_radius)
            
            if not self._check_overlap(x, y, existing_objects):
                return x, y
        
        # If we couldn't find a non-overlapping position, fallback to original random placement
        # This prevents infinite loops when the area is too crowded
        print(f"Warning: Could not find non-overlapping position for {side} side after {max_attempts} attempts")
        if side == 'left':
            x = random.randint(self.object_radius, self.width // 2 - self.object_radius)
            y = random.randint(self.object_radius, self.height // 3)
        else:
            x = random.randint(self.width // 2 + self.object_radius, self.width - self.object_radius)
            y = random.randint(2 * self.height // 3, self.height - self.object_radius)
        
        return x, y
    
    def reset(self):
        """Reset the environment to initial state"""
        self.objects = []
        
        # Create left side objects (moving down) with consistent entity IDs
        for i in range(self.left_objects):
            x, y = self._generate_non_overlapping_position('left', self.objects)
            entity_id = f"left_{i+1}"
            role = f"left_object_{i+1}"
            
            self.objects.append({
                'entity_id': entity_id,
                'role': role,
                'x': x,
                'y': y,
                'side': 'left',
                'color': self.entity_colors[entity_id],
                'velocity': self.velocity,
                'creation_order': i
            })
        
        # Create right side objects (moving up) with consistent entity IDs
        for i in range(self.right_objects):
            x, y = self._generate_non_overlapping_position('right', self.objects)
            entity_id = f"right_{i+1}"
            role = f"right_object_{i+1}"
            
            self.objects.append({
                'entity_id': entity_id,
                'role': role,
                'x': x,
                'y': y,
                'side': 'right',
                'color': self.entity_colors[entity_id],
                'velocity': self.velocity,
                'creation_order': i
            })
    
    def step(self):
        """Update object positions with intervention support"""
        for obj in self.objects:
            # Determine if this object should have reversed movement
            is_reversed = False
            if self.is_intervention:
                if obj['side'] == 'left' and obj['creation_order'] in self.left_intervention_indices:
                    is_reversed = True
                elif obj['side'] == 'right' and obj['creation_order'] in self.right_intervention_indices:
                    is_reversed = True
            
            if obj['side'] == 'left':
                if is_reversed:
                    # Intervention: left objects move up
                    obj['y'] -= obj['velocity']
                    # Reset to bottom if out of bounds
                    if obj['y'] < -self.object_radius:
                        obj['y'] = self.height + self.object_radius
                        other_objects = [o for o in self.objects if o != obj]
                        x, _ = self._generate_non_overlapping_position('left', other_objects, max_attempts=20)
                        obj['x'] = x
                else:
                    # Normal: left objects move down
                    obj['y'] += obj['velocity']
                    # Reset to top if out of bounds
                    if obj['y'] > self.height + self.object_radius:
                        obj['y'] = -self.object_radius
                        other_objects = [o for o in self.objects if o != obj]
                        x, _ = self._generate_non_overlapping_position('left', other_objects, max_attempts=20)
                        obj['x'] = x
            else:  # right side
                if is_reversed:
                    # Intervention: right objects move down
                    obj['y'] += obj['velocity']
                    # Reset to top if out of bounds
                    if obj['y'] > self.height + self.object_radius:
                        obj['y'] = -self.object_radius
                        other_objects = [o for o in self.objects if o != obj]
                        x, _ = self._generate_non_overlapping_position('right', other_objects, max_attempts=20)
                        obj['x'] = x
                else:
                    # Normal: right objects move up
                    obj['y'] -= obj['velocity']
                    # Reset to bottom if out of bounds
                    if obj['y'] < -self.object_radius:
                        obj['y'] = self.height + self.object_radius
                        other_objects = [o for o in self.objects if o != obj]
                        x, _ = self._generate_non_overlapping_position('right', other_objects, max_attempts=20)
                        obj['x'] = x
    
    def render(self, show_labels=True) -> np.ndarray:
        """
        Render the environment and return the frame as numpy array
        
        Args:
            show_labels: Whether to show text labels
            
        Returns:
            numpy array representing the rendered frame (BGR format)
        """
        # Create blank frame
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Draw dividing line only when visual guides are requested
        if show_labels:
            cv2.line(frame, (self.width // 2, 0), (self.width // 2, self.height), self.WHITE, 2)
        
        # Draw objects
        for obj in self.objects:
            center = (int(obj['x']), int(obj['y']))
            # Draw all objects the same way, without special borders for interventions
            cv2.circle(frame, center, self.object_radius, obj['color'], -1)
        
        # Draw labels if requested
        if show_labels:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6  # Smaller font size
            thickness = 2
            
            # Calculate text sizes for centering
            (text_width_down, text_height), _ = cv2.getTextSize("DOWN", font, font_scale, thickness)
            (text_width_up, text_height), _ = cv2.getTextSize("UP", font, font_scale, thickness)
            
            # Left side label - centered at top of left part
            left_center_x = (self.width // 4) - (text_width_down // 2)
            cv2.putText(frame, "DOWN", (left_center_x, 25), font, font_scale, self.WHITE, thickness)
            
            # Right side label - centered at top of right part
            right_center_x = (3 * self.width // 4) - (text_width_up // 2)
            cv2.putText(frame, "UP", (right_center_x, 25), font, font_scale, self.WHITE, thickness)
        
        return frame
    
    def generate_video(self, num_frames=100, fps=30, seed: Optional[int] = None, 
                      include_labels: bool = True) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Generate a video of the environment
        
        Args:
            num_frames: Number of frames to generate
            fps: Frames per second (for metadata)
            seed: Random seed for reproducible results (optional for this environment)
            include_labels: Whether to include visual labels
            
        Returns:
            Tuple of (frames, metadata)
        """
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Ensure a clean starting state for each video generation
        self.reset()
        
        frames = []
        frame_metadata = []
        
        for frame_idx in range(num_frames):
            # Update physics
            self.step()
            
            # Render frame
            frame = self.render(show_labels=include_labels)
            frames.append(frame)
            
            # Collect frame metadata with entity and role information
            frame_info = {
                'frame_idx': frame_idx,
                'left_objects': len([obj for obj in self.objects if obj['side'] == 'left']),
                'right_objects': len([obj for obj in self.objects if obj['side'] == 'right']),
                'total_objects': len(self.objects),
                'objects': [
                    {
                        'entity_id': obj['entity_id'],
                        'role': obj['role'],
                        'x': obj['x'],
                        'y': obj['y'],
                        'side': obj['side'],
                        'color': obj['color'],
                        'creation_order': obj['creation_order']
                    } for obj in self.objects
                ]
            }
            frame_metadata.append(frame_info)
        
        # Create overall metadata with entity mapping and intervention information
        metadata = {
            'num_frames': num_frames,
            'fps': fps,
            'resolution': (self.width, self.height),
            'left_objects': self.left_objects,
            'right_objects': self.right_objects,
            'object_radius': self.object_radius,
            'velocity': self.velocity,
            'is_intervention': self.is_intervention,
            'intervention_info': {
                'left_intervention_indices': self.left_intervention_indices,
                'right_intervention_indices': self.right_intervention_indices,
                'intervention_type': self.intervention_params.get('intervention_type', 'none')
            },
            'entity_mapping': {
                'colors': self.entity_colors,
                'entities': {
                    entity_id: {
                        'role': f"{side}_object_{i+1}",
                        'side': side,
                        'creation_order': i,
                        'has_intervention': (
                            (side == 'left' and i in self.left_intervention_indices) or
                            (side == 'right' and i in self.right_intervention_indices)
                        ) if self.is_intervention else False
                    }
                    for side in ['left', 'right']
                    for i in range(self.left_objects if side == 'left' else self.right_objects)
                    for entity_id in [f"{side}_{i+1}"]
                }
            },
            'frames': frame_metadata
        }
        
        return frames, metadata
    
    def save_video(self, filename: str, num_frames=100, fps=30, seed: Optional[int] = None, 
                   include_labels: bool = True):
        """
        Save video to file
        
        Args:
            filename: Output filename (should end with .mp4, .avi, etc.)
            num_frames: Number of frames to generate
            fps: Frames per second
            seed: Random seed for reproducible results
            include_labels: Whether to include visual labels
        """
        frames, metadata = self.generate_video(num_frames, fps, seed, include_labels)
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (self.width, self.height))
        
        # Write frames
        for frame in frames:
            out.write(frame)
        
        # Release everything
        out.release()
        print(f"Video saved to {filename}")
        
        # Also save metadata
        import json
        metadata_filename = filename.rsplit('.', 1)[0] + '_metadata.json'
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {metadata_filename}")
    
    def save_frames(self, output_dir: str, num_frames=100, seed: Optional[int] = None, 
                   include_labels: bool = False, format: str = 'png'):
        """
        Save individual frames to directory
        
        Args:
            output_dir: Output directory for frames
            num_frames: Number of frames to generate
            seed: Random seed for reproducible results
            include_labels: Whether to include visual labels
            format: Image format ('png', 'jpg', 'jpeg')
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        frames, metadata = self.generate_video(num_frames, seed=seed, include_labels=include_labels)
        
        # Save frames
        for i, frame in enumerate(frames):
            filename = os.path.join(output_dir, f"frame_{i:06d}.{format}")
            cv2.imwrite(filename, frame)
        
        print(f"Saved {len(frames)} frames to {output_dir}")
        
        # Also save metadata
        import json
        metadata_filename = os.path.join(output_dir, 'metadata.json')
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {metadata_filename}")
    
    def save_gif(self, filename: str, num_frames=100, fps=30, seed: Optional[int] = None, 
                 include_labels: bool = True, duration: Optional[int] = None):
        """
        Save animation as GIF file
        
        Args:
            filename: Output filename (should end with .gif)
            num_frames: Number of frames to generate
            fps: Frames per second (for calculating duration)
            seed: Random seed for reproducible results
            include_labels: Whether to include visual labels
            duration: Frame duration in milliseconds (overrides fps if provided)
        """
        frames, metadata = self.generate_video(num_frames, seed=seed, include_labels=include_labels)
        
        # Calculate duration if not provided
        if duration is None:
            duration = int(1000 / fps)  # Convert fps to milliseconds
        
        # Convert frames to PIL Images
        from PIL import Image
        pil_images = []
        for frame in frames:
            # Convert BGR to RGB for PIL
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            pil_images.append(pil_image)
        
        # Save as animated GIF
        if pil_images:
            pil_images[0].save(
                filename,
                save_all=True,
                append_images=pil_images[1:],
                duration=duration,
                loop=0
            )
            print(f"GIF saved to {filename}")
            
            # Also save metadata
            import json
            metadata_filename = filename.rsplit('.', 1)[0] + '_metadata.json'
            with open(metadata_filename, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Metadata saved to {metadata_filename}")
        else:
            print("No frames to save as GIF")
    
    def export_data(self, base_filename: str, num_frames=100, seed: Optional[int] = None,
                   include_labels: bool = True, formats: List[str] = ['frames', 'gif', 'video']):
        """
        Export simulation data in multiple formats
        
        Args:
            base_filename: Base name for output files (without extension)
            num_frames: Number of frames to generate
            seed: Random seed for reproducible results  
            include_labels: Whether to include visual labels
            formats: List of formats to export ('frames', 'gif', 'video', 'metadata')
        """
        frames, metadata = self.generate_video(num_frames, seed=seed, include_labels=include_labels)
        
        exported = []
        
        if 'frames' in formats:
            frames_dir = f"{base_filename}_frames"
            self.save_frames(frames_dir, num_frames, seed, include_labels)
            exported.append(f"Frames: {frames_dir}/")
            
        if 'gif' in formats:
            gif_filename = f"{base_filename}.gif"
            self.save_gif(gif_filename, num_frames, seed=seed, include_labels=include_labels)
            exported.append(f"GIF: {gif_filename}")
            
        if 'video' in formats:
            video_filename = f"{base_filename}.mp4"
            self.save_video(video_filename, num_frames, seed=seed, include_labels=include_labels)
            exported.append(f"Video: {video_filename}")
            
        if 'metadata' in formats and 'metadata' not in ['frames', 'gif', 'video']:
            # Save standalone metadata if not already saved by other formats
            import json
            metadata_filename = f"{base_filename}_metadata.json"
            with open(metadata_filename, 'w') as f:
                json.dump(metadata, f, indent=2)
            exported.append(f"Metadata: {metadata_filename}")
        
        print(f"✅ Exported data in {len(exported)} formats:")
        for item in exported:
            print(f"   - {item}")
        
        return frames, metadata
    
    def run_interactive(self, fps=30, include_labels: bool = True):
        """
        Run the environment interactively with OpenCV window
        
        Args:
            fps: Frames per second for display
            include_labels: Whether to show labels
        """
        print("Interactive mode - Press 'q' to quit, 'r' to reset")
        
        while True:
            # Update physics
            self.step()
            
            # Render frame
            frame = self.render(show_labels=include_labels)
            
            # Display frame
            cv2.imshow('Two Parts Environment', frame)
            
            # Handle key presses
            key = cv2.waitKey(1000 // fps) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.reset()
                print("Environment reset")
        
        cv2.destroyAllWindows()
    
    def get_state(self) -> List[Tuple[int, int, str]]:
        """Get current state of all objects"""
        return [(obj['x'], obj['y'], obj['side']) for obj in self.objects]

if __name__ == "__main__":
    # Create environment
    env = TwoPartsEnv(left_objects=3, right_objects=4)
    
    # Example 1: Run interactively
    print("Running interactive mode...")
    print("Controls:")
    print("  'q' - Quit")
    print("  'r' - Reset environment")
    env.run_interactive(fps=30)
    
    # Example 2: Generate and save video
    print("\nGenerating video...")
    env.reset()  # Reset to clean state
    env.save_video("two_parts_demo.mp4", num_frames=300, fps=30)
    
    # Example 3: Generate frames for analysis
    print("\nGenerating frames for analysis...")
    env.reset()
    frames, metadata = env.generate_video(num_frames=60, fps=30)
    
    print(f"Generated {len(frames)} frames")
    print(f"Frame shape: {frames[0].shape}")
    print(f"Left objects: {metadata['left_objects']}")
    print(f"Right objects: {metadata['right_objects']}")
    
    # Show final frame
    cv2.imshow('Final Frame', frames[-1])
    cv2.waitKey(2000)  # Show for 2 seconds
    cv2.destroyAllWindows()