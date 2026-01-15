#!/usr/bin/env python3
"""
Extensible Two Parts Environment
Uses rule-based system for easy modifications
"""

import cv2
import random
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from rule_system.config_system import VideoConfig


class ExtensibleTwoPartsEnv:
    """Extended TwoPartsEnv that uses configuration-driven rules"""
    
    def __init__(self, config: Optional[VideoConfig] = None, **kwargs):
        """
        Initialize environment with configuration
        
        Args:
            config: VideoConfig object for rule-based generation
            **kwargs: Override parameters (width, height, etc.)
        """
        self.config = config or VideoConfig()
        
        # Get parameters from config, with kwargs overrides
        video_params = self.config.get_video_params()
        object_params = self.config.get_object_params()
        
        self.width = kwargs.get('width', video_params.get('width', 400))
        self.height = kwargs.get('height', video_params.get('height', 300))
        self.left_objects = kwargs.get('left_objects', object_params.get('left_objects', 5))
        self.right_objects = kwargs.get('right_objects', object_params.get('right_objects', 5))
        self.object_radius = kwargs.get('object_radius', object_params.get('radius', 15))
        self.velocity = kwargs.get('velocity', object_params.get('velocity', 2))
        
        # Get rule engine from configuration
        self.rule_engine = self.config.get_rule_engine()
        
        # Base colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        
        self.reset()
    
    def reset(self):
        """Reset environment and apply initial rules"""
        self.objects = []
        self.frame_count = 0
        
        # Create left side objects
        for i in range(self.left_objects):
            x, y = self._generate_non_overlapping_position('left', self.objects)
            entity_id = f"left_{i+1}"
            role = f"left_object_{i+1}"
            
            obj = {
                'entity_id': entity_id,
                'role': role,
                'x': x,
                'y': y,
                'side': 'left',
                'color': None,  # Will be set by color rule
                'velocity': self.velocity,
                'creation_order': i,
                'intervention_reverse': False
            }
            self.objects.append(obj)
        
        # Create right side objects
        for i in range(self.right_objects):
            x, y = self._generate_non_overlapping_position('right', self.objects)
            entity_id = f"right_{i+1}"
            role = f"right_object_{i+1}"
            
            obj = {
                'entity_id': entity_id,
                'role': role,
                'x': x,
                'y': y,
                'side': 'right',
                'color': None,  # Will be set by color rule
                'velocity': self.velocity,
                'creation_order': i,
                'intervention_reverse': False
            }
            self.objects.append(obj)
        
        # Apply initial rules (colors, placement, etc.)
        frame_info = {'frame_count': self.frame_count, 'width': self.width, 'height': self.height}
        self.objects = self.rule_engine.apply_all_rules(self.objects, frame_info)
        
        # Handle cross-placement repositioning after rules are applied
        self._apply_cross_placements()
    
    def step(self):
        """Update physics with rule-based modifications"""
        self.frame_count += 1
        frame_info = {'frame_count': self.frame_count, 'width': self.width, 'height': self.height}
        
        # Apply rules before physics update
        self.objects = self.rule_engine.apply_all_rules(self.objects, frame_info)
        
        # Update positions based on rules and interventions
        for obj in self.objects:
            self._update_object_position(obj)
    
    def _update_object_position(self, obj: Dict):
        """Update object position based on its side, interventions, and cross-placement"""
        # Determine effective side and movement based on cross-placement
        effective_side = obj.get('side', 'left')
        cross_movement = obj.get('cross_movement')
        
        # Use cross_movement if object is cross-placed, otherwise use side-based movement
        if cross_movement:
            if cross_movement == 'up':
                obj['y'] -= obj['velocity']
                if obj['y'] < -self.object_radius:
                    obj['y'] = self.height + self.object_radius
                    self._reposition_object_horizontally(obj, effective_side)
            elif cross_movement == 'down':
                obj['y'] += obj['velocity']
                if obj['y'] > self.height + self.object_radius:
                    obj['y'] = -self.object_radius
                    self._reposition_object_horizontally(obj, effective_side)
            return
        
        # Standard movement logic with intervention support
        if effective_side == 'left':
            if obj.get('intervention_reverse', False):
                # Intervention: left objects move up
                obj['y'] -= obj['velocity']
                if obj['y'] < -self.object_radius:
                    obj['y'] = self.height + self.object_radius
                    self._reposition_object_horizontally(obj, 'left')
            else:
                # Normal: left objects move down
                obj['y'] += obj['velocity']
                if obj['y'] > self.height + self.object_radius:
                    obj['y'] = -self.object_radius
                    self._reposition_object_horizontally(obj, 'left')
        else:  # right side
            if obj.get('intervention_reverse', False):
                # Intervention: right objects move down
                obj['y'] += obj['velocity']
                if obj['y'] > self.height + self.object_radius:
                    obj['y'] = -self.object_radius
                    self._reposition_object_horizontally(obj, 'right')
            else:
                # Normal: right objects move up
                obj['y'] -= obj['velocity']
                if obj['y'] < -self.object_radius:
                    obj['y'] = self.height + self.object_radius
                    self._reposition_object_horizontally(obj, 'right')
    
    def _reposition_object_horizontally(self, obj: Dict, side: str):
        """Reposition object horizontally when it wraps around"""
        other_objects = [o for o in self.objects if o != obj]
        x, _ = self._generate_non_overlapping_position(side, other_objects, max_attempts=20)
        obj['x'] = x
    
    def _check_overlap(self, new_x: float, new_y: float, existing_objects: List[Dict]) -> bool:
        """Check for overlap with existing objects"""
        min_distance = 2 * self.object_radius + 2
        
        for obj in existing_objects:
            distance = np.sqrt((new_x - obj['x'])**2 + (new_y - obj['y'])**2)
            if distance < min_distance:
                return True
        return False
    
    def _generate_non_overlapping_position(self, side: str, existing_objects: List[Dict], 
                                          max_attempts: int = 100) -> Tuple[int, int]:
        """Generate non-overlapping position for object"""
        for _ in range(max_attempts):
            if side == 'left':
                x = random.randint(self.object_radius, self.width // 2 - self.object_radius)
                y = random.randint(self.object_radius, self.height // 3)
            else:  # right side
                x = random.randint(self.width // 2 + self.object_radius, self.width - self.object_radius)
                y = random.randint(2 * self.height // 3, self.height - self.object_radius)
            
            if not self._check_overlap(x, y, existing_objects):
                return x, y
        
        # Fallback if no non-overlapping position found
        if side == 'left':
            x = random.randint(self.object_radius, self.width // 2 - self.object_radius)
            y = random.randint(self.object_radius, self.height // 3)
        else:
            x = random.randint(self.width // 2 + self.object_radius, self.width - self.object_radius)
            y = random.randint(2 * self.height // 3, self.height - self.object_radius)
        
        return x, y
    
    def _apply_cross_placements(self):
        """Reposition cross-placed objects to their target sides"""
        for obj in self.objects:
            if obj.get('cross_placed', False):
                target_side = obj.get('side')
                other_objects = [o for o in self.objects if o != obj]
                
                # Generate new position on target side
                new_x, new_y = self._generate_non_overlapping_position(target_side, other_objects)
                obj['x'] = new_x
                obj['y'] = new_y
    
    def render(self, show_labels=True) -> np.ndarray:
        """Render environment with rule-applied modifications"""
        # Create blank frame
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Draw dividing line if labels requested
        if show_labels:
            cv2.line(frame, (self.width // 2, 0), (self.width // 2, self.height), self.WHITE, 2)
        
        # Draw objects with their rule-determined colors
        for obj in self.objects:
            center = (int(obj['x']), int(obj['y']))
            color = obj.get('color', self.WHITE)
            cv2.circle(frame, center, self.object_radius, color, -1)
        
        # Draw labels if requested
        if show_labels:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Left side label
            (text_width_down, text_height), _ = cv2.getTextSize("DOWN", font, font_scale, thickness)
            left_center_x = (self.width // 4) - (text_width_down // 2)
            cv2.putText(frame, "DOWN", (left_center_x, 25), font, font_scale, self.WHITE, thickness)
            
            # Right side label
            (text_width_up, text_height), _ = cv2.getTextSize("UP", font, font_scale, thickness)
            right_center_x = (3 * self.width // 4) - (text_width_up // 2)
            cv2.putText(frame, "UP", (right_center_x, 25), font, font_scale, self.WHITE, thickness)
        
        return frame
    
    def generate_video(self, num_frames=100, fps=30, seed: Optional[int] = None, 
                      include_labels: bool = True) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Generate video with configuration-driven rules"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.reset()
        
        frames = []
        frame_metadata = []
        
        for frame_idx in range(num_frames):
            self.step()
            frame = self.render(show_labels=include_labels)
            frames.append(frame)
            
            # Collect frame metadata
            frame_metadata.append({
                'frame_index': frame_idx,
                'objects': [
                    {
                        'entity_id': obj['entity_id'],
                        'x': obj['x'],
                        'y': obj['y'],
                        'color': obj['color'],
                        'intervention_active': obj.get('intervention_reverse', False)
                    }
                    for obj in self.objects
                ]
            })
        
        metadata = {
            'config': self.config.config,
            'video_stats': {
                'num_frames': len(frames),
                'fps': fps,
                'resolution': [self.width, self.height]
            },
            'frame_data': frame_metadata
        }
        
        return frames, metadata