"""
Falling Circles Environment Core

Extracted physics simulation engine that can be used by any script
to generate falling circles videos with consistent mechanics.

This module contains:
- VideoParams: Parameter configuration class
- Circle: Physics object for individual circles  
- FallingCirclesEnvironment: Core simulation engine
- generate_falling_circles_video: Main generation function (for compatibility)

Usage:
    from falling_circles_env import FallingCirclesEnvironment, VideoParams
    
    params = VideoParams(hole_diameter=40, wind_strength=2.0, num_circles=8)
    env = FallingCirclesEnvironment(params)
    frames, metadata = env.generate_video(seed=42)
"""

import os
import random
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Optional, Any


class VideoParams:
    """Parameters controlling falling circles video generation"""
    def __init__(self, 
                 num_frames=60,
                 width=224,
                 height=224,
                 num_circles=15,
                 circle_size_min=4,
                 circle_size_max=8,
                 hole_diameter=20,
                 hole_x_position=0.5,  # fraction of width
                 wind_strength=0.0,    # pixels per frame
                 wind_direction=1,     # 1 for left->right, -1 for right->left
                 gravity=0.8,          # pixels per frame^2
                 spawn_rate=0.3,       # probability of spawning per frame
                 circle_color=None,
                 background_color=None,
                 noise_level=0.0):
        self.num_frames = num_frames
        self.width = width
        self.height = height
        self.num_circles = num_circles
        self.circle_size_min = circle_size_min
        self.circle_size_max = circle_size_max
        self.hole_diameter = hole_diameter
        self.hole_x_position = hole_x_position
        self.wind_strength = wind_strength
        self.wind_direction = wind_direction
        self.gravity = gravity
        self.spawn_rate = spawn_rate
        self.circle_color = circle_color or [200, 200, 255]  # light blue default
        self.background_color = background_color or [50, 50, 50]  # dark gray default
        self.noise_level = noise_level
    
    def to_dict(self):
        return {
            'num_frames': self.num_frames,
            'width': self.width,
            'height': self.height,
            'num_circles': self.num_circles,
            'circle_size_min': self.circle_size_min,
            'circle_size_max': self.circle_size_max,
            'hole_diameter': self.hole_diameter,
            'hole_x_position': self.hole_x_position,
            'wind_strength': self.wind_strength,
            'wind_direction': self.wind_direction,
            'gravity': self.gravity,
            'spawn_rate': self.spawn_rate,
            'circle_color': self.circle_color,
            'background_color': self.background_color,
            'noise_level': self.noise_level
        }


class Circle:
    """Represents a single falling circle with physics properties"""
    def __init__(self, x, y, size, color, circle_id):
        self.x = float(x)
        self.y = float(y)
        self.vx = 0.0
        self.vy = 0.0
        self.size = size
        self.color = color
        self.id = circle_id
        self.active = True
        self.exited = False
        self.stuck_counter = 0


class FallingCirclesEnvironment:
    """
    Core physics simulation environment for falling circles
    
    This class encapsulates all the physics logic and rendering,
    providing a clean interface for generating videos with consistent mechanics.
    """
    
    def __init__(self, params: VideoParams):
        """
        Initialize the environment with given parameters
        
        Args:
            params: VideoParams object containing simulation settings
        """
        self.params = params
        self.reset()
    
    def reset(self):
        """Reset the environment to initial state"""
        self.circles = []
        self.next_circle_id = 0
        self.spawn_timer = 0.0
        self.exit_count = 0
        self.frames_since_last_exit = 0
        
        # Calculate hole position
        self.hole_x = int(self.params.hole_x_position * self.params.width)
        self.hole_y = self.params.height - 10
    
    def generate_video(self, seed: Optional[int] = None, include_labels: bool = False, 
                      actual_jam_type: Optional[str] = None) -> Tuple[List[np.ndarray], Dict]:
        """
        Generate a complete video simulation
        
        Args:
            seed: Random seed for reproducible results
            include_labels: Whether to include visual labels on frames
            actual_jam_type: Override jam type label (for display only)
            
        Returns:
            Tuple of (frames, metadata)
            - frames: List of numpy arrays representing video frames
            - metadata: Dictionary with simulation details and statistics
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.reset()
        
        frames = []
        frame_metadata = []
        
        for frame_idx in range(self.params.num_frames):
            # Update physics
            self._update_physics()
            
            # Render frame
            frame_img = self._render_frame(frame_idx, include_labels, actual_jam_type)
            
            # Convert to numpy array and store
            frames.append(np.array(frame_img))
            
            # Collect frame metadata
            active_circles = [c for c in self.circles if c.active]
            visible_circles = self._get_visible_circles(active_circles)
            
            frame_meta = {
                'frame_idx': frame_idx,
                'num_active_circles': len(active_circles),
                'num_visible_circles': len(visible_circles),
                'circles_exited': self.exit_count,
                'circles_stuck': sum(1 for c in active_circles if c.stuck_counter > 10),
                'frames_since_last_exit': self.frames_since_last_exit,
                'hole_position': [self.hole_x, self.hole_y],
                'wind_effect': self.params.wind_strength * self.params.wind_direction,
                'circles': visible_circles
            }
            frame_metadata.append(frame_meta)
            
            self.frames_since_last_exit += 1
        
        # Determine final jam type
        actual_jam_type = self._determine_jam_type()
        
        # Compile complete metadata
        metadata = {
            'params': self.params.to_dict(),
            'seed': seed,
            'frames': frame_metadata,
            'actual_jam_type': actual_jam_type,
            'exit_statistics': {
                'total_spawned': len(self.circles),
                'total_exited': self.exit_count,
                'exit_ratio': self.exit_count / max(len(self.circles), 1),
                'final_stuck_count': sum(1 for c in self.circles if c.stuck_counter > 5)
            }
        }
        
        return frames, metadata
    
    def _update_physics(self):
        """Update physics for one simulation step"""
        # Spawn new circles
        self._spawn_circles()
        
        # Update existing circles
        active_circles = [c for c in self.circles if c.active]
        
        for circle in active_circles:
            self._update_circle_physics(circle, active_circles)
    
    def _spawn_circles(self):
        """Handle circle spawning logic"""
        # Stop spawning new circles once any circle has exited
        if self.exit_count > 0:
            return
            
        self.spawn_timer += self.params.spawn_rate
        
        while self.spawn_timer >= 1.0 and len(self.circles) < self.params.num_circles:
            spawn_x = random.uniform(self.params.circle_size_max, 
                                   self.params.width - self.params.circle_size_max)
            spawn_y = random.uniform(-20, -5)
            size = random.randint(self.params.circle_size_min, self.params.circle_size_max)
            
            new_circle = Circle(spawn_x, spawn_y, size, self.params.circle_color.copy(), 
                              self.next_circle_id)
            self.circles.append(new_circle)
            self.next_circle_id += 1
            self.spawn_timer -= 1.0
    
    def _update_circle_physics(self, circle: Circle, all_circles: List[Circle]):
        """Update physics for a single circle"""
        # Apply gravity
        circle.vy += self.params.gravity
        
        # Apply wind
        circle.vx += self.params.wind_strength * self.params.wind_direction
        
        # Apply drag
        circle.vx *= 0.99
        circle.vy *= 0.999
        
        # Calculate new position
        new_x = circle.x + circle.vx
        new_y = circle.y + circle.vy
        
        # Handle wall collisions
        new_x, circle.vx = self._handle_wall_collision(new_x, circle.vx, circle.size)
        
        # Handle funnel collision and exit
        new_y, circle.vy = self._handle_funnel_collision(circle, new_x, new_y)
        
        # Handle circle-circle collisions
        new_x, new_y = self._handle_circle_collisions(circle, new_x, new_y, all_circles)
        
        # Update circle position
        circle.x = new_x
        circle.y = new_y
    
    def _handle_wall_collision(self, new_x: float, vx: float, size: int) -> Tuple[float, float]:
        """Handle collision with side walls"""
        if new_x - size < 0:
            new_x = size
            vx = abs(vx) * 0.3
        elif new_x + size > self.params.width:
            new_x = self.params.width - size
            vx = -abs(vx) * 0.3
        return new_x, vx
    
    def _handle_funnel_collision(self, circle: Circle, new_x: float, new_y: float) -> Tuple[float, float]:
        """Handle collision with funnel surface and exit logic"""
        barrier_y = self.params.height - 5
        funnel_center_x = self.hole_x
        funnel_width = self.params.hole_diameter
        funnel_depth = 40
        
        # Calculate funnel surface height at circle's x position
        if new_x <= funnel_center_x:
            # Left side of funnel
            progress = new_x / funnel_center_x
            funnel_surface_y = (barrier_y - funnel_depth) + progress * funnel_depth
        else:
            # Right side of funnel
            progress = (self.params.width - new_x) / (self.params.width - funnel_center_x)
            funnel_surface_y = (barrier_y - funnel_depth) + progress * funnel_depth
        
        # Check if circle hits the funnel surface
        if new_y + circle.size >= funnel_surface_y:
            # Check exit conditions
            if self._can_circle_exit(circle, new_x, new_y, funnel_center_x, funnel_width):
                # Circle exits
                circle.active = False
                circle.exited = True
                self.exit_count += 1
                self.frames_since_last_exit = 0
            else:
                # Collision with funnel surface - implement slope physics
                new_y = funnel_surface_y - circle.size
                circle.vy = self._apply_funnel_physics(circle, new_x, funnel_center_x)
        
        return new_y, circle.vy
    
    def _can_circle_exit(self, circle: Circle, x: float, y: float, 
                        hole_center: float, hole_width: int) -> bool:
        """Check if a circle can exit through the hole"""
        hole_left = hole_center - hole_width // 2
        hole_right = hole_center + hole_width // 2
        barrier_y = self.params.height - 5
        
        # Multiple exit conditions
        hole_center_distance = abs(x - hole_center)
        circle_diameter = circle.size * 2
        
        # Condition 1: Circle center is well within hole
        center_within_hole = hole_center_distance < (hole_width // 2 - circle.size)
        
        # Condition 2: Circle overlaps significantly with hole opening
        circle_left = x - circle.size
        circle_right = x + circle.size
        overlap_left = max(circle_left, hole_left)
        overlap_right = min(circle_right, hole_right)
        overlap_width = max(0, overlap_right - overlap_left)
        significant_overlap = overlap_width > circle_diameter * 0.6
        
        # Allow exit if conditions are met
        return ((center_within_hole or significant_overlap) and
                circle_diameter <= hole_width * 0.95 and
                y >= barrier_y - 15)
    
    def _apply_funnel_physics(self, circle: Circle, x: float, hole_center: float) -> float:
        """Apply funnel slope physics to redirect circle toward hole"""
        if x < hole_center:
            # Left side - slope toward center
            slope_direction_x = 1.0
            distance_to_center = hole_center - x
        else:
            # Right side - slope toward center
            slope_direction_x = -1.0
            distance_to_center = x - hole_center
        
        # Apply sliding force down the slope
        slide_force = 0.6 + (distance_to_center / 30.0)
        circle.vx += slope_direction_x * slide_force
        
        # Prevent reverse motion away from center
        if x < hole_center and circle.vx < 0:
            circle.vx = abs(circle.vx) * 0.5
        elif x > hole_center and circle.vx > 0:
            circle.vx = -abs(circle.vx) * 0.5
        
        # Maintain downward motion
        vy = max(circle.vy * 0.4, 0.2)
        circle.vx *= 0.9
        
        # Update stuck counter
        if abs(circle.vx) < 0.1 and abs(vy) < 0.1:
            circle.stuck_counter += 1
        
        return vy
    
    def _handle_circle_collisions(self, circle: Circle, new_x: float, new_y: float, 
                                 all_circles: List[Circle]) -> Tuple[float, float]:
        """Handle collisions between circles"""
        for other in all_circles:
            if other.id != circle.id and other.active:
                dx = new_x - other.x
                dy = new_y - other.y
                distance = math.sqrt(dx*dx + dy*dy)
                min_distance = circle.size + other.size
                
                if distance < min_distance and distance > 0:
                    # Simple collision response
                    overlap = min_distance - distance
                    dx_norm = dx / distance
                    dy_norm = dy / distance
                    
                    # Push circles apart
                    push = overlap * 0.5
                    new_x += dx_norm * push
                    new_y += dy_norm * push
                    
                    # Velocity exchange
                    circle.vx += dx_norm * 0.1
                    circle.vy += dy_norm * 0.1
        
        return new_x, new_y
    
    def _render_frame(self, frame_idx: int, include_labels: bool = False, 
                     actual_jam_type: Optional[str] = None) -> Image.Image:
        """Render a single frame of the simulation"""
        # Create frame
        img = Image.new('RGB', (self.params.width, self.params.height), 
                       tuple(self.params.background_color))
        draw = ImageDraw.Draw(img)
        
        # Draw funnel structure
        self._draw_funnel(draw)
        
        # Draw circles
        active_circles = [c for c in self.circles if c.active]
        self._draw_circles(draw, active_circles)
        
        # Add labels if requested
        if include_labels:
            self._draw_labels(draw, frame_idx, active_circles, actual_jam_type)
        
        # Add noise if specified
        if self.params.noise_level > 0:
            img = self._add_noise(img)
        
        return img
    
    def _draw_funnel(self, draw: ImageDraw.Draw):
        """Draw the funnel structure"""
        barrier_y = self.params.height - 5
        funnel_center_x = self.hole_x
        funnel_width = self.params.hole_diameter
        funnel_depth = 40
        
        hole_left = funnel_center_x - funnel_width // 2
        hole_right = funnel_center_x + funnel_width // 2
        
        # Left side of funnel
        left_funnel_points = [
            (0, barrier_y - funnel_depth),
            (hole_left, barrier_y),
            (hole_left, self.params.height),
            (0, self.params.height)
        ]
        draw.polygon(left_funnel_points, fill=(100, 100, 100))
        
        # Right side of funnel
        right_funnel_points = [
            (hole_right, barrier_y),
            (self.params.width, barrier_y - funnel_depth),
            (self.params.width, self.params.height),
            (hole_right, self.params.height)
        ]
        draw.polygon(right_funnel_points, fill=(100, 100, 100))
        
        # Hole opening
        hole_opening = [
            (hole_left, barrier_y),
            (hole_right, barrier_y),
            (hole_right, self.params.height),
            (hole_left, self.params.height)
        ]
        draw.polygon(hole_opening, fill=(20, 20, 20))
        
        # Hole edges
        draw.line([(hole_left, barrier_y), (hole_left, barrier_y+5)], 
                 fill=(150, 150, 150), width=2)
        draw.line([(hole_right, barrier_y), (hole_right, barrier_y+5)], 
                 fill=(150, 150, 150), width=2)
    
    def _draw_circles(self, draw: ImageDraw.Draw, circles: List[Circle]):
        """Draw all visible circles"""
        for circle in circles:
            if circle.y > -circle.size and circle.y < self.params.height + circle.size:
                x, y = int(circle.x), int(circle.y)
                size = circle.size
                bbox = [x - size, y - size, x + size, y + size]
                draw.ellipse(bbox, fill=tuple(circle.color))
    
    def _draw_labels(self, draw: ImageDraw.Draw, frame_idx: int, 
                    circles: List[Circle], actual_jam_type: Optional[str]):
        """Draw informational labels on the frame"""
        try:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
            except:
                font = ImageFont.load_default()
        except:
            font = None
        
        labels = []
        
        # Wind information
        if self.params.wind_strength == 0:
            wind_label = "Wind: None"
        else:
            direction = "R→L" if self.params.wind_direction == -1 else "L→R"
            wind_label = f"Wind: {self.params.wind_strength:.1f} {direction}"
        labels.append(wind_label)
        
        # Hole and circle info
        info_label = (f"Hole: {self.params.hole_diameter}px | "
                     f"Circle Radius: {self.params.circle_size_min}-{self.params.circle_size_max}px")
        labels.append(info_label)
        
        # Current status
        status_label = (f"Frame: {frame_idx:02d} | Active: {len(circles)} | "
                       f"Total: {len(self.circles)} | Exited: {self.exit_count}")
        labels.append(status_label)
        
        # Jam type information
        if actual_jam_type:
            jam_label = f"Jam Type: {actual_jam_type.replace('_', ' ').title()}"
            labels.append(jam_label)
        
        # Draw labels with background
        y_offset = 5
        for label in labels:
            if font:
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                text_width = len(label) * 6
                text_height = 12
            
            # Background rectangle
            bg_rect = [2, y_offset-2, text_width+6, y_offset+text_height+2]
            draw.rectangle(bg_rect, fill=(0, 0, 0, 180))
            
            # Text
            if font:
                draw.text((4, y_offset), label, fill=(255, 255, 255), font=font)
            else:
                draw.text((4, y_offset), label, fill=(255, 255, 255))
            
            y_offset += text_height + 4
    
    def _add_noise(self, img: Image.Image) -> Image.Image:
        """Add noise to the image"""
        img_array = np.array(img)
        noise = np.random.normal(0, self.params.noise_level * 255, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def _get_visible_circles(self, circles: List[Circle]) -> List[Dict]:
        """Get metadata for visible circles"""
        visible = []
        for circle in circles:
            if circle.y > -circle.size and circle.y < self.params.height + circle.size:
                visible.append({
                    'id': circle.id,
                    'position': [circle.x, circle.y],
                    'velocity': [circle.vx, circle.vy],
                    'size': circle.size,
                    'color': circle.color,
                    'stuck_counter': circle.stuck_counter
                })
        return visible
    
    def _determine_jam_type(self) -> str:
        """Determine the jam type based on simulation results"""
        total_spawned = len(self.circles)
        exit_ratio = self.exit_count / max(total_spawned, 1)
        final_stuck = sum(1 for c in self.circles if c.stuck_counter > 5)
        
        if exit_ratio > 0.99 and final_stuck < 1:
            return "no_jam"
        elif exit_ratio > 0.3 and self.frames_since_last_exit < 30:
            return "partial_jam"
        else:
            return "full_jam"


# Compatibility function - matches original API
def generate_falling_circles_video(params: VideoParams, seed: Optional[int] = None, 
                                 include_labels: bool = False, 
                                 actual_jam_type: Optional[str] = None) -> Tuple[List[np.ndarray], Dict]:
    """
    Generate falling circles video - compatibility function
    
    This function provides the same API as the original generate_falling_circles_video
    but uses the new environment-based implementation.
    
    Args:
        params: VideoParams object
        seed: Random seed
        include_labels: Whether to include labels
        actual_jam_type: Override jam type for display
        
    Returns:
        Tuple of (frames, metadata)
    """
    env = FallingCirclesEnvironment(params)
    return env.generate_video(seed=seed, include_labels=include_labels, 
                             actual_jam_type=actual_jam_type)