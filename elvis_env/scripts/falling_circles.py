# scripts/falling_circles.py
"""
Generate videos of circles falling downward with wind effects and exit hole dynamics.
Scenarios: no_jam, partial_jam, full_jam based on hole size and circle properties.

Usage:
  python scripts/falling_circles.py --num_videos 1000 --out data/falling_circles --workers 8
"""
import os
import json
import random
import argparse
import math
import csv
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path

# ------- video generation parameters ---------------------------------------
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

# ------- parameter sampling ------------------------------------------------
def sample_video_params(num_frames=60, is_train=True):
    """Sample random parameters for video generation - jam type determined after simulation"""
    
    width, height = 224, 224
    
    # All parameters are now randomly sampled without jam type bias
    
    # Random number of circles
    num_circles = random.randint(3, 8)
    
    # Random physics parameters
    gravity = 1.0  # Constant gravity for physics consistency
    spawn_rate = random.uniform(0.15, 0.6)
    
    # Random wind parameters
    wind_strength = random.uniform(0, 0.2)
    wind_direction = random.choice([-1, 1])  # left or right
    
    # Random circle size parameters (ensure minimum diameter of 10px = radius 5)
    circle_size_min = random.randint(5, 10)   # Minimum radius 5 = 10px diameter
    circle_size_max = random.randint(circle_size_min + 2, 15)  # Maximum radius 15 = 30px diameter
    
    # Random hole diameter (independent of circle sizes)
    hole_diameter = random.randint(15, 30)
    
    # Hole position (mostly center, sometimes off-center)
    if random.random() < 0.8:
        hole_x_position = 0.5  # center
    else:
        hole_x_position = random.uniform(0.3, 0.7)  # slightly off-center
    
    # Visual parameters
    if is_train:
        circle_color = [random.randint(150, 255), random.randint(150, 255), random.randint(200, 255)]
        background_color = [random.randint(20, 60) for _ in range(3)]
    else:
        # More diverse colors for test
        color_style = random.choice(["blue", "red", "green", "mixed"])
        if color_style == "blue":
            circle_color = [random.randint(100, 200), random.randint(150, 255), random.randint(200, 255)]
        elif color_style == "red":
            circle_color = [random.randint(200, 255), random.randint(100, 200), random.randint(100, 200)]
        elif color_style == "green":
            circle_color = [random.randint(100, 200), random.randint(200, 255), random.randint(100, 200)]
        else:  # mixed
            circle_color = [random.randint(150, 255) for _ in range(3)]
        
        background_color = [random.randint(10, 80) for _ in range(3)]
    
    noise_level = random.uniform(0.0, 0.05) if is_train else random.uniform(0.0, 0.1)
    
    return VideoParams(
        num_frames=num_frames,
        width=width,
        height=height,
        num_circles=num_circles,
        circle_size_min=circle_size_min,
        circle_size_max=circle_size_max,
        hole_diameter=hole_diameter,
        hole_x_position=hole_x_position,
        wind_strength=wind_strength,
        wind_direction=wind_direction,
        gravity=gravity,
        spawn_rate=spawn_rate,
        circle_color=circle_color,
        background_color=background_color,
        noise_level=noise_level
    )

# ------- physics simulation ------------------------------------------------
class Circle:
    def __init__(self, x, y, size, color, circle_id):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.size = size
        self.color = color
        self.id = circle_id
        self.active = True
        self.exited = False
        self.stuck_counter = 0

def generate_falling_circles_video(params, seed=None, include_labels=False, actual_jam_type=None):
    """
    Generate falling circles video with wind and hole dynamics
    Returns: frames (list of numpy arrays), meta_data (dict)
    """
    import numpy as np
    import math
    from PIL import Image, ImageDraw
    
    random.seed(seed)
    np.random.seed(seed)
    
    frames = []
    meta_data = {
        'params': params.to_dict(),
        'seed': seed,
        'frames': [],
        'actual_jam_type': 'unknown'
    }
    
    # Calculate hole position
    hole_x = int(params.hole_x_position * params.width)
    hole_y = params.height - 10  # near bottom
    
    # Track circles
    circles = []
    next_circle_id = 0
    spawn_timer = 0
    
    # Statistics for jam detection
    exit_count = 0
    stuck_count = 0
    frames_since_last_exit = 0
    
    for frame_idx in range(params.num_frames):
        # Create frame
        img = Image.new('RGB', (params.width, params.height), tuple(params.background_color))
        draw = ImageDraw.Draw(img)
        
        # Draw funnel-shaped bottom (V-shaped from left to right) with visible hole
        barrier_y = params.height - 5
        funnel_center_x = hole_x
        funnel_width = params.hole_diameter
        funnel_depth = 40  # How deep the V-shape goes
        
        # Calculate hole boundaries
        hole_left = funnel_center_x - funnel_width//2
        hole_right = funnel_center_x + funnel_width//2
        
        # Create left side of funnel (from left edge to hole)
        left_funnel_points = [
            (0, barrier_y - funnel_depth),          # Left edge top
            (hole_left, barrier_y),                 # Left edge of hole
            (hole_left, params.height),             # Left edge bottom
            (0, params.height)                      # Left edge bottom corner
        ]
        draw.polygon(left_funnel_points, fill=(100, 100, 100))
        
        # Create right side of funnel (from hole to right edge)
        right_funnel_points = [
            (hole_right, barrier_y),                # Right edge of hole
            (params.width, barrier_y - funnel_depth), # Right edge top
            (params.width, params.height),          # Right edge bottom corner
            (hole_right, params.height)             # Right edge bottom
        ]
        draw.polygon(right_funnel_points, fill=(100, 100, 100))
        
        # Draw hole opening (dark area to show it's an opening)
        hole_opening = [
            (hole_left, barrier_y),
            (hole_right, barrier_y),
            (hole_right, params.height),
            (hole_left, params.height)
        ]
        draw.polygon(hole_opening, fill=(20, 20, 20))  # Dark hole
        
        # Draw hole edges for clarity
        draw.line([(hole_left, barrier_y), (hole_left, barrier_y+5)], fill=(150, 150, 150), width=2)
        draw.line([(hole_right, barrier_y), (hole_right, barrier_y+5)], fill=(150, 150, 150), width=2)
        
        # Add informational labels at the top (only if requested)
        if include_labels:
            try:
                from PIL import ImageFont
                # Try to use a default font
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
                except:
                    font = ImageFont.load_default()
            except:
                font = None
            # Create label text
            labels = []
            # Wind information
            if params.wind_strength == 0:
                wind_label = "Wind: None"
            else:
                direction = "R→L" if params.wind_direction == -1 else "L→R"
                wind_label = f"Wind: {params.wind_strength:.1f} {direction}"
            labels.append(wind_label)
            
            # Hole and circle info
            info_label = f"Hole: {params.hole_diameter}px | Circle Radius: {params.circle_size_min}-{params.circle_size_max}px"
            labels.append(info_label)
            
            # Current status
            status_label = f"Frame: {frame_idx:02d} | Active: {len([c for c in circles if c.active])} | Total: {len(circles)} | Exited: {exit_count}"
            labels.append(status_label)
            
            # Jam type information (if available)
            if actual_jam_type:
                jam_label = f"Jam Type: {actual_jam_type.replace('_', ' ').title()}"
                labels.append(jam_label)
            
            # Draw labels with background for better readability
            y_offset = 5
            for label in labels:
                # Get text size
                if font:
                    bbox = draw.textbbox((0, 0), label, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                else:
                    text_width = len(label) * 6  # Approximate
                    text_height = 12
                
                # Draw background rectangle
                bg_rect = [2, y_offset-2, text_width+6, y_offset+text_height+2]
                draw.rectangle(bg_rect, fill=(0, 0, 0, 180))  # Semi-transparent black
                
                # Draw text
                if font:
                    draw.text((4, y_offset), label, fill=(255, 255, 255), font=font)
                else:
                    draw.text((4, y_offset), label, fill=(255, 255, 255))
                
                y_offset += text_height + 4
        
        # Spawn new circles (limited to total num_circles)
        spawn_timer += params.spawn_rate
        while spawn_timer >= 1.0 and len(circles) < params.num_circles:
            spawn_x = random.uniform(params.circle_size_max, params.width - params.circle_size_max)
            spawn_y = random.uniform(-20, -5)
            size = random.randint(params.circle_size_min, params.circle_size_max)
            
            new_circle = Circle(spawn_x, spawn_y, size, params.circle_color.copy(), next_circle_id)
            circles.append(new_circle)
            next_circle_id += 1
            spawn_timer -= 1.0
        
        # Update physics for each circle
        active_circles = [c for c in circles if c.active]
        
        for circle in active_circles:
            # Apply gravity
            circle.vy += params.gravity
            
            # Apply wind
            circle.vx += params.wind_strength * params.wind_direction
            
            # Apply drag
            circle.vx *= 0.99
            circle.vy *= 0.999
            
            # Update position
            new_x = circle.x + circle.vx
            new_y = circle.y + circle.vy
            
            # Collision with walls
            if new_x - circle.size < 0:
                new_x = circle.size
                circle.vx = abs(circle.vx) * 0.3
            elif new_x + circle.size > params.width:
                new_x = params.width - circle.size
                circle.vx = -abs(circle.vx) * 0.3
            
            # Collision with V-shaped funnel bottom
            barrier_y = params.height - 5
            funnel_center_x = hole_x
            funnel_width = params.hole_diameter
            funnel_depth = 40
            
            # Calculate the funnel surface height at the circle's x position
            # Linear interpolation from edges to center
            if new_x <= funnel_center_x:
                # Left side of funnel
                progress = new_x / funnel_center_x
                funnel_surface_y = (barrier_y - funnel_depth) + progress * funnel_depth
            else:
                # Right side of funnel
                progress = (params.width - new_x) / (params.width - funnel_center_x)
                funnel_surface_y = (barrier_y - funnel_depth) + progress * funnel_depth
            
            # Check if circle hits the funnel surface
            if new_y + circle.size >= funnel_surface_y:
                # Check if circle is positioned to exit through hole
                hole_left = funnel_center_x - funnel_width//2
                hole_right = funnel_center_x + funnel_width//2
                
                # Very lenient exit conditions - multiple ways to exit
                hole_center_distance = abs(new_x - funnel_center_x)
                circle_diameter = circle.size * 2
                
                # Condition 1: Circle center is well within hole and circle fits
                center_within_hole = hole_center_distance < (funnel_width//2 - circle.size)
                
                # Condition 2: Circle overlaps significantly with hole opening
                circle_left = new_x - circle.size
                circle_right = new_x + circle.size
                overlap_left = max(circle_left, hole_left)
                overlap_right = min(circle_right, hole_right)
                overlap_width = max(0, overlap_right - overlap_left)
                significant_overlap = overlap_width > circle_diameter * 0.6  # 60% overlap
                
                # Allow exit if either condition is met and circle can fit
                if ((center_within_hole or significant_overlap) and
                    circle_diameter <= funnel_width * 0.95 and  # Circle fits with 5% margin
                    new_y >= barrier_y - 15):  # Extended exit zone
                    # Circle exits through hole - disappears
                    circle.active = False
                    circle.exited = True
                    exit_count += 1
                    frames_since_last_exit = 0
                else:
                    # Collision with funnel surface - implement proper slope physics
                    new_y = funnel_surface_y - circle.size
                    
                    # Calculate slope direction and normal
                    if new_x < funnel_center_x:
                        # Left side - slope goes down-right toward center
                        slope_direction_x = 1.0  # Toward center (right)
                        slope_direction_y = funnel_depth / funnel_center_x  # Down
                        distance_to_center = funnel_center_x - new_x
                    else:
                        # Right side - slope goes down-left toward center
                        slope_direction_x = -1.0  # Toward center (left)
                        slope_direction_y = funnel_depth / (params.width - funnel_center_x)  # Down
                        distance_to_center = new_x - funnel_center_x
                    
                    # Normalize slope direction
                    slope_length = math.sqrt(slope_direction_x**2 + slope_direction_y**2)
                    slope_dir_x = slope_direction_x / slope_length
                    slope_dir_y = slope_direction_y / slope_length
                    
                    # Apply stronger sliding force down the slope toward hole
                    # The farther from center, the stronger the force
                    slide_force = 0.6 + (distance_to_center / 30.0)  # Increased base force
                    circle.vx += slope_dir_x * slide_force
                    circle.vy += slope_dir_y * slide_force * 0.5  # More downward component
                    
                    # Prevent reverse motion away from center
                    if new_x < funnel_center_x and circle.vx < 0:  # Left side moving left
                        circle.vx = abs(circle.vx) * 0.5  # Redirect toward center
                    elif new_x > funnel_center_x and circle.vx > 0:  # Right side moving right
                        circle.vx = -abs(circle.vx) * 0.5  # Redirect toward center
                    
                    # Dampen bouncing but maintain flow
                    circle.vy = max(circle.vy * 0.4, 0.2)  # Keep downward motion
                    
                    # Reduced friction to allow better flow
                    circle.vx *= 0.9
                    
                    # Only count as stuck if moving very slowly
                    if abs(circle.vx) < 0.1 and abs(circle.vy) < 0.1:
                        circle.stuck_counter += 1
            
            # Collision with other circles (simplified)
            for other in active_circles:
                if other.id != circle.id:
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
                        
                        # Velocity exchange (simplified)
                        circle.vx += dx_norm * 0.1
                        circle.vy += dy_norm * 0.1
            
            circle.x = new_x
            circle.y = new_y
        
        # Draw circles
        visible_circles = []
        for circle in active_circles:
            if circle.y > -circle.size and circle.y < params.height + circle.size:
                # Draw circle
                x, y = int(circle.x), int(circle.y)
                size = circle.size
                bbox = [x - size, y - size, x + size, y + size]
                draw.ellipse(bbox, fill=tuple(circle.color))
                
                visible_circles.append({
                    'id': circle.id,
                    'position': [circle.x, circle.y],
                    'velocity': [circle.vx, circle.vy],
                    'size': circle.size,
                    'color': circle.color,
                    'stuck_counter': circle.stuck_counter
                })
        
        # Add noise if specified
        if params.noise_level > 0:
            img_array = np.array(img)
            noise = np.random.normal(0, params.noise_level * 255, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)
        
        frames.append(np.array(img))
        
        # Update jam detection statistics
        frames_since_last_exit += 1
        current_stuck = sum(1 for c in active_circles if c.stuck_counter > 10)
        
        # Frame metadata
        frame_meta = {
            'frame_idx': frame_idx,
            'num_active_circles': len(active_circles),
            'num_visible_circles': len(visible_circles),
            'circles_exited': exit_count,
            'circles_stuck': current_stuck,
            'frames_since_last_exit': frames_since_last_exit,
            'hole_position': [hole_x, hole_y],
            'wind_effect': params.wind_strength * params.wind_direction,
            'circles': visible_circles
        }
        meta_data['frames'].append(frame_meta)
    
    # Determine actual jam type based on simulation
    total_spawned = len(circles)
    exit_ratio = exit_count / max(total_spawned, 1)
    final_stuck = sum(1 for c in circles if c.stuck_counter > 5)
    
    if exit_ratio > 0.99 and final_stuck < 1:
        actual_jam_type = "no_jam"
    elif exit_ratio > 0.3 and frames_since_last_exit < 30:
        actual_jam_type = "partial_jam"
    else:
        actual_jam_type = "full_jam"
    
    meta_data['actual_jam_type'] = actual_jam_type
    meta_data['exit_statistics'] = {
        'total_spawned': total_spawned,
        'total_exited': exit_count,
        'exit_ratio': exit_ratio,
        'final_stuck_count': final_stuck
    }
    
    return frames, meta_data

# ------- worker function ---------------------------------------------------
def _generate_video_worker(video_idx, params, output_root, split, export_gif=True, gif_fps=10):
    """Worker function to generate a single falling circles video"""
    import csv
    import math
    
    seed = random.randint(0, 2_000_000_000)
    
    # Generate video frames WITHOUT labels for PNG files
    frames, meta_data = generate_falling_circles_video(params, seed, include_labels=False)
    
    # Create output directory
    video_name = f"video_{video_idx:05d}"
    video_dir = os.path.join(output_root, split, video_name)
    os.makedirs(video_dir, exist_ok=True)
    
    # Save frames as PNG files (without labels)
    for frame_idx, frame in enumerate(frames):
        from PIL import Image
        img = Image.fromarray(frame)
        frame_path = os.path.join(video_dir, f"frame_{frame_idx:03d}.png")
        img.save(frame_path)
    
    # Save meta.json
    meta_path = os.path.join(video_dir, "meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    
    # Save frame-level facts as CSV
    csv_path = os.path.join(video_dir, "frame_facts.csv")
    
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'frame_idx', 'num_active_circles', 'num_visible_circles',
            'circles_exited', 'circles_stuck', 'frames_since_last_exit',
            'hole_x', 'hole_y', 'hole_diameter', 'wind_effect',
            'gravity', 'actual_jam_type',
            # Circle aggregates
            'avg_circle_y', 'min_circle_y', 'max_circle_y',
            'avg_velocity_y', 'total_kinetic_energy',
            'circles_near_hole', 'flow_rate'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        prev_exit_count = 0
        for frame_meta in meta_data['frames']:
            # Calculate aggregate statistics
            if frame_meta['circles']:
                positions_y = [c['position'][1] for c in frame_meta['circles']]
                velocities_y = [c['velocity'][1] for c in frame_meta['circles']]
                
                avg_y = sum(positions_y) / len(positions_y)
                min_y = min(positions_y)
                max_y = max(positions_y)
                avg_vy = sum(velocities_y) / len(velocities_y)
                
                # Kinetic energy approximation
                total_ke = sum(v[0]**2 + v[1]**2 for v in [c['velocity'] for c in frame_meta['circles']])
                
                # Circles near hole
                hole_x = frame_meta['hole_position'][0]
                near_hole = sum(1 for c in frame_meta['circles'] 
                              if abs(c['position'][0] - hole_x) < params.hole_diameter)
            else:
                avg_y = min_y = max_y = avg_vy = total_ke = near_hole = 0
            
            # Flow rate (circles exited this frame)
            current_exits = frame_meta['circles_exited']
            flow_rate = current_exits - prev_exit_count
            prev_exit_count = current_exits
            
            row = {
                'frame_idx': frame_meta['frame_idx'],
                'num_active_circles': frame_meta['num_active_circles'],
                'num_visible_circles': frame_meta['num_visible_circles'],
                'circles_exited': frame_meta['circles_exited'],
                'circles_stuck': frame_meta['circles_stuck'],
                'frames_since_last_exit': frame_meta['frames_since_last_exit'],
                'hole_x': frame_meta['hole_position'][0],
                'hole_y': frame_meta['hole_position'][1],
                'hole_diameter': params.hole_diameter,
                'wind_effect': frame_meta['wind_effect'],
                'gravity': params.gravity,
                'actual_jam_type': meta_data['actual_jam_type'],
                'avg_circle_y': avg_y,
                'min_circle_y': min_y,
                'max_circle_y': max_y,
                'avg_velocity_y': avg_vy,
                'total_kinetic_energy': total_ke,
                'circles_near_hole': near_hole,
                'flow_rate': flow_rate
            }
            writer.writerow(row)
    
    # Save GIF if requested (with labels, in separate folder)
    if export_gif:
        # Generate frames WITH labels for GIF, including the actual jam type
        actual_jam_type = meta_data['actual_jam_type']
        gif_frames, _ = generate_falling_circles_video(params, seed, include_labels=True, actual_jam_type=actual_jam_type)
        
        # Create separate gifs directory
        gifs_dir = os.path.join(output_root, "gifs", split)
        os.makedirs(gifs_dir, exist_ok=True)
        
        from PIL import Image
        gif_path = os.path.join(gifs_dir, f"{video_name}.gif")
        images = [Image.fromarray(frame) for frame in gif_frames]
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=int(1000/gif_fps),
            loop=0
        )
    
    return {
        'video_idx': video_idx,
        'video_name': video_name,
        'split': split,
        'video_dir': video_dir,
        'num_frames': len(frames),
        'params': params.to_dict(),
        'seed': seed,
        'actual_jam_type': meta_data['actual_jam_type'],
        'exit_statistics': meta_data['exit_statistics']
    }

# ------- main function -----------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate falling circles videos with wind and hole dynamics")
    parser.add_argument("--num_videos", type=int, default=10,
                       help="Total number of videos to generate")
    parser.add_argument("--out", type=str, default="data/falling_circles",
                       help="Output directory for videos")
    parser.add_argument("--num_frames", type=int, default=60,
                       help="Number of frames per video")
    parser.add_argument("--workers", type=int, default=8,
                       help="Number of parallel workers")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                       help="Ratio of training videos (default: 0.8)")
    parser.add_argument("--export_gif", action="store_true", default=True,
                       help="Export GIF files for visualization")
    parser.add_argument("--no_gif", action="store_true",
                       help="Disable GIF export")
    parser.add_argument("--gif_fps", type=int, default=10,
                       help="Frame rate for GIF files")
    parser.add_argument("--balanced_scenarios", action="store_true",
                       help="Ensure equal distribution of jam types")
    
    args = parser.parse_args()
    
    # Handle GIF export
    export_gif = args.export_gif and not args.no_gif
    
    # Create output directories
    os.makedirs(args.out, exist_ok=True)
    train_dir = os.path.join(args.out, "train")
    test_dir = os.path.join(args.out, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Split videos into train/test
    num_train = int(args.num_videos * args.train_ratio)
    num_test = args.num_videos - num_train
    
    # Generate video parameters (all random now)
    tasks = []
    
    for i in range(args.num_videos):
        split = "train" if i < num_train else "test"
        is_train = (split == "train")
        params = sample_video_params(args.num_frames, is_train=is_train)
        
        tasks.append((i, params, split))
    
    # Generate videos in parallel
    results = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(_generate_video_worker, video_idx, params, args.out, split, export_gif, args.gif_fps) 
                  for video_idx, params, split in tasks]
        
        for future in tqdm(as_completed(futures), total=args.num_videos, desc="Generating falling circles videos"):
            results.append(future.result())
    
    # Save dataset manifest
    manifest_path = os.path.join(args.out, "dataset_manifest.json")
    
    # Analyze jam type distribution
    jam_counts = {"no_jam": 0, "partial_jam": 0, "full_jam": 0}
    for result in results:
        jam_counts[result['actual_jam_type']] += 1
    
    manifest_data = {
        'total_videos': args.num_videos,
        'train_videos': num_train,
        'test_videos': num_test,
        'video_params': {
            'num_frames': args.num_frames,
            'scenario_types': ['no_jam', 'partial_jam', 'full_jam']
        },
        'jam_type_distribution': jam_counts,
        'videos': sorted(results, key=lambda x: x['video_idx'])
    }
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest_data, f, indent=2)
    
    # Print summary
    gif_status = f"with GIFs (fps={args.gif_fps})" if export_gif else "without GIFs"
    
    print(f"\n✅ Generated {args.num_videos} falling circles videos")
    print(f"📁 Output directory: {args.out}")
    print(f"🚂 Training videos: {num_train}")
    print(f"🧪 Test videos: {num_test}")
    print(f"🌪️ Jam type distribution: {jam_counts}")
    print(f"🎨 Frame format: PNG {gif_status}")
    print(f"📊 Causal data: CSV files with flow dynamics")
    print(f"📄 Dataset manifest: {manifest_path}")

if __name__ == "__main__":
    main()