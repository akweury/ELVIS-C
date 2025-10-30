# scripts/spread_line.py
"""
Generate N videos where objects start at center and spread to horizontal line.
Each video includes meta.json with frame-level ground-truth information.
80% training videos (3-5 objects), 20% test videos (5-10 objects) with intervention capability at any frame.

Usage:
  python scripts/spread_line.py --num_videos 1000 --out data/spread_line --workers 8
"""
import os
import json
import random
import argparse
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path

from elvis_env.pairing.make_pairs import make_intervention_pair
from elvis_env.io.export import export_pair

# ------- video generation parameters ---------------------------------------
class VideoParams:
    """Parameters controlling video generation logic"""
    def __init__(self, 
                 num_frames=20,
                 width=224,
                 height=224,
                 num_objects=4,
                 object_size=8,
                 object_color=None,
                 background_color=None,
                 noise_level=0.0,
                 intervention_frame=None,
                 intervention_type=None,
                 shape_type="circle",
                 initial_formation="circle"):
        self.num_frames = num_frames
        self.width = width
        self.height = height
        self.num_objects = num_objects
        self.object_size = object_size
        self.object_color = object_color or [255, 255, 255]  # white default
        self.background_color = background_color or [0, 0, 0]  # black default
        self.noise_level = noise_level
        self.intervention_frame = intervention_frame
        self.intervention_type = intervention_type
        self.shape_type = shape_type
        self.initial_formation = initial_formation
    
    def to_dict(self):
        return {
            'num_frames': self.num_frames,
            'width': self.width,
            'height': self.height,
            'num_objects': self.num_objects,
            'object_size': self.object_size,
            'object_color': self.object_color,
            'background_color': self.background_color,
            'noise_level': self.noise_level,
            'intervention_frame': self.intervention_frame,
            'intervention_type': self.intervention_type,
            'shape_type': self.shape_type,
            'initial_formation': self.initial_formation
        }

# ------- intervention types ------------------------------------------------
def sample_intervention_params(num_frames):
    """Sample random intervention parameters"""
    # No interventions for this dataset
    intervention_frame = None
    intervention_type = None
    
    return intervention_frame, intervention_type

def sample_video_params(num_frames=20, is_train=True):
    """Sample random parameters for video generation"""
    if is_train:
        # TRAINING: Conservative parameters for consistent learning
        num_objects = random.randint(3, 5)
        object_size = random.randint(8, 12)  # Medium sizes
        # Bright, saturated colors
        object_color = [random.randint(150, 255) for _ in range(3)]
        # Dark backgrounds
        background_color = [random.randint(0, 30) for _ in range(3)]
        noise_level = random.uniform(0.0, 0.05)  # Low noise
        shape_type = "circle"  # Only circles for training
        initial_formation = "circle"  # Only circular formation
    else:
        # TEST: Diverse parameters for generalization testing
        num_objects = random.randint(5, 12)  # More objects
        object_size = random.randint(4, 20)  # Wide size range
        
        # More diverse colors including pastels and darker tones
        color_style = random.choice(["bright", "pastel", "dark", "monochrome"])
        if color_style == "bright":
            object_color = [random.randint(200, 255) for _ in range(3)]
        elif color_style == "pastel":
            object_color = [random.randint(100, 200) for _ in range(3)]
        elif color_style == "dark":
            object_color = [random.randint(50, 150) for _ in range(3)]
        else:  # monochrome
            intensity = random.randint(100, 255)
            object_color = [intensity, intensity, intensity]
        
        # Varied backgrounds including lighter ones
        bg_style = random.choice(["dark", "light", "colored"])
        if bg_style == "dark":
            background_color = [random.randint(0, 50) for _ in range(3)]
        elif bg_style == "light":
            background_color = [random.randint(200, 255) for _ in range(3)]
        else:  # colored
            background_color = [random.randint(30, 100) for _ in range(3)]
        
        noise_level = random.uniform(0.0, 0.15)  # Higher noise
        shape_type = random.choice(["circle", "square", "triangle"])
        initial_formation = random.choice(["circle", "grid", "random", "line"])
    
    # Sample intervention (disabled for now)
    intervention_frame, intervention_type = sample_intervention_params(num_frames)
    
    return VideoParams(
        num_frames=num_frames,
        num_objects=num_objects,
        object_size=object_size,
        object_color=object_color,
        background_color=background_color,
        noise_level=noise_level,
        intervention_frame=intervention_frame,
        intervention_type=intervention_type,
        shape_type=shape_type,
        initial_formation=initial_formation
    )

# ------- video generation core ---------------------------------------------
def generate_spread_line_video(params, seed):
    """
    Generate a spread line video where N objects start at center and spread to horizontal line.
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
        'frames': []
    }
    
    # Initial positions based on formation type
    center_x = params.width // 2
    center_y = params.height // 2
    
    # Create initial positions based on formation type
    initial_positions = []
    
    if params.num_objects == 1:
        initial_positions = [(center_x, center_y)]
    elif params.initial_formation == "circle":
        # Place objects evenly around a circle (training default)
        initial_radius = max(params.object_size * 1.5, 20)
        import math
        for i in range(params.num_objects):
            angle = 2 * math.pi * i / params.num_objects
            init_x = center_x + initial_radius * math.cos(angle)
            init_y = center_y + initial_radius * math.sin(angle)
            initial_positions.append((init_x, init_y))
    elif params.initial_formation == "grid":
        # Grid formation for test videos
        grid_size = int(math.ceil(math.sqrt(params.num_objects)))
        spacing = max(params.object_size * 2, 25)
        start_x = center_x - (grid_size - 1) * spacing / 2
        start_y = center_y - (grid_size - 1) * spacing / 2
        
        for i in range(params.num_objects):
            row = i // grid_size
            col = i % grid_size
            init_x = start_x + col * spacing
            init_y = start_y + row * spacing
            initial_positions.append((init_x, init_y))
    elif params.initial_formation == "random":
        # Random positions in center area for test videos
        margin = max(params.object_size * 2, 30)
        for i in range(params.num_objects):
            init_x = center_x + random.uniform(-margin, margin)
            init_y = center_y + random.uniform(-margin, margin)
            initial_positions.append((init_x, init_y))
    elif params.initial_formation == "line":
        # Vertical line formation for test videos
        spacing = max(params.object_size * 1.5, 15)
        start_y = center_y - (params.num_objects - 1) * spacing / 2
        for i in range(params.num_objects):
            init_x = center_x
            init_y = start_y + i * spacing
            initial_positions.append((init_x, init_y))
    
    # Final positions (horizontal line across the image)
    line_y = center_y  # Keep same y-coordinate (horizontal line)
    line_margin = params.width * 0.1  # 10% margin on each side
    line_width = params.width - 2 * line_margin
    
    # Calculate final x positions evenly spaced along the line
    if params.num_objects == 1:
        final_x_positions = [center_x]
    else:
        final_x_positions = [
            line_margin + i * (line_width / (params.num_objects - 1))
            for i in range(params.num_objects)
        ]
    
    # Track current object states
    current_objects = []
    for i in range(params.num_objects):
        init_x, init_y = initial_positions[i]
        current_objects.append({
            'id': i,
            'x': init_x,
            'y': init_y,
            'initial_x': init_x,
            'initial_y': init_y,
            'target_x': final_x_positions[i],
            'target_y': line_y,
            'color': params.object_color.copy(),
            'size': params.object_size,
            'visible': True
        })
    
    for frame_idx in range(params.num_frames):
        # Create frame
        img = Image.new('RGB', (params.width, params.height), tuple(params.background_color))
        draw = ImageDraw.Draw(img)
        
        # Calculate interpolation factor (0 at start, 1 at end)
        if params.num_frames > 1:
            t = frame_idx / (params.num_frames - 1)
        else:
            t = 0
        
        # Apply interventions if needed
        intervention_applied = params.intervention_frame and frame_idx >= params.intervention_frame
        if intervention_applied:
            if params.intervention_type == "freeze_objects":
                # Stop movement by keeping t at intervention frame value
                t = (params.intervention_frame - 1) / (params.num_frames - 1) if params.num_frames > 1 else 0
            elif params.intervention_type == "change_speed":
                # Double the movement speed after intervention
                t_intervention = (params.intervention_frame - 1) / (params.num_frames - 1) if params.num_frames > 1 else 0
                frames_since_intervention = frame_idx - params.intervention_frame + 1
                t = t_intervention + (frames_since_intervention * 2.0 / (params.num_frames - 1))
                t = min(t, 1.0)  # Cap at 1.0
            elif params.intervention_type == "reverse_direction":
                # Reverse the movement direction
                t = 1.0 - t
            elif params.intervention_type == "change_color":
                # Change object colors
                for obj in current_objects:
                    obj['color'] = [255 - c for c in params.object_color]
            elif params.intervention_type == "remove_object":
                # Remove the first object
                if current_objects:
                    current_objects[0]['visible'] = False
            elif params.intervention_type == "add_object":
                # Add a new object (if not already at max)
                if len([obj for obj in current_objects if obj['visible']]) < 10:
                    new_obj = {
                        'id': len(current_objects),
                        'x': center_x,
                        'y': center_y,
                        'initial_x': center_x,
                        'initial_y': center_y,
                        'target_x': center_x + random.uniform(-50, 50),
                        'target_y': line_y,
                        'color': params.object_color.copy(),
                        'size': params.object_size,
                        'visible': True
                    }
                    current_objects.append(new_obj)
        
        # Update object positions
        for obj in current_objects:
            if obj['visible']:
                # Linear interpolation from initial to target position
                obj['x'] = obj['initial_x'] + t * (obj['target_x'] - obj['initial_x'])
                obj['y'] = obj['initial_y'] + t * (obj['target_y'] - obj['initial_y'])
        
        # Draw objects with different shapes
        visible_objects = []
        for obj in current_objects:
            if obj['visible']:
                x, y = int(obj['x']), int(obj['y'])
                size = obj['size']
                
                if params.shape_type == "circle":
                    # Draw circular object
                    bbox = [x - size//2, y - size//2, x + size//2, y + size//2]
                    draw.ellipse(bbox, fill=tuple(obj['color']))
                elif params.shape_type == "square":
                    # Draw square object
                    bbox = [x - size//2, y - size//2, x + size//2, y + size//2]
                    draw.rectangle(bbox, fill=tuple(obj['color']))
                elif params.shape_type == "triangle":
                    # Draw triangular object
                    half_size = size // 2
                    triangle_points = [
                        (x, y - half_size),  # top
                        (x - half_size, y + half_size),  # bottom left
                        (x + half_size, y + half_size)   # bottom right
                    ]
                    draw.polygon(triangle_points, fill=tuple(obj['color']))
                
                visible_objects.append({
                    'id': obj['id'],
                    'position': [obj['x'], obj['y']],
                    'color': obj['color'],
                    'size': obj['size'],
                    'shape': params.shape_type
                })
        
        # Add noise if specified
        if params.noise_level > 0:
            img_array = np.array(img)
            noise = np.random.normal(0, params.noise_level * 255, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)
        
        frames.append(np.array(img))
        
        # Frame metadata
        frame_meta = {
            'frame_idx': frame_idx,
            'interpolation_t': t,
            'objects': visible_objects,
            'intervention_applied': intervention_applied,
            'intervention_type': params.intervention_type if intervention_applied else None,
            'num_visible_objects': len(visible_objects)
        }
        meta_data['frames'].append(frame_meta)
    
    return frames, meta_data

# ------- worker function ---------------------------------------------------
def _generate_video_worker(video_idx, params, output_root, split, export_gif=True, gif_fps=5):
    """Worker function to generate a single video"""
    seed = random.randint(0, 2_000_000_000)
    
    # Generate video
    frames, meta_data = generate_spread_line_video(params, seed)
    
    # Create output directory
    video_name = f"video_{video_idx:05d}"
    video_dir = os.path.join(output_root, split, video_name)
    os.makedirs(video_dir, exist_ok=True)
    
    # Save frames as PNG files
    for frame_idx, frame in enumerate(frames):
        from PIL import Image
        img = Image.fromarray(frame)
        frame_path = os.path.join(video_dir, f"frame_{frame_idx:03d}.png")
        img.save(frame_path)
    
    # Save meta.json
    meta_path = os.path.join(video_dir, "meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    
    # Save GIF if requested
    if export_gif:
        from PIL import Image
        gif_path = os.path.join(video_dir, f"{video_name}.gif")
        images = [Image.fromarray(frame) for frame in frames]
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
        'has_intervention': params.intervention_frame is not None
    }

# ------- main function -----------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate videos where objects spread from center to horizontal line")
    parser.add_argument("--num_videos", type=int, default=30,
                       help="Total number of videos to generate")
    parser.add_argument("--out", type=str, default="data/spread_line",
                       help="Output directory for videos")
    parser.add_argument("--num_frames", type=int, default=20,
                       help="Number of frames per video")
    parser.add_argument("--workers", type=int, default=8,
                       help="Number of parallel workers")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                       help="Ratio of training videos (default: 0.8)")
    parser.add_argument("--export_gif", action="store_true", default=True,
                       help="Export GIF files for visualization")
    parser.add_argument("--no_gif", action="store_true",
                       help="Disable GIF export")
    parser.add_argument("--gif_fps", type=int, default=5,
                       help="Frame rate for GIF files")
    parser.add_argument("--width", type=int, default=224,
                       help="Video width in pixels")
    parser.add_argument("--height", type=int, default=224,
                       help="Video height in pixels")
    
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
    
    # Generate video parameters and assignments
    tasks = []
    for i in range(args.num_videos):
        split = "train" if i < num_train else "test"
        is_train = (split == "train")
        params = sample_video_params(args.num_frames, is_train=is_train)
        params.width = args.width
        params.height = args.height
        
        tasks.append((i, params, split))
    
    # Generate videos in parallel
    results = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(_generate_video_worker, video_idx, params, args.out, split, export_gif, args.gif_fps) 
                  for video_idx, params, split in tasks]
        
        for future in tqdm(as_completed(futures), total=args.num_videos, desc="Generating videos"):
            results.append(future.result())
    
    # Save dataset manifest
    manifest_path = os.path.join(args.out, "dataset_manifest.json")
    manifest_data = {
        'total_videos': args.num_videos,
        'train_videos': num_train,
        'test_videos': num_test,
        'video_params': {
            'num_frames': args.num_frames,
            'width': args.width,
            'height': args.height
        },
        'videos': sorted(results, key=lambda x: x['video_idx'])
    }
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest_data, f, indent=2)
    
    # Print summary
    intervention_count = sum(1 for r in results if r['has_intervention'])
    gif_status = f"with GIFs (fps={args.gif_fps})" if export_gif else "without GIFs"
    
    print(f"\n✅ Generated {args.num_videos} object spreading videos")
    print(f"📁 Output directory: {args.out}")
    print(f"🚂 Training videos: {num_train} (3-5 circles, circular formation, bright colors)")
    print(f"🧪 Test videos: {num_test} (5-12 objects, varied shapes/formations/colors)")
    print(f"🎬 Videos with interventions: {intervention_count}")
    print(f"🎨 Frame format: PNG {gif_status}")
    print(f"📄 Dataset manifest: {manifest_path}")

if __name__ == "__main__":
    main()
