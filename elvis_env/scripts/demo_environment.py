#!/usr/bin/env python3
"""
Demo: Using the Falling Circles Environment Module

This script demonstrates how other scripts can use the extracted
environment module to generate videos with consistent physics.

The environment module provides:
1. Consistent physics simulation across all scripts
2. Reusable environment objects
3. Same API as original falling_circles.py
4. Clean separation of concerns
"""

import sys
import os
from PIL import Image

# Import the environment module
from falling_circles_env import FallingCirclesEnvironment, VideoParams, generate_falling_circles_video

def demo_basic_usage():
    """Demonstrate basic usage of the environment module"""
    print("🎬 Demo 1: Basic Environment Usage")
    print("=" * 40)
    
    # Create parameters
    params = VideoParams(
        width=300,
        height=300,
        num_frames=50,
        hole_diameter=30,
        wind_strength=2.0,
        num_circles=8,
        spawn_rate=0.3
    )
    
    # Method 1: Use compatibility function (same as original API)
    print("Method 1: Compatibility function")
    frames1, metadata1 = generate_falling_circles_video(params, seed=42)
    print(f"  Generated {len(frames1)} frames, jam type: {metadata1['actual_jam_type']}")
    
    # Method 2: Use environment object directly  
    print("Method 2: Environment object")
    env = FallingCirclesEnvironment(params)
    frames2, metadata2 = env.generate_video(seed=42)
    print(f"  Generated {len(frames2)} frames, jam type: {metadata2['actual_jam_type']}")
    
    # Verify they produce identical results
    frames_identical = len(frames1) == len(frames2)
    print(f"  Results identical: {frames_identical}")
    
    return frames1, metadata1

def demo_environment_reuse():
    """Demonstrate reusing the same environment for multiple videos"""
    print("\n🔄 Demo 2: Environment Reuse")
    print("=" * 40)
    
    # Create one environment
    params = VideoParams(width=200, height=200, num_frames=30, hole_diameter=40, num_circles=5)
    env = FallingCirclesEnvironment(params)
    
    # Generate multiple videos with different seeds
    results = []
    for i, seed in enumerate([42, 123, 456]):
        frames, metadata = env.generate_video(seed=seed)
        jam_type = metadata['actual_jam_type']
        exit_count = metadata['exit_statistics']['total_exited']
        
        print(f"  Video {i+1} (seed={seed}): {jam_type}, {exit_count} exits")
        results.append((frames, metadata))
        
        # Reset environment for next video
        env.reset()
    
    return results

def demo_parameter_variations():
    """Demonstrate systematic parameter variation"""
    print("\n⚙️  Demo 3: Parameter Variations")
    print("=" * 40)
    
    base_params = {
        'width': 200,
        'height': 200, 
        'num_frames': 25,
        'wind_strength': 1.0,
        'num_circles': 6,
        'spawn_rate': 0.3
    }
    
    # Test different hole sizes
    hole_sizes = [20, 40, 60]
    print("Testing hole size effects:")
    
    for hole_size in hole_sizes:
        params = VideoParams(**base_params, hole_diameter=hole_size)
        frames, metadata = generate_falling_circles_video(params, seed=42)
        
        jam_type = metadata['actual_jam_type']
        exit_ratio = metadata['exit_statistics']['exit_ratio']
        
        print(f"  Hole {hole_size}px: {jam_type} (exit ratio: {exit_ratio:.2f})")

def demo_custom_physics():
    """Demonstrate custom physics scenarios"""
    print("\n⚗️  Demo 4: Custom Physics Scenarios")
    print("=" * 40)
    
    scenarios = [
        {
            'name': 'Calm Environment',
            'params': VideoParams(width=200, height=200, num_frames=30, 
                                hole_diameter=50, wind_strength=0.0, num_circles=4)
        },
        {
            'name': 'Windy Environment', 
            'params': VideoParams(width=200, height=200, num_frames=30,
                                hole_diameter=30, wind_strength=5.0, num_circles=6)
        },
        {
            'name': 'Crowded Environment',
            'params': VideoParams(width=200, height=200, num_frames=40,
                                hole_diameter=25, wind_strength=1.0, num_circles=15, spawn_rate=0.6)
        }
    ]
    
    for scenario in scenarios:
        frames, metadata = generate_falling_circles_video(scenario['params'], seed=42)
        jam_type = metadata['actual_jam_type']
        total_spawned = metadata['exit_statistics']['total_spawned']
        total_exited = metadata['exit_statistics']['total_exited']
        
        print(f"  {scenario['name']}: {jam_type}")
        print(f"    Spawned: {total_spawned}, Exited: {total_exited}")

def demo_save_video():
    """Demonstrate saving generated video"""
    print("\n💾 Demo 5: Saving Generated Video")
    print("=" * 40)
    
    # Generate a video
    params = VideoParams(width=250, height=250, num_frames=40, hole_diameter=35, 
                        wind_strength=2.5, num_circles=8)
    frames, metadata = generate_falling_circles_video(params, seed=42, include_labels=True)
    
    # Save as GIF
    output_file = "environment_demo.gif"
    pil_frames = [Image.fromarray(frame.astype('uint8')) for frame in frames]
    pil_frames[0].save(
        output_file,
        save_all=True,
        append_images=pil_frames[1:],
        duration=100,
        loop=0
    )
    
    print(f"  ✅ Saved video: {output_file}")
    print(f"     Frames: {len(frames)}")
    print(f"     Jam type: {metadata['actual_jam_type']}")
    print(f"     Exit statistics: {metadata['exit_statistics']}")

def main():
    """Run all demos"""
    print("🎯 Falling Circles Environment Module - Demo Suite")
    print("=" * 60)
    print("This demonstrates how any script can use the environment module")
    print("to generate videos with consistent physics simulation.\n")
    
    # Run demos
    demo_basic_usage()
    demo_environment_reuse()
    demo_parameter_variations()
    demo_custom_physics()
    demo_save_video()
    
    print("\n🎉 All demos completed successfully!")
    print("\nKey Benefits of the Environment Module:")
    print("  ✅ Consistent physics across all scripts")
    print("  ✅ Reusable environment objects")
    print("  ✅ Same API as original falling_circles.py")
    print("  ✅ Clean separation of physics and I/O logic")
    print("  ✅ Easy to test and modify")
    print("\nAny script can now import and use this environment!")

if __name__ == "__main__":
    main()