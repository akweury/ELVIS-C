import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from interface import generate_intervention_video
from PIL import Image

def save_frames_as_gif(frames, output_path, duration=100):
    """Save video frames as an animated GIF"""
    if not frames:
        print(f"⚠️  Warning: No frames to save for {output_path}")
        return
        
    # Convert numpy arrays to PIL Images
    pil_frames = [Image.fromarray(frame.astype('uint8')) for frame in frames]
    
    if not pil_frames:
        print(f"⚠️  Warning: No valid frames for {output_path}")
        return
    
    # Save as GIF
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0
    )
    print(f"✅ Saved: {output_path}")

def test_intervention(name, params, target, value, description):
    """Test a single intervention and save results"""
    print(f"\n🎬 Test: {name}")
    print("=" * 50)
    print(f"Description: {description}")
    print(f"Intervention: {target} → {value}")
    
    try:
        # Generate intervention
        result = generate_intervention_video(
            baseline_parameters=params,
            intervention_target=target,
            intervention_value=value,
            seed=42
        )
        
        # Check if generation was successful
        if not result.success:
            print(f"❌ Generation failed: {result}")
            return None
        
        # Create output directory
        output_dir = f"ai_test_{name.lower().replace(' ', '_')}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save videos
        baseline_path = os.path.join(output_dir, "baseline.gif")
        intervention_path = os.path.join(output_dir, "intervention.gif")
        
        save_frames_as_gif(result.baseline_result.frames, baseline_path)
        save_frames_as_gif(result.intervention_result.frames, intervention_path)
        
        # Print results
        print(f"Success: {result.success}")
        print(f"Effect: {result.effect_description}")
        print(f"Baseline outcome: {result.baseline_result.metadata.get('actual_jam_type', 'unknown')}")
        print(f"Intervention outcome: {result.intervention_result.metadata.get('actual_jam_type', 'unknown')}")
        print(f"Videos saved in: {output_dir}/")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in test {name}: {e}")
        return None

def main():
    """Run comprehensive AI interface tests"""
    print("🤖 AI Model Interface - Comprehensive Test Suite")
    print("=" * 60)
    
    # Test 1: Small hole → Big hole (should reduce jams)
    small_hole_params = {
        "hole_diameter": 25,      # Small hole
        "wind_strength": 1.0,     
        "num_circles": 12,        # Many circles
        "spawn_rate": 0.4,        # Fast spawning
        "width": 400,
        "height": 400,
        "num_frames": 80
    }
    
    test_intervention(
        name="Hole_Enlargement",
        params=small_hole_params,
        target="hole_diameter",
        value=80,
        description="AI enlarges small hole to reduce traffic jams"
    )
    
    # Test 2: No wind → Strong wind (should affect circle movement)
    calm_params = {
        "hole_diameter": 40,
        "wind_strength": 0.0,     # No wind
        "num_circles": 8,
        "spawn_rate": 0.3,
        "width": 400,
        "height": 400,
        "num_frames": 80
    }
    
    test_intervention(
        name="Wind_Introduction",
        params=calm_params,
        target="wind_strength",
        value=6.0,
        description="AI introduces strong wind to affect circle trajectories"
    )
    
    # Test 3: Few circles → Many circles (should create jams)
    few_circles_params = {
        "hole_diameter": 35,
        "wind_strength": 2.0,
        "num_circles": 3,         # Few circles
        "spawn_rate": 0.2,        # Slow spawning
        "width": 400,
        "height": 400,
        "num_frames": 80
    }
    
    test_intervention(
        name="Circle_Increase",
        params=few_circles_params,
        target="num_circles", 
        value=15,
        description="AI increases circle count to create congestion"
    )
    
    # Test 4: Large circles → Small circles (should fit better)
    large_circles_params = {
        "hole_diameter": 40,
        "wind_strength": 2.0,
        "num_circles": 8,
        "circle_size_min": 12,    # Large circles
        "circle_size_max": 18,
        "spawn_rate": 0.3,
        "width": 400,
        "height": 400,
        "num_frames": 80
    }
    
    test_intervention(
        name="Circle_Shrinking",
        params=large_circles_params,
        target="circle_size_max",
        value=8,
        description="AI reduces circle size to improve flow"
    )
    
    print("\n🎯 All tests completed!")
    print("Check the generated directories for before/after GIF comparisons:")
    print("  - ai_test_hole_enlargement/")
    print("  - ai_test_wind_introduction/")
    print("  - ai_test_circle_increase/")
    print("  - ai_test_circle_shrinking/")

if __name__ == "__main__":
    main()