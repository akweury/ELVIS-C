import sys
import os
sys.path.append(os.path.dirname(__file__))
from ai_model_interface import generate_intervention_video
from PIL import Image

def save_frames_as_gif(frames, output_path, duration=100):
    """Save video frames as an animated GIF for visualization"""
    # Convert numpy arrays to PIL Images
    pil_frames = [Image.fromarray(frame.astype('uint8')) for frame in frames]
    
    # Save as GIF
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0
    )
    print(f"✅ Saved: {output_path}")

# Step 1: AI perceives parameters from existing video
# (In practice, AI would extract these from analyzing a real video)
ai_perceived_params = {
    "hole_diameter": 35,      # AI detected hole size
    "wind_strength": 1.5,     # AI detected wind strength
    "num_circles": 8,         # AI counted maximum circles
    "circle_size_min": 6,     # AI estimated circle sizes
    "circle_size_max": 10,
    "spawn_rate": 0.35,       # AI estimated spawn frequency
    "width": 400,             # Video dimensions  
    "height": 400,
    "num_frames": 40          # Desired video length (updated to 40 frames)
}

print("🎬 AI Model Intervention Test")
print("=" * 40)
print("Scenario: AI perceives a video with moderate traffic jams")
print("AI Strategy: Enlarge hole to improve circle flow")
print()
print(f"📊 Perceived parameters:")
for key, value in ai_perceived_params.items():
    print(f"  {key}: {value}")
print()
print(f"🎯 Intervention: hole_diameter {ai_perceived_params['hole_diameter']} → 70")

# Step 2: AI decides on intervention
# AI reasoning: "I see circles having difficulty exiting through the small hole.
# I will double the hole size to reduce traffic jams."
result = generate_intervention_video(
    baseline_parameters=ai_perceived_params,
    intervention_target="hole_diameter", 
    intervention_value=70,    # AI's chosen intervention
    seed=42                   # For reproducible results
)

# Step 2.5: Generate labeled versions for GIF visualization
import sys
import os
sys.path.append(os.path.dirname(__file__))
from ai_model_interface import AIVideoInterface
interface = AIVideoInterface()
labeled_result = interface.quick_intervention(
    baseline_parameters=ai_perceived_params,
    intervention_target="hole_diameter",
    intervention_value=70,
    seed=42,
    include_labels=True  # Get labeled frames for GIF
)

# Step 3: AI receives the generated videos
baseline_video = result.baseline_result.frames     # Original scenario
intervention_video = result.intervention_result.frames  # After intervention

# Labeled versions for visualization
baseline_video_labeled = labeled_result.baseline_result.frames
intervention_video_labeled = labeled_result.intervention_result.frames

print(f"\n🎬 Generated videos:")
print(f"  Baseline: {len(baseline_video)} frames")
print(f"  Intervention: {len(intervention_video)} frames")

# Step 4: Save videos for human inspection/AI analysis
output_dir = "ai_intervention_test"
os.makedirs(output_dir, exist_ok=True)

baseline_path = os.path.join(output_dir, "baseline_video.gif")
intervention_path = os.path.join(output_dir, "intervention_video.gif")

save_frames_as_gif(baseline_video_labeled, baseline_path)
save_frames_as_gif(intervention_video_labeled, intervention_path)

# Step 5: AI analyzes the intervention results
print("\n📊 Intervention Analysis:")
print(f"  Success: {result.success}")
print(f"  Changes made: {result.parameters_changed}")
print(f"  Effect: {result.effect_description}")
print(f"  Baseline outcome: {result.baseline_result.metadata.get('actual_jam_type', 'unknown')}")
print(f"  Intervention outcome: {result.intervention_result.metadata.get('actual_jam_type', 'unknown')}")

# Step 6: AI can now use this information for learning
baseline_outcome = result.baseline_result.metadata.get('actual_jam_type', 'unknown')
intervention_outcome = result.intervention_result.metadata.get('actual_jam_type', 'unknown')

if baseline_outcome != intervention_outcome:
    print(f"\n🎉 AI Success: Intervention changed outcome!")
    print(f"   Before: {baseline_outcome} → After: {intervention_outcome}")
else:
    print(f"\n🤔 AI Learning: No outcome change detected")
    print(f"   Both scenarios resulted in: {baseline_outcome}")

print(f"\n📁 Videos saved for inspection:")
print(f"  📂 {output_dir}/")
print(f"    📄 baseline_video.gif")
print(f"    📄 intervention_video.gif")
print(f"\n💡 Open these GIF files to see the intervention effect!")
print(f"   The AI can now use this visual feedback for learning!")