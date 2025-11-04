#!/usr/bin/env python3
"""
AI Model Workflow Example

This example demonstrates the intended workflow for AI models:
1. AI perceives parameters from existing video dataset
2. AI decides on intervention strategy  
3. AI requests intervention videos with specific parameters
4. Package generates videos and returns them to AI
"""

import sys
import os
sys.path.append('..')

from interface.ai_model_interface import AIVideoInterface, generate_intervention_video


def simulate_ai_perception():
    """
    Simulate AI model perceiving parameters from an existing video
    
    In practice, the AI would analyze video frames to extract these parameters.
    Here we simulate what the AI might perceive.
    """
    print("🔍 AI Model: Analyzing existing video...")
    
    # Simulated perception results
    perceived_params = {
        "hole_diameter": 35,        # AI detected small hole
        "wind_strength": 1.5,       # AI detected light wind  
        "num_circles": 8,           # AI counted circles
        "circle_size_min": 6,       # AI measured circle sizes
        "circle_size_max": 10,
        "spawn_rate": 0.4,          # AI observed spawning rate
        "width": 400,               # Video dimensions
        "height": 400,
        "num_frames": 80
    }
    
    print("✅ AI Model: Perception complete!")
    print("   Detected parameters:", {k: v for k, v in perceived_params.items() if k in ['hole_diameter', 'wind_strength', 'num_circles']})
    
    return perceived_params


def ai_intervention_decision(perceived_params):
    """
    Simulate AI model deciding on intervention strategy
    
    Based on perceived parameters, AI decides what intervention might be interesting.
    """
    print("\n🧠 AI Model: Analyzing situation and planning intervention...")
    
    hole_size = perceived_params["hole_diameter"]
    num_circles = perceived_params["num_circles"]
    
    # AI's decision logic
    if hole_size < 40 and num_circles > 5:
        # Small hole + many circles = likely jam
        intervention_strategy = "increase_hole_size"
        target_value = hole_size * 2  # Double the hole size
        reasoning = f"Small hole ({hole_size}) + many circles ({num_circles}) → likely traffic jam. Try larger hole."
    elif hole_size > 60:
        # Large hole = too easy, make it harder
        intervention_strategy = "increase_difficulty"
        target_value = hole_size * 0.7  # Reduce hole size
        reasoning = f"Large hole ({hole_size}) → too easy. Try smaller hole for more interesting dynamics."
    else:
        # Moderate scenario, try wind intervention
        intervention_strategy = "add_wind"
        target_value = 4.0
        reasoning = f"Moderate scenario. Try adding wind to see effect on circle trajectories."
    
    print(f"💡 AI Decision: {intervention_strategy}")
    print(f"   Reasoning: {reasoning}")
    
    if intervention_strategy in ["increase_hole_size", "increase_difficulty"]:
        return "hole_diameter", target_value
    else:
        return "wind_strength", target_value


def main():
    """Main AI model workflow"""
    print("🤖 AI Model Workflow: Parameter-Based Video Generation")
    print("=" * 60)
    
    # Step 1: AI perceives parameters from existing video
    perceived_params = simulate_ai_perception()
    
    # Step 2: AI decides on intervention
    intervention_target, intervention_value = ai_intervention_decision(perceived_params)
    
    # Step 3: AI requests intervention videos from ELVIS-C interface
    print(f"\n📨 AI Model: Requesting intervention videos...")
    print(f"   Target: {intervention_target} = {intervention_value}")
    
    interface = AIVideoInterface()
    
    result = interface.quick_intervention(
        baseline_parameters=perceived_params,
        intervention_target=intervention_target,
        intervention_value=intervention_value,
        seed=42  # For reproducible results
    )
    
    # Step 4: AI receives and analyzes results
    if result.success:
        print("✅ AI Model: Received intervention videos successfully!")
        print(f"   Baseline frames: {len(result.baseline_result.frames)}")
        print(f"   Intervention frames: {len(result.intervention_result.frames)}")
        print(f"   Effect: {result.effect_description}")
        
        # Save videos for inspection
        saved_files = interface.save_videos(
            result, 
            output_dir="ai_experiment_outputs",
            prefix="ai_intervention"
        )
        print(f"   Saved videos: {list(saved_files.keys())}")
        
        # AI analyzes the results
        baseline_outcome = result.baseline_result.metadata.get('actual_jam_type', 'unknown')
        intervention_outcome = result.intervention_result.metadata.get('actual_jam_type', 'unknown')
        
        print(f"\n🔬 AI Analysis:")
        print(f"   Baseline outcome: {baseline_outcome}")
        print(f"   Intervention outcome: {intervention_outcome}")
        
        if baseline_outcome != intervention_outcome:
            print(f"   ✨ Intervention caused outcome change!")
            print(f"   📚 AI Learning: {intervention_target}={intervention_value} → {baseline_outcome} to {intervention_outcome}")
        else:
            print(f"   📊 Same outcome, but dynamics may have changed")
            print(f"   📚 AI Learning: {intervention_target}={intervention_value} → no outcome change")
            
    else:
        print("❌ Error generating intervention videos")
        return
    
    print("\n" + "=" * 60)
    print("🎯 Workflow Complete!")
    print("\nKey points for AI models:")
    print("1. Perceive parameters from existing videos")
    print("2. Decide on intervention strategy") 
    print("3. Request specific parameter changes")
    print("4. Receive generated videos for analysis")
    print("5. Learn from intervention results")


def example_batch_interventions():
    """Example of AI testing multiple intervention hypotheses"""
    print("\n\n🔬 Advanced Example: AI Testing Multiple Hypotheses")
    print("=" * 60)
    
    # AI's perceived baseline
    baseline = {
        "hole_diameter": 30,
        "wind_strength": 0.0,
        "num_circles": 12,
        "width": 400,
        "height": 400,
        "num_frames": 60
    }
    
    # AI wants to test multiple hypotheses
    hypotheses = [
        ("hole_diameter", 60, "Larger hole reduces jams"),
        ("wind_strength", 3.0, "Wind helps circle flow"),
        ("num_circles", 6, "Fewer circles prevents jams"),
        ("spawn_rate", 0.15, "Slower spawning reduces congestion")
    ]
    
    interface = AIVideoInterface()
    results = []
    
    print("🧪 AI testing hypotheses...")
    
    for param, value, hypothesis in hypotheses:
        print(f"\n   Testing: {hypothesis}")
        print(f"   Intervention: {param} = {value}")
        
        result = interface.quick_intervention(
            baseline_parameters=baseline,
            intervention_target=param,
            intervention_value=value,
            seed=42
        )
        
        if result.success:
            baseline_jam = result.baseline_result.metadata.get('actual_jam_type', 'unknown')
            intervention_jam = result.intervention_result.metadata.get('actual_jam_type', 'unknown')
            
            outcome_changed = baseline_jam != intervention_jam
            print(f"   Result: {baseline_jam} → {intervention_jam} {'✓' if outcome_changed else '='}")
            
            results.append({
                'hypothesis': hypothesis,
                'param': param,
                'value': value,
                'outcome_changed': outcome_changed,
                'baseline_jam': baseline_jam,
                'intervention_jam': intervention_jam
            })
        else:
            print(f"   ❌ Failed to generate videos")
    
    # AI analyzes all results
    print(f"\n📊 AI Analysis Summary:")
    successful_interventions = [r for r in results if r['outcome_changed']]
    
    if successful_interventions:
        print(f"   ✨ {len(successful_interventions)} successful interventions:")
        for r in successful_interventions:
            print(f"     • {r['hypothesis']}: {r['baseline_jam']} → {r['intervention_jam']}")
    else:
        print(f"   📈 No outcome changes, but dynamics may differ")
    
    print(f"\n🧠 AI Learning: Tested {len(hypotheses)} causal hypotheses")


def example_parameter_validation():
    """Example of AI validating parameters before video generation"""
    print("\n\n✅ Parameter Validation Example")
    print("=" * 40)
    
    interface = AIVideoInterface()
    
    # Example of AI checking if its perceived parameters are valid
    ai_perceived_params = {
        "hole_diameter": 45,
        "wind_strength": 2.5,
        "num_circles": 7,
        "invalid_param": 999  # AI made an error
    }
    
    print("🔍 AI validating perceived parameters...")
    is_valid, message = interface.validate_parameters(ai_perceived_params)
    
    if is_valid:
        print("✅ Parameters are valid")
    else:
        print(f"⚠️  Parameter issue: {message}")
        print("🔧 AI can fix parameters or use template")
        
        # Get parameter template to understand expected format
        template = interface.get_parameter_template()
        print(f"📋 Available parameters: {list(template.keys())}")


if __name__ == "__main__":
    main()
    example_batch_interventions()
    example_parameter_validation()