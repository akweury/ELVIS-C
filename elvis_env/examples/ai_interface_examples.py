#!/usr/bin/env python3
"""
Example Usage of ELVIS-C Interface for AI Models

This script demonstrates how AI models can easily use the ELVIS-C interface
to generate intervention videos and analyze causal effects.
"""

import os
import sys

# Add the interface to path
sys.path.append(os.path.dirname(__file__))
from interface import VideoInterface, InterventionInterface, quick_intervene, compare_parameters

def example_basic_intervention():
    """Example 1: Basic single parameter intervention"""
    print("=== Example 1: Basic Intervention ===")
    
    # Create interface
    interface = InterventionInterface(default_seed=42)
    
    # Perform intervention: increase hole diameter from default 40 to 80
    result = interface.intervene(
        intervention_target="hole_diameter",
        intervention_value=80,
        description="Doubling hole size to reduce traffic jams"
    )
    
    print(f"Effect magnitude: {result.effect_magnitude:.3f}")
    print(f"Effect description: {result.effect_description}")
    print(f"Baseline frames: {len(result.baseline_frames)}")
    print(f"Intervention frames: {len(result.intervention_frames)}")
    
    # Save as GIFs for inspection
    interface.video_interface.save_intervention_gifs(
        result, 
        "example1_baseline.gif", 
        "example1_intervention.gif"
    )
    print("Saved: example1_baseline.gif, example1_intervention.gif")
    return result

def example_multi_parameter_intervention():
    """Example 2: Multi-parameter intervention"""
    print("\n=== Example 2: Multi-Parameter Intervention ===")
    
    interface = InterventionInterface(default_seed=42)
    
    # Intervene on multiple parameters at once
    result = interface.multi_intervene(
        interventions={
            "hole_diameter": 60,      # Increase hole size
            "wind_strength": 4.0,     # Increase wind
            "num_circles": 8          # More circles
        },
        description="Creating more challenging scenario"
    )
    
    print(f"Effect magnitude: {result.effect_magnitude:.3f}")
    print(f"Effect description: {result.effect_description}")
    
    # Save results
    interface.video_interface.save_intervention_gifs(
        result,
        "example2_baseline.gif",
        "example2_intervention.gif"
    )
    print("Saved: example2_baseline.gif, example2_intervention.gif")
    return result

def example_parameter_exploration():
    """Example 3: Exploring parameter space"""
    print("\n=== Example 3: Parameter Exploration ===")
    
    interface = InterventionInterface(default_seed=42)
    
    # Explore different wind strengths
    wind_values = [0.5, 2.0, 4.0, 6.0]
    results = interface.explore_parameter(
        parameter_name="wind_strength",
        values=wind_values
    )
    
    print(f"Explored {len(wind_values)} different wind strengths:")
    for i, (value, result) in enumerate(zip(wind_values, results)):
        print(f"  Wind {value}: Effect magnitude {result.effect_magnitude:.3f}")
        
        # Save one example
        if i == 2:  # Save wind=4.0 case
            interface.video_interface.save_intervention_gifs(
                result,
                "example3_baseline.gif", 
                "example3_intervention.gif"
            )
    
    print("Saved: example3_baseline.gif, example3_intervention.gif (wind=4.0 case)")
    return results

def example_quick_functions():
    """Example 4: Using convenience functions"""
    print("\n=== Example 4: Quick Functions ===")
    
    # Quick single intervention
    result1 = quick_intervene("num_circles", 15, seed=42)
    print(f"Quick intervention: {result1.effect_description}")
    
    # Compare multiple values
    results2 = compare_parameters("circle_size_max", [8, 15, 25], seed=42)
    print(f"Compared {len(results2)} circle sizes:")
    for i, result in enumerate(results2):
        print(f"  Size {[8, 15, 25][i]}: {result.effect_magnitude:.3f}")
    
    return result1, results2

def example_custom_baseline():
    """Example 5: Custom baseline parameters"""
    print("\n=== Example 5: Custom Baseline ===")
    
    interface = InterventionInterface(default_seed=42)
    
    # Define custom baseline scenario
    custom_baseline = {
        "hole_diameter": 30,      # Smaller hole
        "wind_strength": 1.0,     # Less wind
        "num_circles": 10,        # More circles
        "spawn_rate": 2           # Faster spawning
    }
    
    # Intervention: just increase hole size
    result = interface.intervene(
        baseline_params=custom_baseline,
        intervention_target="hole_diameter",
        intervention_change=20,  # Add 20 to current size (30 -> 50)
        description="Relieving congestion by enlarging hole"
    )
    
    print(f"Custom baseline intervention: {result.effect_description}")
    print(f"Effect magnitude: {result.effect_magnitude:.3f}")
    
    # Save results
    interface.video_interface.save_intervention_gifs(
        result,
        "example5_baseline.gif",
        "example5_intervention.gif"
    )
    print("Saved: example5_baseline.gif, example5_intervention.gif")
    return result

def example_ai_model_workflow():
    """Example 6: Typical AI model workflow"""
    print("\n=== Example 6: AI Model Workflow ===")
    
    interface = InterventionInterface(default_seed=42)
    
    # Step 1: AI model analyzes current situation
    baseline_params = {
        "hole_diameter": 35,
        "num_circles": 12,
        "spawn_rate": 2
    }
    
    print("AI analyzing situation...")
    
    # Step 2: AI decides on intervention strategy
    # (In real use, this would be the AI's decision-making logic)
    if baseline_params["num_circles"] > 10:
        # Too many circles, try to help them exit
        intervention_strategy = "increase_hole_size"
        new_hole_size = baseline_params["hole_diameter"] * 1.5
    else:
        # Few circles, might try making it harder
        intervention_strategy = "decrease_hole_size"
        new_hole_size = baseline_params["hole_diameter"] * 0.8
    
    print(f"AI strategy: {intervention_strategy}")
    
    # Step 3: Execute intervention
    result = interface.intervene(
        baseline_params=baseline_params,
        intervention_target="hole_diameter",
        intervention_value=new_hole_size,
        description=f"AI strategy: {intervention_strategy}"
    )
    
    # Step 4: Analyze results
    print(f"Intervention result: {result.effect_description}")
    print(f"Success: Effect magnitude = {result.effect_magnitude:.3f}")
    
    # Step 5: AI could use this feedback for learning
    success = result.effect_magnitude > 0.1  # Threshold for "significant effect"
    print(f"AI assessment: {'Successful' if success else 'Minimal effect'}")
    
    return result

def show_parameter_info():
    """Show available parameters and their effects"""
    print("\n=== Available Parameters ===")
    
    interface = InterventionInterface()
    param_info = interface.get_parameter_info()
    
    for param_name, info in param_info.items():
        print(f"\n{param_name}:")
        print(f"  Description: {info['description']}")
        print(f"  Type: {info['type']}")
        print(f"  Range: {info['min']} - {info['max']}")
        print(f"  Default: {info['default']}")
        print(f"  Effect: {info['effect']}")

def main():
    """Run all examples"""
    print("🎬 ELVIS-C Interface Examples for AI Models")
    print("=" * 50)
    
    # Show available parameters
    show_parameter_info()
    
    # Run examples
    example_basic_intervention()
    example_multi_parameter_intervention()
    example_parameter_exploration()
    example_quick_functions()
    example_custom_baseline()
    example_ai_model_workflow()
    
    print("\n✅ All examples completed!")
    print("Check the generated .gif files to see the interventions in action.")
    print("\nFor AI models, the key interfaces are:")
    print("  - InterventionInterface.intervene() for single parameter changes")
    print("  - InterventionInterface.multi_intervene() for multiple parameters")
    print("  - quick_intervene() for simple one-line interventions")

if __name__ == "__main__":
    main()