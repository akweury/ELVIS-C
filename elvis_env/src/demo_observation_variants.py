#!/usr/bin/env python3
"""
Demo script for testing observation variants
"""

from configuration_manager import ConfigurationManager


def demo_observation_variants():
    """Demo the new observation variants feature"""
    print("🎭 Testing Observation Variants Feature")
    print("=" * 50)
    
    manager = ConfigurationManager()
    
    # Generate observation variants for red_blue_colors configuration
    print("🔴🔵 Generating red_blue_colors observation variants...")
    success = manager.generate_observation_variants('red_blue_colors', num_videos_per_variant=3)
    
    if success:
        print("\n✅ Successfully generated observation variants!")
        print("\nGenerated structure:")
        print("data/red_blue_colors/")
        print("├── observation_1/     # Standard: red left↓, blue right↑")
        print("├── observation_2/     # Mixed: some cross-placed")
        print("└── observation_3/     # Heavy mixing: most cross-placed")
        print("\nEach folder contains:")
        print("├── observation_X_XXXXXX_XXXXXX/")
        print("│   ├── frames/         # Video frames")
        print("│   ├── meta.json       # Metadata")
        print("│   └── config.yaml     # Configuration used")
        print("└── visualization/      # GIF previews")
    else:
        print("❌ Failed to generate observation variants")


if __name__ == "__main__":
    demo_observation_variants()