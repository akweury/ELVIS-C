#!/usr/bin/env python3
"""
Test script to verify intervention video generation
"""

import sys
import os
sys.path.append('elvis_env/src')

from generate_two_parts_dataset import TwoPartsDatasetGenerator

def test_intervention_generation():
    """Test that intervention videos can be generated"""
    print("🧪 Testing intervention video generation...")
    
    # Create a small dataset to test
    output_dir = "data/test_intervention"
    generator = TwoPartsDatasetGenerator(output_dir)
    
    # Generate a small test dataset: 2 observation + 2 intervention videos
    generator.generate_dataset(
        num_videos=2,
        num_intervention_videos=2, 
        workers=1,
        create_gifs=True
    )
    
    print("✅ Test completed successfully!")
    print(f"📁 Check output in: {output_dir}")
    print("🎬 Intervention videos show objects with reversed movement patterns (stored in metadata)")

if __name__ == "__main__":
    test_intervention_generation()