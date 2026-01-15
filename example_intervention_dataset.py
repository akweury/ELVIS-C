#!/usr/bin/env python3
"""
Example usage of the updated Two Parts dataset generation with intervention videos
"""

import sys
import os
sys.path.append('elvis_env/src')

from generate_two_parts_dataset import TwoPartsDatasetGenerator

def main():
    """
    Generate a Two Parts dataset with both observation and intervention videos
    """
    print("🎬 Generating Two Parts Dataset with Interventions")
    print("=" * 50)
    
    # Create dataset generator
    output_dir = "data/two_parts_with_interventions"
    generator = TwoPartsDatasetGenerator(output_dir)
    
    # Generate dataset with both observation and intervention videos
    generator.generate_dataset(
        num_videos=10,              # 10 observation videos  
        num_intervention_videos=5,  # 5 intervention videos
        workers=1,
        create_gifs=True
    )
    
    print(f"\n✅ Dataset generated successfully!")
    print(f"📁 Location: {output_dir}")
    print(f"\n📊 Dataset structure:")
    print(f"   🔍 observation/    - Normal videos (left→down, right→up)")
    print(f"   🔬 intervention/   - Intervention videos (some objects move in reverse)")
    print(f"   🎬 visualization/  - GIF previews with labels")
    print(f"   📈 metadata/       - Statistics and analysis")
    print(f"   📋 index/          - Sample indices")
    print(f"   🎯 splits/         - Train/val/test splits")
    
    print(f"\n🔬 In intervention videos:")
    print(f"   • 1-2 left objects move UP (instead of down)")
    print(f"   • 1-2 right objects move DOWN (instead of up)")
    print(f"   • Objects look identical to observation videos")
    print(f"   • Intervention behavior is stored in metadata")

if __name__ == "__main__":
    main()