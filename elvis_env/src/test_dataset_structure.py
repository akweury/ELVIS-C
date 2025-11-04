#!/usr/bin/env python3
"""Test script to verify the falling_circles dataset structure matches falling_circles_v1"""

import os
import sys
from pathlib import Path
import json
import pandas as pd

def test_dataset_structure(dataset_path: str):
    """Test that the dataset structure matches falling_circles_v1"""
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return False
    
    print(f"🔍 Testing dataset structure: {dataset_path}")
    print("="*60)
    
    # Check main directories
    expected_dirs = [
        'observation', 'intervention', 'index', 'audits', 
        'visualization', 'splits', 'metadata', 'scenarios'
    ]
    
    missing_dirs = []
    for dir_name in expected_dirs:
        if not (dataset_path / dir_name).exists():
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    else:
        print("✅ All required directories present")
    
    # Check observation structure
    obs_dir = dataset_path / 'observation'
    obs_samples = list(obs_dir.iterdir())
    
    if not obs_samples:
        print("❌ No observation samples found")
        return False
    
    print(f"✅ Found {len(obs_samples)} observation samples")
    
    # Check first sample structure
    sample_dir = obs_samples[0]
    expected_files = ['frames', 'meta.json', 'stats.csv']
    
    for file_name in expected_files:
        if not (sample_dir / file_name).exists():
            print(f"❌ Missing {file_name} in sample {sample_dir.name}")
            return False
    
    print(f"✅ Sample structure correct for {sample_dir.name}")
    
    # Check frames directory
    frames_dir = sample_dir / 'frames'
    frames = list(frames_dir.glob('frame_*.png'))
    
    if len(frames) != 40:
        print(f"❌ Expected 40 frames, found {len(frames)}")
        return False
    
    print(f"✅ Correct number of frames: {len(frames)}")
    
    # Check meta.json format
    try:
        with open(sample_dir / 'meta.json', 'r') as f:
            meta = json.load(f)
        
        required_keys = ['sample_id', 'sample_type', 'params', 'generation_info', 'physics_simulation']
        for key in required_keys:
            if key not in meta:
                print(f"❌ Missing key '{key}' in meta.json")
                return False
        
        print("✅ meta.json format correct")
    except Exception as e:
        print(f"❌ Error reading meta.json: {e}")
        return False
    
    # Check index files
    index_dir = dataset_path / 'index'
    if not (index_dir / 'samples.csv').exists():
        print("❌ Missing samples.csv in index directory")
        return False
    
    try:
        df = pd.read_csv(index_dir / 'samples.csv')
        expected_columns = ['sample_id', 'sample_type', 'jam_type', 'exit_ratio']
        
        for col in expected_columns:
            if col not in df.columns:
                print(f"❌ Missing column '{col}' in samples.csv")
                return False
        
        print(f"✅ samples.csv format correct ({len(df)} samples)")
    except Exception as e:
        print(f"❌ Error reading samples.csv: {e}")
        return False
    
    # Check splits files
    splits_dir = dataset_path / 'splits'
    split_files = ['train_ids.txt', 'val_ids.txt', 'test_ids.txt']
    
    for split_file in split_files:
        if not (splits_dir / split_file).exists():
            print(f"❌ Missing split file: {split_file}")
            return False
    
    print("✅ Split files present")
    
    # Check visualization files
    viz_dir = dataset_path / 'visualization'
    gifs = list(viz_dir.glob('*.gif'))
    
    if len(gifs) != len(obs_samples):
        print(f"❌ Expected {len(obs_samples)} GIFs, found {len(gifs)}")
        return False
    
    print(f"✅ Visualization GIFs present: {len(gifs)}")
    
    print("\n🎉 All tests passed! Dataset structure matches falling_circles_v1 format")
    return True


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_dataset_structure.py <dataset_path>")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    success = test_dataset_structure(dataset_path)
    
    if not success:
        sys.exit(1)