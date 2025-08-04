# Created by MacBook Pro at 14.07.25
from pathlib import Path
from src.task_registry import load_task_modules_from_patterns
from rtpt import RTPT
from src import config
import random
import time
from tqdm import tqdm
from shutil import rmtree
import argparse
from src.utils.generators import combine_gifs


def cleanup_old_videos(path):
    # remove the existing video_tasks directory if it exists

    for item in path.iterdir():
        if item.is_dir():
            rmtree(item)


def main(args):
    tasks = load_task_modules_from_patterns()

    rtpt = RTPT(name_initials='JS', experiment_name='ELVIS-C_Gen', max_iterations=len(tasks))
    # Start the RTPT tracking
    rtpt.start()
    print(config.root)

    for principle, principle_tasks in tasks.items():
        cleanup_old_videos(config.root / "video_tasks" / principle)
        for name, task_fn in tqdm(principle_tasks.items()):
            # Update the RTPT (subtitle is optional)
            rtpt.step()
            print(f"Generating task: {name}")
            train_path = config.root / "video_tasks" / principle / "train" / name
            test_path = config.root / "video_tasks" / principle / "test" / name
            for e_i in range(config.get_example_num(args.lite)):
                task_fn(e_i, train_path)
                task_fn(e_i, test_path)
            combine_gifs(train_path / "positive", config.root / "video_tasks" / principle / f"{name}.gif")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline models with CUDA support.")
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--lite", action="store_true")
    args = parser.parse_args()

    main(args)
