# Created by MacBook Pro at 14.07.25
from pathlib import Path
from src.task_registry import load_task_modules_from_patterns
from rtpt import RTPT
from src import config
import random
import time
from tqdm import tqdm

def main():
    tasks = load_task_modules_from_patterns()

    rtpt = RTPT(name_initials='JS', experiment_name='ELVIS-C_Gen', max_iterations=len(tasks))
    # Start the RTPT tracking
    rtpt.start()
    print(config.root)
    for principle, principle_tasks in tqdm(tasks.items()):
        for name, task_fn in principle_tasks.items():
            output_path = config.root / "video_tasks" / principle / "train" / name
            print(f"Generating task: {name}")
            task_fn(output_dir=str(output_path), num_pos=3, num_neg=3)
            print(f"Saved to: {output_path}")
        for name, task_fn in principle_tasks.items():
            output_path = config.root / "video_tasks" / principle / "test" / name
            print(f"Generating task: {name}")
            task_fn(output_dir=str(output_path), num_pos=3, num_neg=3)
            print(f"Saved to: {output_path}")

        # Update the RTPT (subtitle is optional)
        rtpt.step(subtitle=f"{principle}")


if __name__ == "__main__":
    main()
