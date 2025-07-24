# Created by MacBook Pro at 14.07.25
from pathlib import Path
from src.task_registry import load_task_modules_from_patterns
from rtpt import RTPT
import random
import time
def main():
    tasks = load_task_modules_from_patterns()

    rtpt = RTPT(name_initials='JS', experiment_name='ELVIS-C_Generation', max_iterations=len(tasks))
    # Start the RTPT tracking
    rtpt.start()
    for principle, principle_tasks in tasks.items():
        for name, task_fn in principle_tasks.items():
            output_path = Path("video_tasks") / principle / "train" / name
            print(f"Generating task: {name}")
            task_fn(output_dir=str(output_path), num_pos=3, num_neg=3)
            print(f"Saved to: {output_path}")
        for name, task_fn in principle_tasks.items():
            output_path = Path("video_tasks") / principle / "test" / name
            print(f"Generating task: {name}")
            task_fn(output_dir=str(output_path), num_pos=3, num_neg=3)
            print(f"Saved to: {output_path}")

        # Update the RTPT (subtitle is optional)
        rtpt.step(subtitle=f"{principle} tasks completed")

if __name__ == "__main__":
    main()
