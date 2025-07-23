# Created by MacBook Pro at 14.07.25
from pathlib import Path
from task_registry import load_task_modules_from_patterns


def main():
    tasks = load_task_modules_from_patterns()
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

if __name__ == "__main__":
    main()
