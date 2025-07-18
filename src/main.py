# Created by MacBook Pro at 14.07.25
from pathlib import Path
from task_registry import load_task_modules_from_patterns


def main():
    tasks = load_task_modules_from_patterns()
    for name, task_fn in tasks.items():
        output_path = Path("video_tasks") / name
        print(f"Generating task: {name}")
        task_fn(output_dir=str(output_path), num_pos=10, num_neg=10)
        print(f"Saved to: {output_path}")



if __name__ == "__main__":
    main()
