# Created by MacBook Pro at 14.07.25

import importlib
from pathlib import Path


def load_task_modules_from_patterns():
    """Dynamically load all pattern modules from the `patterns` folder."""
    task_modules = {}
    base_path = Path(__file__).parent / "patterns"
    task_id = 0

    for principle_dir in base_path.iterdir():
        if principle_dir.is_dir() and (principle_dir / "pattern.py").exists():
            task_modules[principle_dir.name] = {}
            module_name = f"patterns.{principle_dir.name}.pattern"
            module = importlib.import_module(module_name)
            if hasattr(module, "register_tasks"):
                tasks = module.register_tasks()
                for name, task in tasks.items():
                    id_str = f"{task_id:04d}"
                    task_modules[principle_dir.name][f"{id_str}_{name}"] = task
                    task_id += 1
                    if task_id > 9999:
                        break
    return task_modules




