# Created by MacBook Pro at 23.07.25

from utils.generators import create_tasks_v3
from config import size_list, prin_in_neg

from .continuity_chain import continuity_chain_task_fn
from src.patterns.continuity.continuity_smooth_path import continuity_smooth_path_task_fn

def register_tasks():
    tasks = {}
    tasks.update(create_tasks_v3(task_func=continuity_smooth_path_task_fn, size_list=size_list, pin=prin_in_neg))
    # tasks.update(create_tasks_v3(task_func=continuity_chain_task_fn, size_list=size_list, pin=prin_in_neg))
    return tasks
