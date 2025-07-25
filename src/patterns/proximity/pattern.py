# Created by MacBook Pro at 15.07.25

from src.utils.generators import create_tasks_v3
from src.config import size_list, prin_in_neg
from src.patterns.proximity.proximity_avoidance import proximity_avoidance_task_fn
from src.patterns.proximity.proximity_group_trigger import non_overlap_scatter_cluster
from src.patterns.proximity.proximity_leader_trigger import proximity_leader_trigger_task_fn

def register_tasks():
    tasks = {}
    # tasks.update(create_tasks_v3(task_func=proximity_leader_trigger_task_fn, size_list=size_list, pin=prin_in_neg))
    # tasks.update(create_tasks_v3(task_func=proximity_avoidance_task_fn, size_list=size_list, pin=prin_in_neg))
    # tasks.update(create_tasks_v3(task_func=non_overlap_scatter_cluster, size_list=size_list, pin=prin_in_neg))
    return tasks
