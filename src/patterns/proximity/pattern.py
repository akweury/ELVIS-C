# Created by MacBook Pro at 15.07.25

from utils.generators import create_tasks_v3
from config import size_list, prin_in_neg
from src import config

from .proximity_group_trigger import non_overlap_scatter_cluster

def register_tasks():
    tasks = {}
    tasks.update(create_tasks_v3(
        task_func=non_overlap_scatter_cluster,
        variation_keys=["color", "shape", "size"],
        variant_range=range(1, 4),
        size_list=size_list,
        pin=prin_in_neg
    ))
    return tasks
