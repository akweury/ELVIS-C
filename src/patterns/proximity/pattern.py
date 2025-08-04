# Created by MacBook Pro at 15.07.25

from src.utils.generators import create_tasks_v3
from src.config import size_list, prin_in_neg
from src.patterns.proximity.proximity_avoidance import proximity_avoidance_task_fn
from src.patterns.proximity.proximity_group_trigger import non_overlap_scatter_cluster
from src.patterns.proximity.proximity_leader_trigger import proximity_leader_trigger_task_fn
from src import config


def register_tasks(lite=False):
    if lite:
        obj_quantity_list = ["s"]
        grp_num_range = range(2, 3)
        prop_list = ["shape", "color"]
    else:
        obj_quantity_list = list(config.standard_quantity_dict.keys())[:3]
        grp_num_range = range(2, 3)
        prop_list = ["shape", "color", "size", "count"]

    all_tasks = []
    all_names = []

    tasks, names = create_tasks_v3(func=proximity_leader_trigger_task_fn, params=prop_list, grp_nums=grp_num_range, obj_quantity_list=obj_quantity_list)
    all_tasks.extend(tasks)
    all_names.extend(names)

    # tasks.update(create_tasks_v3(task_func=proximity_avoidance_task_fn, size_list=size_list, pin=prin_in_neg))
    # tasks.update(create_tasks_v3(task_func=non_overlap_scatter_cluster, size_list=size_list, pin=prin_in_neg))
    return all_tasks, all_names
