# Created by MacBook Pro at 21.07.25


import numpy as np
import random

def assign_group_objects(count, shape_options, color_options, size_options, cs, cc, cz, x_range, y_range):
    """Assigns properties to a group of objects based on flags."""
    group_shape = random.choice(shape_options) if cs else None
    group_color = random.choice(color_options) if cc else None
    group_size = random.choice(size_options) if cz else None
    objs = []
    for _ in range(count):
        obj_shape = group_shape if cs else random.choice(shape_options)
        obj_color = group_color if cc else random.choice(color_options)
        obj_size = group_size if cz else random.choice(size_options)
        pos = np.array([np.random.uniform(*x_range), np.random.uniform(*y_range)])
        objs.append({'pos': pos, 'shape': obj_shape, 'color': obj_color, 'size': obj_size})
    return objs


def get_converge_positions(center, count, radius=0.1):
    """Returns final positions for convergence in a circle around center."""
    angle_step = 2 * np.pi / count
    return [center + radius * np.array([np.cos(i * angle_step), np.sin(i * angle_step)]) for i in range(count)]

def jitter_position(pos, scale=0.01):
    """Returns a jittered position, clipped to [0, 1]."""
    jitter = scale * np.random.randn(2)
    return np.clip(pos + jitter, 0, 1)


