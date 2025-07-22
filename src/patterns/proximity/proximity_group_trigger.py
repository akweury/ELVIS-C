# Created by MacBook Pro at 15.07.25

import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import imageio

from utils.generators import draw_shape, combine_gifs
from utils import proximity_utils

from utils.proximity_utils import assign_group_objects, get_converge_positions, jitter_position

def generate_proximity_group_video(
    cs: bool, cc: bool, cz: bool,
    size: float, count: int, output_dir: str, frames: int = 20
):
    import random
    os.makedirs(output_dir, exist_ok=True)

    shape_options = ['circle', 'square', 'triangle']
    color_options = ['blue', 'green', 'orange']
    size_options = [size * 0.8, size, size * 1.2]

    move_left = np.random.rand() < 0.5
    left_center = np.array([0.25, 0.5])
    right_center = np.array([0.75, 0.5])

    left_objs = assign_group_objects(
        count, shape_options, color_options, size_options,
        cs, cc, cz, (0.1, 0.4), (0.3, 0.7)
    )
    right_objs = assign_group_objects(
        count, shape_options, color_options, size_options,
        cs, cc, cz, (0.6, 0.9), (0.3, 0.7)
    )

    if move_left:
        final_left = get_converge_positions(left_center, count, radius=0.1)
    else:
        final_right = get_converge_positions(right_center, count, radius=0.1)

    for t in range(frames):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        if move_left:
            # Left group converges
            for i, obj in enumerate(left_objs):
                target = final_left[i]
                alpha = min(1.0, t / 10)
                pos = (1 - alpha) * obj['pos'] + alpha * target
                draw_shape(ax, obj['shape'], pos, obj['size'], obj['color'] if t < 13 else "red")
            # Right group jitters
            for obj in right_objs:
                perturbed = jitter_position(obj['pos'], scale=0.01)
                draw_shape(ax, obj['shape'], perturbed, obj['size'], obj['color'])
        else:
            # Left group jitters
            for obj in left_objs:
                perturbed = jitter_position(obj['pos'], scale=0.01)
                draw_shape(ax, obj['shape'], perturbed, obj['size'], obj['color'])
            # Right group converges
            for i, obj in enumerate(right_objs):
                target = final_right[i]
                alpha = min(1.0, t / 10)
                pos = (1 - alpha) * obj['pos'] + alpha * target
                draw_shape(ax, obj['shape'], pos, obj['size'], obj['color'] if t < 13 else "red")

        if t == 12:
            trigger_center = left_center if move_left else right_center
            ax.plot(trigger_center[0], trigger_center[1], marker='*', markersize=30, color='green')

        fig.savefig(f"{output_dir}/frame_{t:03d}.png")
        plt.close()

def generate_proximity_group_video_negative(
    cs: bool, cc: bool, cz: bool,
    size: float, count: int, output_dir: str, frames: int = 20
):
    """
    Generate a negative video for proximity_group_trigger:
    - No convergence, or wrong objects change
    - Trigger is misaligned or missing
    """
    import random
    os.makedirs(output_dir, exist_ok=True)

    shape_options = ['circle', 'square', 'triangle']
    color_options = ['blue', 'green', 'orange']
    size_options = [size * 0.8, size, size * 1.2]

    center = np.array([0.5, 0.5])

    # Assign properties for left group
    left_shape = random.choice(shape_options) if cs else None
    left_color = random.choice(color_options) if cc else None
    left_size = random.choice(size_options) if cz else None
    left_objs = []
    for _ in range(count):
        obj_shape = left_shape if cs else random.choice(shape_options)
        obj_color = left_color if cc else random.choice(color_options)
        obj_size = left_size if cz else random.choice(size_options)
        pos = np.array([np.random.uniform(0.1, 0.4), np.random.uniform(0.3, 0.7)])
        left_objs.append({'pos': pos, 'shape': obj_shape, 'color': obj_color, 'size': obj_size})

    # Assign properties for right group
    right_shape = random.choice(shape_options) if cs else None
    right_color = random.choice(color_options) if cc else None
    right_size = random.choice(size_options) if cz else None
    right_objs = []
    for _ in range(count):
        obj_shape = right_shape if cs else random.choice(shape_options)
        obj_color = right_color if cc else random.choice(color_options)
        obj_size = right_size if cz else random.choice(size_options)
        pos = np.array([np.random.uniform(0.6, 0.9), np.random.uniform(0.3, 0.7)])
        right_objs.append({'pos': pos, 'shape': obj_shape, 'color': obj_color, 'size': obj_size})

    # Decide which half to change color after trigger
    change_left = np.random.rand() < 0.5

    for t in range(frames):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Left objects jitter randomly — no convergence
        for obj in left_objs:
            jitter = 0.01 * np.random.randn(2)
            perturbed = np.clip(obj['pos'] + jitter, 0, 1)
            if change_left and t >= 13:
                draw_shape(ax, obj['shape'], perturbed, obj['size'], "red")
            else:
                draw_shape(ax, obj['shape'], perturbed, obj['size'], obj['color'])

        # Right objects also jitter — distractor movement
        for obj in right_objs:
            jitter = 0.01 * np.random.randn(2)
            perturbed = np.clip(obj['pos'] + jitter, 0, 1)
            if not change_left and t >= 13:
                draw_shape(ax, obj['shape'], perturbed, obj['size'], "red")
            else:
                draw_shape(ax, obj['shape'], perturbed, obj['size'], obj['color'])

        # Misaligned or early trigger
        if t == 5:
            ax.plot(center[0], center[1], marker='*', markersize=30, color='green')

        fig.savefig(f"{output_dir}/frame_{t:03d}.png")
        plt.close()


def generate_proximity_task_batch(
        cs: bool,
        cc: bool,
        cz: bool,
        obj_size: float,
        obj_count: int,
        base_dir: str,
        num_per_class: int = 10
):
    """
    Generate a batch of proximity_group_trigger videos for a given task configuration.
    Produces `num_per_class` positive and `num_per_class` negative samples.
    """
    base_path = Path(base_dir)
    pos_dir = base_path / "positive"
    neg_dir = base_path / "negative"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_per_class):
        out_dir = pos_dir / f"{i:05d}"
        generate_proximity_group_video(
            cs=cs,
            cc=cc,
            cz=cz,
            size=obj_size,
            count=obj_count,
            output_dir=str(out_dir)
        )
        frame_paths = [out_dir / f"frame_{t:03d}.png" for t in range(20)]
        images = [imageio.imread(str(p)) for p in frame_paths]
        gif_path = pos_dir / f"{i:05d}.gif"
        imageio.mimsave(str(gif_path), images, duration=0.1)

    for i in range(num_per_class):
        out_dir = neg_dir / f"{i:05d}"
        generate_proximity_group_video_negative(
            cs=cs,
            cc=cc,
            cz=cz,
            size=obj_size,
            count=obj_count,
            output_dir=str(neg_dir / f"{i:05d}")
        )
        frame_paths = [out_dir / f"frame_{t:03d}.png" for t in range(20)]
        images = [imageio.imread(str(p)) for p in frame_paths]
        gif_path = neg_dir / f"{i:05d}.gif"
        imageio.mimsave(str(gif_path), images, duration=0.1)
    combine_gifs(pos_dir, neg_dir, base_path / "combined.gif")


def non_overlap_scatter_cluster(cs, cc, cz, size_label):
    size_map = {'s': 0.05, 'm': 0.08, 'l': 0.12}
    count_map = {'s': 5, 'm': 10, 'l': 15}
    obj_size = size_map[size_label]
    obj_count = count_map[size_label]

    def task_fn(output_dir: str, num_pos: int, num_neg: int):
        generate_proximity_task_batch(
            cs=cs,
            cc=cc,
            cz=cz,
            obj_size=obj_size,
            obj_count=obj_count,
            base_dir=output_dir,
            num_per_class=min(num_pos, num_neg)
        )

    return task_fn
