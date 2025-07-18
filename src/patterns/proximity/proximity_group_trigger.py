# Created by MacBook Pro at 15.07.25

import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import imageio

from utils.generators import draw_shape, combine_gifs


def generate_proximity_group_video(shape: str, color: str, size: float, count: int, output_dir: str, frames: int = 20):
    """Generate a proximity-based grouping demo where either left or right objects converge to their half center and trigger an effect."""
    os.makedirs(output_dir, exist_ok=True)

    # Decide which half will move
    move_left = np.random.rand() < 0.5
    left_center = np.array([0.25, 0.5])
    right_center = np.array([0.75, 0.5])

    # Initial random positions
    left_objs = [np.array([np.random.uniform(0.1, 0.4), np.random.uniform(0.3, 0.7)]) for _ in range(count)]
    right_objs = [np.array([np.random.uniform(0.6, 0.9), np.random.uniform(0.3, 0.7)]) for _ in range(count)]

    # Final positions around the chosen center
    angle_step = 2 * np.pi / count
    if move_left:
        final_left = [left_center + 0.1 * np.array([np.cos(i * angle_step), np.sin(i * angle_step)]) for i in range(count)]
    else:
        final_right = [right_center + 0.1 * np.array([np.cos(i * angle_step), np.sin(i * angle_step)]) for i in range(count)]

    for t in range(frames):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Animate convergence for the chosen half
        if move_left:
            for i, start in enumerate(left_objs):
                target = final_left[i]
                alpha = min(1.0, t / 10)
                pos = (1 - alpha) * start + alpha * target
                draw_shape(ax, shape, pos, size, color if t < 13 else "red")
            # Static right objects
            for pos in right_objs:
                draw_shape(ax, shape, pos, size, color)
        else:
            for pos in left_objs:
                draw_shape(ax, shape, pos, size, color)
            for i, start in enumerate(right_objs):
                target = final_right[i]
                alpha = min(1.0, t / 10)
                pos = (1 - alpha) * start + alpha * target
                draw_shape(ax, shape, pos, size, color if t < 13 else "red")

        # Trigger flash at the chosen center
        if t == 12:
            trigger_center = left_center if move_left else right_center
            ax.plot(trigger_center[0], trigger_center[1], marker='*', markersize=30, color='green')

        fig.savefig(f"{output_dir}/frame_{t:03d}.png")
        plt.close()


def generate_proximity_group_video_negative(
        shape: str,
        color: str,
        size: float,
        count: int,
        output_dir: str,
        frames: int = 20
):
    """
    Generate a negative video for proximity_group_trigger:
    - No convergence, or wrong objects change
    - Trigger is misaligned or missing
    """
    os.makedirs(output_dir, exist_ok=True)

    center = np.array([0.5, 0.5])

    # Random left and right object positions
    left_objs = [np.array([np.random.uniform(0.1, 0.4), np.random.uniform(0.3, 0.7)]) for _ in range(count)]
    right_objs = [np.array([np.random.uniform(0.6, 0.9), np.random.uniform(0.3, 0.7)]) for _ in range(count)]

    # Decide which half to change color after trigger
    change_left = np.random.rand() < 0.5

    for t in range(frames):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Left objects jitter randomly — no convergence
        for pos in left_objs:
            jitter = 0.01 * np.random.randn(2)
            perturbed = np.clip(pos + jitter, 0, 1)
            if change_left and t >= 13:
                draw_shape(ax, shape, perturbed, size, "red")
            else:
                draw_shape(ax, shape, perturbed, size, color)

        # Right objects also jitter — distractor movement
        for pos in right_objs:
            jitter = 0.01 * np.random.randn(2)
            perturbed = np.clip(pos + jitter, 0, 1)
            # Fake causal effect on wrong side
            if not change_left and t >= 13:
                draw_shape(ax, shape, perturbed, size, "red")
            else:
                draw_shape(ax, shape, perturbed, size, color)

        # Misaligned or early trigger
        if t == 5:
            ax.plot(center[0], center[1], marker='*', markersize=30, color='green')

        fig.savefig(f"{output_dir}/frame_{t:03d}.png")
        plt.close()


def generate_proximity_task_batch(
        shape: str,
        color: str,
        size: float,
        count: int,
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
            shape=shape,
            color=color,
            size=size,
            count=count,
            output_dir=str(out_dir)
        )
        frame_paths = [out_dir / f"frame_{t:03d}.png" for t in range(20)]
        images = [imageio.imread(str(p)) for p in frame_paths]
        gif_path = pos_dir / f"{i:05d}.gif"
        imageio.mimsave(str(gif_path), images, duration=0.1)

    for i in range(num_per_class):
        out_dir = neg_dir / f"{i:05d}"
        generate_proximity_group_video_negative(
            shape=shape,
            color=color,
            size=size,
            count=count,
            output_dir=str(neg_dir / f"{i:05d}")
        )
        frame_paths = [out_dir / f"frame_{t:03d}.png" for t in range(20)]
        images = [imageio.imread(str(p)) for p in frame_paths]
        gif_path = neg_dir / f"{i:05d}.gif"
        imageio.mimsave(str(gif_path), images, duration=0.1)
    combine_gifs(pos_dir, neg_dir, base_path/"combined.gif")

def non_overlap_scatter_cluster(shape, color, size_label, variant):
    size_map = {'s': 0.05, 'm': 0.08, 'l': 0.12}
    size = size_map[size_label]
    count = 3 + variant

    def task_fn(output_dir: str, num_pos: int, num_neg: int):
        generate_proximity_task_batch(
            shape=shape,
            color=color,
            size=size,
            count=count,
            base_dir=output_dir,
            num_per_class=min(num_pos, num_neg)
        )

    return task_fn
