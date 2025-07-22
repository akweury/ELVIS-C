# Created by MacBook Pro at 21.07.25


import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import imageio

from utils.generators import draw_shape, combine_gifs
from utils.proximity_utils import assign_group_objects, jitter_position

def generate_proximity_avoidance_video(
    cs: bool, cc: bool, cz: bool,
    size: float, count: int, output_dir: str, frames: int = 20
):
    os.makedirs(output_dir, exist_ok=True)
    shape_options = ['circle', 'square', 'triangle']
    color_options = ['blue', 'green', 'orange']
    size_options = [size * 0.8, size, size * 1.2]

    objs = assign_group_objects(
        count, shape_options, color_options, size_options,
        cs, cc, cz, (0.2, 0.8), (0.2, 0.8)
    )
    obstacle_center = np.array([0.5, 0.5])
    obstacle_radius = 0.15

    for t in range(frames):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        circle = plt.Circle(obstacle_center, obstacle_radius, color='red', alpha=0.3)
        ax.add_patch(circle)

        for obj in objs:
            pos = obj['pos']
            vec = pos - obstacle_center
            dist = np.linalg.norm(vec)
            if dist < obstacle_radius + 0.05:
                direction = vec / (dist + 1e-6)
                pos = pos + direction * 0.03 * (1 - t / frames)
                pos = np.clip(pos, 0.05, 0.95)
            else:
                pos = jitter_position(pos, scale=0.01)
            obj['pos'] = pos
            draw_shape(ax, obj['shape'], pos, obj['size'], obj['color'])

        if t == 12:
            ax.plot(obstacle_center[0], obstacle_center[1], marker='*', markersize=30, color='green')

        fig.savefig(f"{output_dir}/frame_{t:03d}.png")
        plt.close()

def generate_proximity_avoidance_video_negative(
    cs: bool, cc: bool, cz: bool,
    size: float, count: int, output_dir: str, frames: int = 20
):
    os.makedirs(output_dir, exist_ok=True)
    shape_options = ['circle', 'square', 'triangle']
    color_options = ['blue', 'green', 'orange']
    size_options = [size * 0.8, size, size * 1.2]

    objs = assign_group_objects(
        count, shape_options, color_options, size_options,
        cs, cc, cz, (0.2, 0.8), (0.2, 0.8)
    )
    obstacle_center = np.array([0.5, 0.5])
    obstacle_radius = 0.15

    for t in range(frames):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        circle = plt.Circle(obstacle_center, obstacle_radius, color='red', alpha=0.3)
        ax.add_patch(circle)

        for obj in objs:
            pos = obj['pos']
            # Negative: objects move randomly, not away from obstacle
            pos = jitter_position(pos, scale=0.03)
            obj['pos'] = np.clip(pos, 0.05, 0.95)
            draw_shape(ax, obj['shape'], obj['pos'], obj['size'], obj['color'])

        if t == 5:
            ax.plot(obstacle_center[0], obstacle_center[1], marker='*', markersize=30, color='green')

        fig.savefig(f"{output_dir}/frame_{t:03d}.png")
        plt.close()

def generate_proximity_avoidance_task_batch(
    cs: bool, cc: bool, cz: bool,
    obj_size: float, obj_count: int,
    base_dir: str, num_per_class: int = 10
):
    base_path = Path(base_dir)
    pos_dir = base_path / "positive"
    neg_dir = base_path / "negative"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_per_class):
        out_dir = pos_dir / f"{i:05d}"
        generate_proximity_avoidance_video(
            cs=cs, cc=cc, cz=cz,
            size=obj_size, count=obj_count,
            output_dir=str(out_dir)
        )
        frame_paths = [out_dir / f"frame_{t:03d}.png" for t in range(20)]
        images = [imageio.imread(str(p)) for p in frame_paths]
        gif_path = pos_dir / f"{i:05d}.gif"
        imageio.mimsave(str(gif_path), images, duration=0.1)

    for i in range(num_per_class):
        out_dir = neg_dir / f"{i:05d}"
        generate_proximity_avoidance_video_negative(
            cs=cs, cc=cc, cz=cz,
            size=obj_size, count=obj_count,
            output_dir=str(out_dir)
        )
        frame_paths = [out_dir / f"frame_{t:03d}.png" for t in range(20)]
        images = [imageio.imread(str(p)) for p in frame_paths]
        gif_path = neg_dir / f"{i:05d}.gif"
        imageio.mimsave(str(gif_path), images, duration=0.1)
    combine_gifs(pos_dir, neg_dir, base_path / "combined.gif")

def proximity_avoidance_task_fn(cs, cc, cz, size_label):
    size_map = {'s': 0.05, 'm': 0.08, 'l': 0.12}
    count_map = {'s': 5, 'm': 10, 'l': 15}
    obj_size = size_map[size_label]
    obj_count = count_map[size_label]

    def task_fn(output_dir: str, num_pos: int, num_neg: int):
        generate_proximity_avoidance_task_batch(
            cs=cs, cc=cc, cz=cz,
            obj_size=obj_size, obj_count=obj_count,
            base_dir=output_dir,
            num_per_class=min(num_pos, num_neg)
        )
    return task_fn
