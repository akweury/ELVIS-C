# Created by MacBook Pro at 23.07.25
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import imageio

from src.utils.generators import draw_shape, combine_gifs
from src.utils.proximity_utils import assign_group_objects, jitter_position

def generate_continuity_chain_video(cs, cc, cz, size, count, output_dir, frames=20):
    os.makedirs(output_dir, exist_ok=True)
    shape_options = ['circle', 'square', 'triangle']
    color_options = ['blue', 'green', 'orange']
    size_options = [size * 0.8, size, size * 1.2]

    objs = assign_group_objects(
        count, shape_options, color_options, size_options,
        cs, cc, cz, (0.2, 0.8), (0.2, 0.8)
    )
    # Choose a random start point for the chain
    chain_start = np.array([0.2, 0.5])
    chain_dir = np.array([0.7, 0.0]) / (count - 1)

    for t in range(frames):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        for i, obj in enumerate(objs):
            # Target position in the chain
            target = chain_start + chain_dir * i
            # Move towards target
            obj['pos'] += (target - obj['pos']) * 0.1
            obj['pos'] = jitter_position(obj['pos'], scale=0.005)
            obj['pos'] = np.clip(obj['pos'], 0.05, 0.95)
            draw_shape(ax, obj['shape'], obj['pos'], obj['size'], obj['color'])

        fig.savefig(f"{output_dir}/frame_{t:03d}.png")
        plt.close()

def generate_continuity_chain_video_negative(cs, cc, cz, size, count, output_dir, frames=20):
    os.makedirs(output_dir, exist_ok=True)
    shape_options = ['circle', 'square', 'triangle']
    color_options = ['blue', 'green', 'orange']
    size_options = [size * 0.8, size, size * 1.2]

    objs = assign_group_objects(
        count, shape_options, color_options, size_options,
        cs, cc, cz, (0.2, 0.8), (0.2, 0.8)
    )

    for t in range(frames):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        for obj in objs:
            # Random movement, not forming a chain
            obj['pos'] = jitter_position(obj['pos'], scale=0.03)
            obj['pos'] = np.clip(obj['pos'], 0.05, 0.95)
            draw_shape(ax, obj['shape'], obj['pos'], obj['size'], obj['color'])

        fig.savefig(f"{output_dir}/frame_{t:03d}.png")
        plt.close()

def generate_continuity_chain_task_batch(cs, cc, cz, obj_size, obj_count, base_dir, num_per_class=10):
    base_path = Path(base_dir)
    pos_dir = base_path / "positive"
    neg_dir = base_path / "negative"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_per_class):
        out_dir = pos_dir / f"{i:05d}"
        generate_continuity_chain_video(cs, cc, cz, obj_size, obj_count, str(out_dir))
        frame_paths = [out_dir / f"frame_{t:03d}.png" for t in range(20)]
        images = [imageio.imread(str(p)) for p in frame_paths]
        gif_path = pos_dir / f"{i:05d}.gif"
        mp4_path = pos_dir / f"{i:05d}.mp4"
        imageio.mimsave(str(gif_path), images, duration=0.1)
        imageio.mimsave(str(mp4_path), images, fps=10, macro_block_size=None, plugin='ffmpeg')

    for i in range(num_per_class):
        out_dir = neg_dir / f"{i:05d}"
        generate_continuity_chain_video_negative(cs, cc, cz, obj_size, obj_count, str(out_dir))
        frame_paths = [out_dir / f"frame_{t:03d}.png" for t in range(20)]
        images = [imageio.imread(str(p)) for p in frame_paths]
        gif_path = neg_dir / f"{i:05d}.gif"
        mp4_path = neg_dir / f"{i:05d}.mp4"
        imageio.mimsave(str(gif_path), images, duration=0.1)
        imageio.mimsave(str(mp4_path), images, fps=10, macro_block_size=None, plugin='ffmpeg')

    combine_gifs(pos_dir, neg_dir, base_path / "combined.gif")

def continuity_chain_task_fn(cs, cc, cz, size_label):
    size_map = {'s': 0.05, 'm': 0.08, 'l': 0.12}
    count_map = {'s': 5, 'm': 10, 'l': 15}
    obj_size = size_map[size_label]
    obj_count = count_map[size_label]

    def task_fn(output_dir: str, num_pos: int, num_neg: int):
        generate_continuity_chain_task_batch(
            cs=cs, cc=cc, cz=cz,
            obj_size=obj_size, obj_count=obj_count,
            base_dir=output_dir,
            num_per_class=min(num_pos, num_neg)
        )
    return task_fn




