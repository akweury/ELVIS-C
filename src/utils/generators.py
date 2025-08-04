# Created by MacBook Pro at 15.07.25

import imageio
from PIL import Image
from pathlib import Path
import cv2
import numpy as np
from matplotlib.patches import Circle, Rectangle, RegularPolygon
from itertools import combinations
import random


def get_all_combs(given_list):
    # Generate all combinations of all lengths
    all_combinations = []
    for r in range(1, len(given_list) + 1):
        all_combinations.extend(combinations(given_list, r))

    # Convert to a list of lists (optional)
    all_combinations = [list(comb) for comb in all_combinations]
    return all_combinations


def draw_shape(ax, shape, center, size, color):
    x, y = center
    if shape == 'circle':
        ax.add_patch(Circle((x, y), size / 2, color=color))
    elif shape == 'square':
        ax.add_patch(Rectangle((x - size / 2, y - size / 2), size, size, color=color))
    elif shape == 'triangle':
        ax.add_patch(RegularPolygon((x, y), numVertices=3, radius=size / 2, orientation=3.14 / 2, facecolor=color))
    else:
        raise ValueError(f"Unknown shape: {shape}")


def draw_shape_cv2(img, shape, pos, size, color):
    # Map color names to BGR
    color_map = {
        'red': (0, 0, 255),
        'blue': (255, 0, 0),
        'green': (0, 255, 0),
        'orange': (0, 128, 255)
    }
    bgr = color_map.get(color, (0, 0, 0))
    h, w = img.shape[:2]
    cx, cy = int(pos[0] * w), int(pos[1] * h)
    s = int(size * w)
    if shape == 'circle':
        cv2.circle(img, (cx, cy), s // 2, bgr, -1)
    elif shape == 'square':
        cv2.rectangle(img, (cx - s // 2, cy - s // 2), (cx + s // 2, cy + s // 2), bgr, -1)
    elif shape == 'triangle':
        pts = np.array([
            [cx, cy - s // 2],
            [cx - s // 2, cy + s // 2],
            [cx + s // 2, cy + s // 2]
        ], np.int32)
        cv2.fillPoly(img, [pts], bgr)


# def create_tasks_v3(task_func, size_list, pin=False):
#     tasks = {}
#     count = 0
#     consider_options = [True, False]
#     for consider_shape in consider_options:
#         for consider_color in consider_options:
#             for consider_size in consider_options:
#                 for size_label in size_list:
#                     name = f"{count}_{task_func.__name__}_s{int(consider_shape)}c{int(consider_color)}{size_label}"
#
#                     def make_task(cs=consider_shape, cc=consider_color, cz=consider_size):
#                         return task_func(name, cs, cc, cz, size_label)
#
#                     tasks[name] = make_task()
#                     count += 1
#     return tasks


def create_tasks_v3(func, params, grp_nums, obj_quantity_list):
    tasks = []
    names = []
    counter = 0
    for rel_comb in get_all_combs(params):
        irrelevant_params = [k for k in params if k not in rel_comb and k != "position"]
        for irrel_comb in get_all_combs(irrelevant_params):
            for grp_num in grp_nums:
                for oq in obj_quantity_list:
                    task_name = (
                            f"{counter}_{func.__name__}_rel_"
                            + "_".join(f"{k}" for k in rel_comb)
                            + f"_{grp_num}_{oq}_irrel_"
                            + "_".join(f"{k}" for k in irrel_comb))
                    counter += 1
                    if task_name in tasks:
                        raise ValueError(f"Duplicate task key detected: {task_name}")
                    tasks.append(lambda id, save_path, tn=task_name, group_num=grp_num, relative_comb=rel_comb, irrelative_comb=irrel_comb,
                                        obj_quantity=oq: func(id, save_path, tn, relative_comb, irrelative_comb, group_num, obj_quantity))
                    names.append(task_name)
    return tasks, names


def combine_gifs(pos_dir, output_path):
    pos_gifs = sorted(Path(pos_dir).glob("*.gif"))

    # Load frames for each gif
    pos_frames = [list(imageio.get_reader(str(gif))) for gif in pos_gifs]

    num_frames = min(len(frames) for frames in pos_frames)
    combined_frames = []

    for t in range(num_frames):
        # Get t-th frame from each gif, convert to PIL Image
        pos_row = [Image.fromarray(frames[t]) for frames in pos_frames]

        # Concatenate horizontally
        pos_concat = Image.new('RGB', (sum(img.width for img in pos_row), pos_row[0].height))
        x_offset = 0
        for img in pos_row:
            pos_concat.paste(img, (x_offset, 0))
            x_offset += img.width

        # Stack rows vertically
        final_img = Image.new('RGB', (pos_concat.width, pos_concat.height))
        final_img.paste(pos_concat, (0, 0))
        combined_frames.append(final_img)

    # Save as GIF
    combined_frames[0].save(
        output_path,
        save_all=True,
        append_images=combined_frames[1:],
        duration=100,
        loop=0
    )

def get_proper_sublist(lst):
    if not lst:
        return []  # Return an empty list if the input list is empty
    if len(lst) == 1:
        return []
    sublist_size = random.randint(1, len(lst) - 1)  # Ensure it's a proper sublist
    return random.sample(lst, sublist_size)  # Randomly select elements

