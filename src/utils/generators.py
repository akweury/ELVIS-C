# Created by MacBook Pro at 15.07.25

import imageio
from PIL import Image
from pathlib import Path

from matplotlib.patches import Circle, Rectangle, RegularPolygon


def draw_shape(ax, shape, center, size, color):
    x, y = center
    if shape == 'circle':
        ax.add_patch(Circle((x, y), size / 2, color=color))
    elif shape == 'square':
        ax.add_patch(Rectangle((x - size / 2, y - size / 2), size, size, color=color))
    elif shape == 'triangle':
        ax.add_patch(RegularPolygon((x, y), 3, size / 2, orientation=3.14 / 2, color=color))
    else:
        raise ValueError(f"Unknown shape: {shape}")


# def create_tasks_v3(task_func, variation_keys, variant_range, size_list, pin=False):
#     shapes = ['circle', 'square', 'triangle']
#     colors = ['blue', 'green', 'orange']
#     tasks = {}
#     for shape in shapes:
#         for color in colors:
#             for size in size_list:
#                 for variant in variant_range:
#                     name = f"{task_func.__name__}_{shape}_{color}_{size}_{variant}"
#                     tasks[name] = lambda s=shape, c=color, sz=size, v=variant: task_func(s, c, sz, v)
#     return tasks

def create_tasks_v3(task_func, variation_keys, variant_range, size_list, pin=False):
    shapes = ['circle', 'square', 'triangle']
    colors = ['blue', 'green', 'orange']
    tasks = {}

    for shape in shapes:
        for color in colors:
            for size in size_list:
                for variant in variant_range:
                    name = f"{task_func.__name__}_{shape}_{color}_{size}_{variant}"
                    def make_task(s=shape, c=color, sz=size, v=variant):
                        return task_func(s, c, sz, v)
                    tasks[name] = make_task()

    return tasks


def combine_gifs(pos_dir, neg_dir, output_path):
    pos_gifs = sorted(Path(pos_dir).glob("*.gif"))
    neg_gifs = sorted(Path(neg_dir).glob("*.gif"))

    # Load frames for each gif
    pos_frames = [list(imageio.get_reader(str(gif))) for gif in pos_gifs]
    neg_frames = [list(imageio.get_reader(str(gif))) for gif in neg_gifs]

    num_frames = min(len(frames) for frames in pos_frames + neg_frames)
    combined_frames = []

    for t in range(num_frames):
        # Get t-th frame from each gif, convert to PIL Image
        pos_row = [Image.fromarray(frames[t]) for frames in pos_frames]
        neg_row = [Image.fromarray(frames[t]) for frames in neg_frames]

        # Concatenate horizontally
        pos_concat = Image.new('RGB', (sum(img.width for img in pos_row), pos_row[0].height))
        x_offset = 0
        for img in pos_row:
            pos_concat.paste(img, (x_offset, 0))
            x_offset += img.width

        neg_concat = Image.new('RGB', (sum(img.width for img in neg_row), neg_row[0].height))
        x_offset = 0
        for img in neg_row:
            neg_concat.paste(img, (x_offset, 0))
            x_offset += img.width

        # Stack rows vertically
        final_img = Image.new('RGB', (pos_concat.width, pos_concat.height + neg_concat.height))
        final_img.paste(pos_concat, (0, 0))
        final_img.paste(neg_concat, (0, pos_concat.height))

        combined_frames.append(final_img)

    # Save as GIF
    combined_frames[0].save(
        output_path,
        save_all=True,
        append_images=combined_frames[1:],
        duration=100,
        loop=0
    )