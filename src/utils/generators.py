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
        ax.add_patch(RegularPolygon((x, y), numVertices=3, radius=size / 2, orientation=3.14 / 2, facecolor=color))
    else:
        raise ValueError(f"Unknown shape: {shape}")


def create_tasks_v3(task_func, size_list, pin=False):
    tasks = {}
    consider_options = [True, False]
    for consider_shape in consider_options:
        for consider_color in consider_options:
            for consider_size in consider_options:
                for size_label in size_list:
                    name = f"{task_func.__name__}_s{int(consider_shape)}c{int(consider_color)}{size_label}"
                    def make_task(cs=consider_shape, cc=consider_color, cz=consider_size):
                        return task_func(cs, cc, cz, size_label)

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
