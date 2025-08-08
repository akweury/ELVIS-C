# Created by MacBook Pro at 23.07.25
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import imageio
import random
from src.utils.generators import draw_shape, combine_gifs, draw_shape_cv2, get_proper_sublist
from src.utils.proximity_utils import assign_group_objects, jitter_position
import cv2
import math
import json

from src import config


def line_circle_intersect(p1, p2, center, radius):
    # Check if the segment p1-p2 intersects the circle
    d = p2 - p1
    f = p1 - center
    a = np.dot(d, d)
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - radius ** 2
    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        return False  # No intersection
    discriminant = math.sqrt(discriminant)
    t1 = (-b - discriminant) / (2 * a)
    t2 = (-b + discriminant) / (2 * a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1)


def generate_proximity_leader_trigger_video(params, irrel_param, grp_num, obj_num, obj_size, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    img_size = config.img_size

    settings = {
        "red_start": np.array([0.1, 0.1]),
        "obstacle_radius": 0.15,
        "obstacle_center": np.array([0.5, 0.5]),
        "x_range": (0.2, 0.8),
        "y_range": (0.2, 0.8),
    }
    logic = {"shape": ["square", "circle"],
             "color": ["green", "yellow"],
             "size": [obj_size],
             "count": True}

    # Decide if red moves to center or to a random point outside the circle
    move_to_center = random.choice([True, False])
    if move_to_center:
        red_end = settings["obstacle_center"]
    else:
        # Pick a random point such that the line does NOT intersect the circle
        while True:
            candidate = np.random.uniform(0.1, 0.9, size=2)
            if np.linalg.norm(candidate - settings["obstacle_center"]) > settings["obstacle_radius"] + 0.05:
                if not line_circle_intersect(settings["red_start"], candidate, settings["obstacle_center"], settings["obstacle_radius"]):
                    red_end = candidate
                    break

    cs = True if "shape" in params else False
    cc = True if "color" in params else False
    cz = True if "size" in params else False

    if "shape" in irrel_param:
        logic["shape"] = [random.choice(config.all_shapes)]
    if "color" in irrel_param:
        logic["color"] = [random.choice(config.color_large_exclude_gray)]
    if "size" in irrel_param:
        logic["size"] = [random.choice(list(config.size_map.values()))]

    objs = []
    objs += assign_group_objects(1, logic["shape"], ["red"], logic["size"], cs, True, cz,
                                 settings["x_range"], settings["y_range"])
    objs += assign_group_objects(obj_num, logic["shape"], logic["color"], logic["size"], cs, cc, cz,
                                 settings["x_range"], settings["y_range"])
    red_entered = False

    for t in range(config.frame_length):
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

        # Draw transparent obstacle
        center_px = (int(settings["obstacle_center"][0] * img_size), int(settings["obstacle_center"][1] * img_size))
        radius_px = int(settings["obstacle_radius"] * img_size)
        overlay = img.copy()
        cv2.circle(overlay, center_px, radius_px, (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

        # Move red object along straight line, always moving
        alpha = min(1.0, t / (config.frame_length - 1))
        objs[0]['pos'] = (1 - alpha) * settings["red_start"] + alpha * red_end
        objs[0]['pos'] = jitter_position(objs[0]['pos'], scale=0.003)
        # Check if red entered the circle
        if move_to_center and not red_entered:
            if np.linalg.norm(objs[0]['pos'] - settings["obstacle_center"]) < settings["obstacle_radius"]:
                red_entered = True
        draw_shape_cv2(img, objs[0]['shape'], objs[0]['pos'], objs[0]['size'], objs[0]['color'])

        # Move other objects
        for obj in objs[1:]:
            if move_to_center and red_entered:
                vec = obj['pos'] - settings["obstacle_center"]
                dist = np.linalg.norm(vec)
                direction = vec / (dist + 1e-6)
                obj['pos'] += direction * 0.03
            else:
                obj['pos'] = jitter_position(obj['pos'], scale=0.005)
            obj['pos'] = np.clip(obj['pos'], 0.05, 0.95)
            draw_shape_cv2(img, obj['shape'], obj['pos'], obj['size'], obj['color'])

        cv2.imwrite(f"{out_dir}/frame_{t:03d}.png", img)
        # Save ground-truth JSON for this frame
        frame_info = [{
            "position": [0.5, 0.5],
            "shape": "circle",
            "color": "red",
            "size": radius_px / img_size * 2
        }]
        for obj in objs:
            frame_info.append({
                "position": obj['pos'].tolist(),
                "shape": obj['shape'],
                "color": obj['color'],
                "size": obj['size'],
            })
        with open(f"{out_dir}/frame_{t:03d}.json", "w") as f:
            json.dump(frame_info, f, indent=2)


def generate_proximity_leader_trigger_task_batch(id, base_path, param, irrel_param, grp_num, obj_num_name):
    pos_dir = base_path / "positive"
    neg_dir = base_path / "negative"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    out_dir = pos_dir / f"{id:05d}"
    obj_size = config.size_map[obj_num_name]
    obj_num = config.standard_quantity_dict[obj_num_name]
    generate_proximity_leader_trigger_video(param, irrel_param, grp_num, obj_num, obj_size, str(out_dir))
    frame_paths = [out_dir / f"frame_{t:03d}.png" for t in range(20)]
    images = [imageio.imread(str(p)) for p in frame_paths]
    gif_path = pos_dir / f"{id:05d}.gif"
    imageio.mimsave(str(gif_path), images, duration=0.1)


def proximity_leader_trigger_task_fn(id, save_dir, task_name, param, irrel_param, grp_num, obj_num):
    base_path = Path(save_dir)
    generate_proximity_leader_trigger_task_batch(id, base_path, param, irrel_param, grp_num, obj_num)

    # generate combined GIFs for positive and negative samples
