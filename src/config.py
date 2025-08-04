# Created by MacBook Pro at 14.07.25

from pathlib import Path
import matplotlib

# config.py
img_size = 224
size_map = {'s': 0.12, 'm': 0.08, 'l': 0.05, 'xl': 0.04, 'xxl': 0.03, 'xxxl': 0.02}
standard_quantity_dict = {"s": 5,
                          "m": 15,
                          "l": 20,
                          "xl": 25,
                          "xxl": 30,
                          "xxxl": 35}

frame_length = 20


def get_example_num(lite=False):
    """Return the number of examples based on the mode."""
    return 3 if lite else 100


# Size variants for shape
size_list = ["s", "m", "l"]
shapes = ['circle', 'square', 'triangle']
colors = ['blue', 'green', 'orange']

# Whether to include principle in negative samples
prin_in_neg = False

root = Path(__file__).parents[0]
raw_patterns = root / 'video_tasks'

output_dir = root / "output"
if not output_dir.exists():
    output_dir.mkdir(parents=True)

all_shapes = [
    "triangle",
    "square",
    "circle",
    "pentagon",
    "hexagon",
    "star",
    "cross",
    "plus",
    "diamond",
    "heart",
    "spade",
    "club",
]

color_matplotlib = {k: tuple(int(v[i:i + 2], 16) for i in (1, 3, 5)) for k, v in
                    list(matplotlib.colors.cnames.items())}
color_matplotlib.pop("darkslategray")
color_matplotlib.pop("lightslategray")
color_matplotlib.pop("black")
color_matplotlib.pop("darkgray")
color_large = [k for k, v in list(color_matplotlib.items())]
color_large_exclude_gray = [item for item in color_large if item != "lightgray" and item != "lightgrey"]
