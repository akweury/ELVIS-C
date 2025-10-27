from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional, List, Literal
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PIL import Image, ImageDraw
import json
import os
import random

# ---- Action schema -------------------------------------------------------

ActionOp = Literal["move", "freeze", "change_color", "add", "remove", "set_pos"]

@dataclass
class Action:
    op: ActionOp
    obj: Optional[str] = None
    dx: float = 0.0
    dy: float = 0.0
    color: Optional[Tuple[int, int, int]] = None
    pos: Optional[Tuple[float, float]] = None  # normalized [0,1]

# ---- Config / Scene ------------------------------------------------------

@dataclass
class SceneCfg:
    width: int = 320
    height: int = 240
    num_frames: int = 16
    shapes: Tuple[str, ...] = ("circle", "square", "triangle")
    colors: List[Tuple[int, int, int]] = field(default_factory=lambda: [(220,20,60),(30,144,255),(60,179,113),(255,140,0)])
    size_px: Tuple[int, int] = (18, 26)
    allow_ops: Tuple[ActionOp, ...] = ("move","freeze","change_color","add","remove","set_pos")

@dataclass
class Obj:
    id: str
    shape: str
    color: Tuple[int, int, int]
    size: int
    # normalized center position in [0,1]
    x: float
    y: float
    # velocity in normalized coords per frame
    vx: float = 0.0
    vy: float = 0.0
    frozen: bool = False

# ---- Environment ---------------------------------------------------------

class ElvisVideoEnv(gym.Env):
    """
    Minimal S-level ELVIS video environment:
      - Deterministic 2D kinematics (no physics)
      - Render RGB frames (H,W,3) uint8
      - Maintain object-centric state with stable IDs
      - Expose simple interventions via Action
    Observation: Dict(frame: uint8[H,W,3], symbols: dict-like metadata)
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg_path: Optional[str] = None, render_mode: str = "rgb_array"):
        super().__init__()
        self.render_mode = render_mode
        self.rng = np.random.RandomState(42)
        self._load_cfg(cfg_path)
        self.frame_idx = 0
        self.scene_id = 0
        self.objects: Dict[str, Obj] = {}
        self._build_spaces()

    # ---- Gym API ----
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
            random.seed(seed)
        self.frame_idx = 0
        self.scene_id += 1
        self.objects = self._sample_scene()
        obs = self._observe()
        info = {"scene_id": self.scene_id}
        return obs, info

    def step(self, action: Dict[str, Any]):
        self._apply_action(self._parse_action(action))
        self._advance_kinematics()
        self.frame_idx += 1
        obs = self._observe()
        terminated = False
        truncated = self.frame_idx >= self.cfg.num_frames
        reward = 0.0  # S-level: no task reward; causal/ILP uses exports
        info = {"frame_idx": self.frame_idx}
        return obs, reward, terminated, truncated, info

    # ---- Spaces ----
    def _build_spaces(self):
        H, W = self.cfg.height, self.cfg.width
        self.observation_space = spaces.Dict({
            "frame": spaces.Box(0, 255, shape=(H, W, 3), dtype=np.uint8),
            # symbols are exported as a JSON-like dict; not strictly boxed here
        })
        # Action space is dict-like; we keep python-side validation for flexibility
        self.action_space = spaces.Dict({
            "op": spaces.Text(min_length=2, max_length=16),
            "obj": spaces.Text(min_length=0, max_length=8),
            "dx": spaces.Box(low=-1.0, high=1.0, shape=()),
            "dy": spaces.Box(low=-1.0, high=1.0, shape=()),
        })

    # ---- Scene & Kinematics ----
    def _sample_scene(self) -> Dict[str, Obj]:
        # simple 2–4 randomized objects
        n = self.rng.randint(2, 5)
        objs: Dict[str, Obj] = {}
        for i in range(n):
            oid = chr(ord('A') + i)
            shape = self.rng.choice(self.cfg.shapes)
            color = tuple(self.cfg.colors[self.rng.randint(0, len(self.cfg.colors))])
            size = int(self.rng.randint(self.cfg.size_px[0], self.cfg.size_px[1]+1))
            x, y = self.rng.uniform(0.2, 0.8), self.rng.uniform(0.2, 0.8)
            vx, vy = self.rng.uniform(-0.02, 0.02), self.rng.uniform(-0.02, 0.02)
            objs[oid] = Obj(oid, shape, color, size, x, y, vx, vy, False)
        return objs

    def _advance_kinematics(self):
        for o in self.objects.values():
            if o.frozen:
                continue
            o.x = np.clip(o.x + o.vx, 0.05, 0.95)
            o.y = np.clip(o.y + o.vy, 0.05, 0.95)

    # ---- Actions ----
    def _parse_action(self, a: Dict[str, Any]) -> Action:
        op = a.get("op")
        if op not in self.cfg.allow_ops:
            raise ValueError(f"Unsupported op: {op}")
        return Action(
            op=op,
            obj=a.get("obj"),
            dx=float(a.get("dx", 0.0)),
            dy=float(a.get("dy", 0.0)),
            color=tuple(a["color"]) if "color" in a else None,
            pos=tuple(a["pos"]) if "pos" in a else None,
        )

    def _apply_action(self, act: Action):
        if act.op == "move":
            if act.obj and act.obj in self.objects:
                o = self.objects[act.obj]
                o.x = np.clip(o.x + act.dx, 0.0, 1.0)
                o.y = np.clip(o.y + act.dy, 0.0, 1.0)
        elif act.op == "freeze":
            if act.obj and act.obj in self.objects:
                self.objects[act.obj].frozen = True
        elif act.op == "change_color":
            if act.obj and act.obj in self.objects and act.color:
                self.objects[act.obj].color = tuple(int(c) for c in act.color)
        elif act.op == "set_pos":
            if act.obj and act.obj in self.objects and act.pos:
                self.objects[act.obj].x = float(np.clip(act.pos[0], 0.0, 1.0))
                self.objects[act.obj].y = float(np.clip(act.pos[1], 0.0, 1.0))
        elif act.op == "add":
            oid = self._next_id()
            color = tuple(self.cfg.colors[self.rng.randint(0, len(self.cfg.colors))])
            size = int(self.rng.randint(self.cfg.size_px[0], self.cfg.size_px[1]+1))
            x, y = (act.pos if act.pos else (self.rng.uniform(0.2, 0.8), self.rng.uniform(0.2, 0.8)))
            shape = self.rng.choice(self.cfg.shapes)
            self.objects[oid] = Obj(oid, shape, color, size, x, y, 0.0, 0.0, False)
        elif act.op == "remove":
            if act.obj and act.obj in self.objects:
                self.objects.pop(act.obj, None)

    def _next_id(self) -> str:
        # IDs are A, B, C, ... then AA, AB ... (minimal)
        base = [o.id for o in self.objects.values()]
        for i in range(26*3):
            if i < 26:
                cand = chr(ord('A') + i)
            else:
                cand = "A" + chr(ord('A') + (i-26)%26)
            if cand not in base:
                return cand
        return f"X{len(base)}"

    # ---- Observation / Rendering / Symbols ----
    def _observe(self) -> Dict[str, Any]:
        frame = self._render_frame()
        symbols = self._extract_symbols()
        return {"frame": frame, "symbols": symbols}

    def _render_frame(self) -> np.ndarray:
        W, H = self.cfg.width, self.cfg.height
        img = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        for o in self.objects.values():
            cx, cy = int(o.x * W), int(o.y * H)
            s = o.size
            if o.shape == "circle":
                draw.ellipse((cx - s, cy - s, cx + s, cy + s), fill=tuple(o.color), outline=(0,0,0), width=2)
            elif o.shape == "square":
                draw.rectangle((cx - s, cy - s, cx + s, cy + s), fill=tuple(o.color), outline=(0,0,0), width=2)
            else:  # triangle (upright)
                pts = [(cx, cy - s), (cx - s, cy + s), (cx + s, cy + s)]
                draw.polygon(pts, fill=tuple(o.color), outline=(0,0,0))
        return np.array(img, dtype=np.uint8)

    def _extract_symbols(self) -> Dict[str, Any]:
        # Minimal symbol dump: objects list; grouping left for symbolics module later
        objs = []
        for o in self.objects.values():
            objs.append({
                "id": o.id,
                "shape": o.shape,
                "color": o.color,
                "size_px": o.size,
                "pos": [round(o.x, 4), round(o.y, 4)],
                "vel": [round(o.vx, 4), round(o.vy, 4)],
                "frozen": o.frozen
            })
        return {
            "frame_idx": self.frame_idx,
            "scene_id": self.scene_id,
            "objects": objs
        }

    # ---- Config loader ----
    def _load_cfg(self, cfg_path: Optional[str]):
        # very small YAML parser via json/dict to avoid extra deps; we rely on provided defaults if missing
        # since we already have pydantic & yaml in deps in pyproject we could parse, but keep minimal
        import yaml
        if cfg_path and os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                y = yaml.safe_load(f)
        else:
            y = {}
        canvas = y.get("canvas", {})
        video = y.get("video", {})
        objects = y.get("objects", {})
        inter = y.get("interventions", {})
        self.cfg = SceneCfg(
            width=int(canvas.get("width", 320)),
            height=int(canvas.get("height", 240)),
            num_frames=int(video.get("num_frames", 16)),
            shapes=tuple(objects.get("shapes", ["circle","square","triangle"])),
            colors=[tuple(c) for c in objects.get("colors", [(220,20,60),(30,144,255),(60,179,113),(255,140,0)])],
            size_px=tuple(objects.get("size_px", [18, 26])),
            allow_ops=tuple(inter.get("allowed_ops", ["move","freeze","change_color","add","remove","set_pos"]))
        )