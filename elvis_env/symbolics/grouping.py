"""
Grouping & proximity scores (per-frame).

Inputs (per frame): a list of objects, each like:
{
  "id": "A",
  "shape": "circle",
  "color": [r,g,b],
  "size_px": 22,
  "pos": [x_norm, y_norm],   # in [0,1]
  "vel": [vx_norm, vy_norm],
  "frozen": false
}

This module computes:
  1) pairwise distances
  2) soft proximity scores (sigmoid of distance)
  3) instantaneous groups (using tau_on only; hysteresis handled in events/)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import math
import itertools
import numpy as np


@dataclass
class ProximityParams:
    # distance→score: s = sigmoid((tau_on - d) / sigma_d)
    alpha: float = 7.0       # scales tau_on by avg object size (normalized)
    beta: float = 1.35       # tau_off = beta * tau_on (used in events layer)
    sigma_d: float = 0.02    # softness for sigmoid
    min_tau_on: float = 0.05 # clamp for small scenes
    max_tau_on: float = 0.30 # clamp upper bound

    # if canvas dims unknown, treat pixel size normalization as 1.0
    default_canvas_min_px: int = 1


def _avg_norm_size(objects: List[Dict[str, Any]], width: int | None, height: int | None, default_min_px: int) -> float:
    """Average object size normalized by min(canvas_dim)."""
    if not objects:
        return 0.08  # a safe fallback
    min_dim = float(min(width or default_min_px, height or default_min_px))
    sizes = [float(o.get("size_px", 20)) / (min_dim if min_dim > 0 else default_min_px) for o in objects]
    return float(np.clip(np.mean(sizes), 1e-4, 0.5))


def _tau_on_off(objects: List[Dict[str, Any]], width: int | None, height: int | None, p: ProximityParams) -> Tuple[float, float]:
    avg_sz = _avg_norm_size(objects, width, height, p.default_canvas_min_px)
    tau_on = p.alpha * avg_sz
    tau_on = float(np.clip(tau_on, p.min_tau_on, p.max_tau_on))
    tau_off = p.beta * tau_on
    return tau_on, tau_off


def _pairwise_dist(objects: List[Dict[str, Any]]) -> List[Tuple[str, str, float]]:
    """Return list of (id1, id2, distance_norm)."""
    out: List[Tuple[str, str, float]] = []
    by_id = {o["id"]: o for o in objects}
    ids = sorted(by_id.keys())
    for a, b in itertools.combinations(ids, 2):
        pa = by_id[a]["pos"]
        pb = by_id[b]["pos"]
        d = math.dist(pa, pb)  # positions are already normalized in [0,1]
        out.append((a, b, float(d)))
    return out


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def compute_proximity_frame(
    objects: List[Dict[str, Any]],
    width: int | None = None,
    height: int | None = None,
    params: ProximityParams | None = None,
) -> Dict[str, Any]:
    """
    Compute per-pair distances, soft scores, and instantaneous groups for a single frame.
    Returns:
      {
        "tau_on": float,
        "tau_off": float,
        "pairs": [
           {"a":"A","b":"B","dist":0.12,"score":0.83}
        ],
        "groups": [
           {"members":["A","B"],"principle":"proximity"}
        ]
      }
    """
    p = params or ProximityParams()
    tau_on, tau_off = _tau_on_off(objects, width, height, p)
    pairs_info: List[Dict[str, Any]] = []
    groups: List[Dict[str, Any]] = []

    for a, b, d in _pairwise_dist(objects):
        score = _sigmoid((tau_on - d) / p.sigma_d)
        pairs_info.append({"a": a, "b": b, "dist": d, "score": score})
        if d <= tau_on:  # instantaneous grouping (no hysteresis here)
            groups.append({"members": [a, b], "principle": "proximity"})

    return {
        "tau_on": tau_on,
        "tau_off": tau_off,
        "pairs": pairs_info,
        "groups": groups,
    }


def compute_sequence_proximity(
    frames_objects: List[List[Dict[str, Any]]],
    width: int | None = None,
    height: int | None = None,
    params: ProximityParams | None = None,
) -> List[Dict[str, Any]]:
    """
    Convenience: compute proximity info for a sequence of frames.
    Returns a list; each element is the dict returned by compute_proximity_frame().
    """
    out: List[Dict[str, Any]] = []
    for objs in frames_objects:
        out.append(compute_proximity_frame(objs, width=width, height=height, params=params))
    return out


def move_toward_flags(
    frames_objects: List[List[Dict[str, Any]]],
    epsilon: float = 1e-3,
) -> List[Dict[Tuple[str, str], bool]]:
    """
    For each t -> t+1, indicate if distance strictly decreases by > epsilon (per pair).
    Returns:
      A list of dicts for t=0..T-2:
        [{("A","B"): True, ("A","C"): False, ...}, ...]
    """
    flags: List[Dict[Tuple[str, str], bool]] = []
    T = len(frames_objects)
    if T <= 1:
        return flags

    def dist_map(objs: List[Dict[str, Any]]) -> Dict[Tuple[str, str], float]:
        m: Dict[Tuple[str, str], float] = {}
        for a, b, d in _pairwise_dist(objs):
            key = tuple(sorted((a, b)))
            m[key] = d
        return m

    prev = dist_map(frames_objects[0])
    for t in range(1, T):
        cur = dist_map(frames_objects[t])
        pairs = {}
        for k in cur.keys():
            dt = cur[k] - prev.get(k, cur[k])
            pairs[k] = (dt < -epsilon)
        flags.append(pairs)
        prev = cur
    return flags