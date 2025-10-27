from __future__ import annotations
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import os
from elvis_env.envs.elvis_video_env import ElvisVideoEnv, Action

@dataclass
class Rollout:
    frames: List  # list[np.ndarray]
    symbols: List # list[dict]

@dataclass
class Pair:
    baseline: Rollout
    intervention: Rollout
    meta: Dict[str, Any]

def _rollout(env: ElvisVideoEnv, num_frames: int, intervene_at: Optional[int]=None, do: Optional[Dict[str,Any]]=None):
    frames, symbols = [], []
    obs, _ = env.reset()
    frames.append(obs["frame"])
    symbols.append(obs["symbols"])
    for t in range(1, num_frames):
        # 在指定帧施加干预
        if intervene_at is not None and do is not None and t == intervene_at:
            env.step(do)  # do 是 dict，会在 env 内部解析为 Action
        else:
            env.step({"op": "move", "obj": None, "dx": 0.0, "dy": 0.0})  # 空动作推进时间
        o = env._observe()
        frames.append(o["frame"])
        symbols.append(o["symbols"])
    return Rollout(frames=frames, symbols=symbols)

def make_intervention_pair(cfg_path: str, seed: int, do: Dict[str, Any], t_intervene: int, num_frames: int=16) -> Pair:
    """
    通过“同一seed、同一初始场景”构造 baseline vs intervention。
    - baseline: 无干预，仅时间推进
    - intervention: 在 t_intervene 执行 do(·)
    """     
    # baseline env
    env0 = ElvisVideoEnv(cfg_path=cfg_path)
    env0.reset(seed=seed)
    baseline = _rollout(env0, num_frames=num_frames)

    # intervention env（与 baseline 同 seed 重置，保证起点一致）
    env1 = ElvisVideoEnv(cfg_path=cfg_path)
    env1.reset(seed=seed)
    intervention = _rollout(env1, num_frames=num_frames, intervene_at=t_intervene, do=do)

    W, H = env0.cfg.width, env0.cfg.height

    # 基本元数据
    meta = {
        "seed": seed,
        "cfg_path": cfg_path,
        "num_frames": num_frames,
        "intervention": do,
        "t_intervene": t_intervene,
        "scene_id_baseline": baseline.symbols[0]["scene_id"],
        "scene_id_intervention": intervention.symbols[0]["scene_id"],
        "canvas": {"width": W, "height": H},      # <-- added
        "note": "Baseline and intervention are matched by identical seeds and initial sampling.",
    }
    return Pair(baseline=baseline, intervention=intervention, meta=meta)

