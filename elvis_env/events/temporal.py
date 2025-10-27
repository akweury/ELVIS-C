"""
Temporal event extraction (merge / dissolve / persist) with hysteresis & stability.

Inputs:
  - per-frame proximity outputs from symbolics/grouping.compute_proximity_frame()
  - optional move_toward flags from symbolics/grouping.move_toward_flags()

Events we emit:
  {"type":"merge","members":["A","B"],"frame":t}
  {"type":"dissolve","members":["A","B"],"frame":t}
  {"type":"persist","members":["A","B"],"frame":t}
  {"type":"move_toward","subject":"A","target":"B","frame":t}  # optional

Hysteresis policy:
  - Use tau_on for entering group, tau_off (> tau_on) for leaving group.
  - Require k consecutive frames satisfying the corresponding condition.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import itertools

@dataclass
class EventParams:
    k_stable: int = 3          # consecutive frames required to confirm state change
    include_persist: bool = True
    emit_move_toward: bool = True  # if you also pass move_toward flags
    # name of principle written to event payload (for clarity / future extension)
    principle_name: str = "proximity"


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted((a, b)))


def detect_events_for_sequence(
    proximity_seq: List[Dict[str, Any]],
    move_toward_seq: List[Dict[Tuple[str, str], bool]] | None = None,
    params: EventParams | None = None,
) -> List[Dict[str, Any]]:
    """
    Arguments:
      proximity_seq: list over frames t=0..T-1; each item like:
        {
          "tau_on": float,
          "tau_off": float,
          "pairs": [{"a","b","dist","score"}, ...],
          "groups": [{"members":[a,b], "principle":"proximity"}, ...]
        }
      move_toward_seq: optional list over transitions t=0..T-2 with flags {pair: bool}

    Returns:
      events: a flat list of event dicts with "frame" keys.
    """
    p = params or EventParams()
    T = len(proximity_seq)
    if T == 0:
        return []

    # Prepare per-frame grouped sets
    grouped_sets: List[set[Tuple[str, str]]] = []
    for t in range(T):
        gset = set()
        for g in proximity_seq[t].get("groups", []):
            members = tuple(sorted(g["members"]))
            gset.add(members)
        grouped_sets.append(gset)

    # Track state with hysteresis counters per pair
    # state[pair] in {"grouped", "ungrouped"}
    state: Dict[Tuple[str, str], str] = {}
    enter_cnt: Dict[Tuple[str, str], int] = {}
    exit_cnt: Dict[Tuple[str, str], int] = {}

    # Collect all pairs seen anywhere
    all_pairs: set[Tuple[str, str]] = set()
    for t in range(T):
        pairs = {(i["a"], i["b"]) for i in proximity_seq[t].get("pairs", [])}
        all_pairs |= { _pair_key(a,b) for (a,b) in pairs }

    events: List[Dict[str, Any]] = []

    def is_grouped_instant(t: int, pair: Tuple[str, str]) -> bool:
        return pair in grouped_sets[t]

    for t in range(T):
        # optional: move_toward events at transition t-1->t (emit at t)
        if p.emit_move_toward and move_toward_seq is not None and t > 0:
            for (a, b), flag in move_toward_seq[t-1].items():
                if flag:
                    events.append({
                        "type": "move_toward",
                        "subject": a,
                        "target": b,
                        "frame": t,
                        "principle": p.principle_name
                    })

        for pair in sorted(all_pairs):
            inst = is_grouped_instant(t, pair)
            st = state.get(pair, "ungrouped")

            if st == "ungrouped":
                # check entering condition (instant grouped)
                if inst:
                    c = enter_cnt.get(pair, 0) + 1
                    enter_cnt[pair] = c
                    exit_cnt[pair] = 0
                    if c >= p.k_stable:
                        # confirm MERGE
                        state[pair] = "grouped"
                        enter_cnt[pair] = 0
                        events.append({
                            "type": "merge",
                            "members": list(pair),
                            "frame": t,
                            "principle": p.principle_name
                        })
                else:
                    enter_cnt[pair] = 0
                    exit_cnt[pair] = 0
                    if p.include_persist:
                        # persist "ungrouped" is usually not emitted; skip
                        pass

            else:  # st == "grouped"
                # check leaving condition (instant ungrouped)
                if not inst:
                    c = exit_cnt.get(pair, 0) + 1
                    exit_cnt[pair] = c
                    enter_cnt[pair] = 0
                    if c >= p.k_stable:
                        # confirm DISSOLVE
                        state[pair] = "ungrouped"
                        exit_cnt[pair] = 0
                        events.append({
                            "type": "dissolve",
                            "members": list(pair),
                            "frame": t,
                            "principle": p.principle_name
                        })
                else:
                    exit_cnt[pair] = 0
                    enter_cnt[pair] = 0
                    if p.include_persist:
                        events.append({
                            "type": "persist",
                            "members": list(pair),
                            "frame": t,
                            "principle": p.principle_name
                        })

    return events