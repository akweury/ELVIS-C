
# scripts/gen_pairs.py
"""
Generate K intervention pairs (baseline vs do(·)) with diversity and parallelism.
Usage:
  python scripts/gen_pairs.py --num_pairs 1000 --out data/pairs --workers 8
"""
import os, json, random, argparse
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from elvis_env.pairing.make_pairs import make_intervention_pair
from elvis_env.io.export import export_pair

# ------- intervention samplers (diversity) ---------------------------------
def _rand_color():
    # RGB ints [0,255]
    return [random.randint(0,255), random.randint(0,255), random.randint(0,255)]

def sample_do(op_pool=("freeze","move","change_color","remove","set_pos")):
    op = random.choice(op_pool)
    if op == "freeze":
        return {"op":"freeze","obj":"A"}
    if op == "move":
        # small normalized displacement
        dx = random.uniform(-0.2, 0.2)
        dy = random.uniform(-0.2, 0.2)
        return {"op":"move","obj":"A","dx":dx,"dy":dy}
    if op == "change_color":
        return {"op":"change_color","obj":"A","color":_rand_color()}
    if op == "remove":
        return {"op":"remove","obj":"A"}
    if op == "set_pos":
        return {"op":"set_pos","obj":"A","pos":[random.uniform(0.1,0.9), random.uniform(0.1,0.9)]}
    return {"op":"freeze","obj":"A"}

# ------- worker -------------------------------------------------------------
def _make_and_export(i, cfg_path, out_root, num_frames, t_intervene, op_pool, export_gif=True, gif_fps=3):
    seed = random.randint(0, 2_000_000_000)
    do = sample_do(op_pool)
    pair = make_intervention_pair(
        cfg_path=cfg_path,
        seed=seed,
        do=do,
        t_intervene=t_intervene,
        num_frames=num_frames
    )
    out_dir = os.path.join(out_root, f"scene_{i:05d}")
    export_pair(pair, out_dir=out_dir, export_gif=export_gif, gif_fps=gif_fps)
    # simple manifest line return
    return {
        "idx": i,
        "seed": seed,
        "out_dir": out_dir,
        "do": do,
        "num_frames": num_frames,
        "t_intervene": t_intervene
    }

# ------- main ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/default.yaml")
    ap.add_argument("--out", default="data/pairs")
    ap.add_argument("--num_pairs", type=int, default=1000)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--t_intervene", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ops", type=str, default="freeze,move,change_color,remove,set_pos",
                    help="comma-separated ops from {freeze,move,change_color,remove,set_pos}")
    ap.add_argument("--export_gif", action="store_true", default=True,
                    help="Export animated GIF files for easy viewing (default: True)")
    ap.add_argument("--no_gif", action="store_true",
                    help="Disable GIF export to save space and time")
    ap.add_argument("--gif_fps", type=int, default=3,
                    help="Frame rate for GIF animation (default: 3)")
    args = ap.parse_args()

    # Handle GIF export flag
    export_gif = args.export_gif and not args.no_gif

    os.makedirs(args.out, exist_ok=True)
    op_pool = tuple([s.strip() for s in args.ops.split(",") if s.strip()])

    worker = partial(
        _make_and_export,
        cfg_path=args.cfg,
        out_root=args.out,
        num_frames=args.num_frames,
        t_intervene=args.t_intervene,
        op_pool=op_pool,
        export_gif=export_gif,
        gif_fps=args.gif_fps
    )
    
    
    manifests = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = [ex.submit(worker, i) for i in range(args.num_pairs)]
        for fut in tqdm(as_completed(futures), total=args.num_pairs, desc="Generating pairs"):
            manifests.append(fut.result())

    # write a dataset-level manifest for easy loading
    manifest_path = os.path.join(args.out, "manifest.jsonl")
    with open(manifest_path, "w") as f:
        for m in sorted(manifests, key=lambda x: x["idx"]):
            f.write(json.dumps(m) + "\n")

    gif_status = f"with GIFs (fps={args.gif_fps})" if export_gif else "without GIFs"
    print(f"\n✅ Done. Generated {args.num_pairs} pairs at: {args.out} ({gif_status})")
    print(f"📄 Manifest: {manifest_path}")

if __name__ == "__main__":
    main()