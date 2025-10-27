from elvis_env.envs.elvis_video_env import ElvisVideoEnv
from elvis_env.pairing.make_pairs import make_intervention_pair
from elvis_env.io.export import export_pair

def main():
    # 生成一个最小的干预 pair：baseline vs do(freeze, obj='A' at t=4)
    env_cfg = "configs/default.yaml"
    pair = make_intervention_pair(
        cfg_path=env_cfg,
        seed=123,
        do={"op": "freeze", "obj": "A"},
        t_intervene=4,
        num_frames=12
    )
    export_pair(pair, out_dir="data/samples/demo_pair", export_gif=True, gif_fps=3)
    print("Exported to data/samples/demo_pair (including GIF files)")
    print("You can view:")
    print("  - baseline.gif: Shows the normal behavior")
    print("  - intervention.gif: Shows behavior with intervention (freeze obj='A' at t=4)")

if __name__ == "__main__":
    main()