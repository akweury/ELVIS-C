# Created by MacBook Pro at 23.07.25


import argparse
from src import config

import os



def evaluate_model(model_entry, principle, batch_size, data_path, device, img_num, epochs):
    model_name = model_entry["name"]
    model_module = model_entry["module"]

    print(f"{principle} Evaluating {model_name} on {device}...")
    model_module(data_path, principle, batch_size, device=device, img_num=img_num, epochs=epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline models with CUDA support.")
    parser.add_argument("--principle", type=str, required=True, help="Specify the principle to filter data.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device_id", type=int, help="Specify GPU device ID. If not provided, CPU will be used.")
    parser.add_argument("--epochs", type=int,default=10)
    parser.add_argument("--img_num", type=int, default=5)
    parser.add_argument("--batch_size", type=int)
    args = parser.parse_args()

    if args.device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from src.models import vit
    # from src.models import llava
    from src.models import deepseek
    import torch

    # List of baseline models
    baseline_models = [
        {"name": "ViT-Base-Patch32-384", "module": vit.run_vit},
        # {"name": "Llava", "module": llava.run_llava},
        {"name": "deepseek", "module": deepseek.run_deepseek},
    ]

    device = "cuda:0" if torch.cuda.is_available() and args.device_id is not None else "cpu"
    # # Determine device based on device_id flag
    # if args.device_id is not None and torch.cuda.is_available():
    #     device = f"cuda:{args.device_id}"
    # else:
    #     device = "cpu"

    # Construct the data path based on the principle argument
    prin_path = config.raw_patterns/args.principle
    print(f"Starting model evaluations with data from {str(prin_path)}...")

    if args.model == "llava":
        model = baseline_models[1]
    elif args.model == "vit":
        model = baseline_models[0]
    elif args.model == "deepseek":
        model = baseline_models[2]
    else:
        raise ValueError(f"Model {args.model} is not supported. Choose from {', '.join([m['name'] for m in baseline_models])}.")

    evaluate_model(model, args.principle, args.batch_size, prin_path, device, args.img_num, args.epochs)

    print("All model evaluations completed.")
