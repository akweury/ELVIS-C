# Created by MacBook Pro at 24.07.25


from vlm2vec.src.arguments import ModelArguments, DataArguments
from vlm2vec.src.model.model import MMEBModel
from vlm2vec.src.model.processor import load_processor, QWEN2_VL, VLM_VIDEO_TOKENS
from vlm2vec.src.utils import batch_to_device
from vlm2vec.src.model.vlm_backbone.qwen2_vl.qwen_vl_utils import process_vision_info

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image
import torch
import os
import wandb
import json
from pathlib import Path
from src.models import conversations
from src.utils.data_utils import load_videos


def init_wandb(batch_size):
    wandb.init(project="Gestalt-C-Baseline", config={"batch_size": batch_size})


# def run_vlm2vec_legacy(data_path, principle, batch_size, device, img_num, epochs):
#     # Setup model and processor
#     model_args = ModelArguments(
#         model_name='VLM2Vec/VLM2Vec-V2.0',
#         pooling='last',
#         normalize=True,
#         model_backbone='qwen2_vl',
#         lora=True
#     )
#     data_args = DataArguments()
#     processor = load_processor(model_args, data_args)
#     model = MMEBModel.load(model_args).to(device, dtype=torch.bfloat16)
#     model.eval()
#
#     # List of video tasks to evaluate
#     video_tasks = ["task1", "task2", "task3"]  # Replace with actual task names
#     metrics = {}
#
#     for task in video_tasks:
#         # Load task-specific data
#         task_data_path = os.path.join(data_path, task)
#         # Implement task-specific data loading here
#
#         # Prepare batch (adapt to your dataset)
#         processor_inputs = {
#             "text": [...],  # List of texts for this batch
#             "images": [...],  # List of PIL Images for this batch
#         }
#         inputs = Qwen2_VL_process_fn(processor_inputs, processor)
#         inputs = batch_to_device(inputs, device)
#         with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
#             qry_output = model(qry=inputs)["qry_reps"]
#
#         candidate_texts = [...]  # List of candidate texts
#         candidate_inputs = Qwen2_VL_process_fn({"text": candidate_texts, "images": [None] * len(candidate_texts)}, processor)
#         candidate_inputs = batch_to_device(candidate_inputs, device)
#         with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
#             tgt_output = model(tgt=candidate_inputs)["tgt_reps"]
#
#         similarity = model.compute_similarity(qry_output, tgt_output)
#
#         # Compute metrics for this task (e.g., accuracy, recall, etc.)
#         task_metrics = {
#             "accuracy": ...,  # Compute accuracy
#             "recall": ...,  # Compute recall
#             "f1": ...,  # Compute F1 score
#         }
#         metrics[task] = task_metrics
#
#     return metrics

def load_vlm2vec_model(device):
    model_args = ModelArguments(
        model_name='Qwen/Qwen2-VL-7B-Instruct',
        checkpoint_path='TIGER-Lab/VLM2Vec-Qwen2VL-7B',
        pooling='last',
        normalize=True,
        model_backbone='qwen2_vl',
        lora=True
    )
    data_args = DataArguments()

    processor = load_processor(model_args, data_args)
    model = MMEBModel.load(model_args)
    model = model.to(device)
    model.eval()

    return model, processor


def infer_logic_rules(model, processor, train_positive, train_negative, device, principle):
    # Collect video representations for all training videos
    video_reps = []
    video_labels = []

    # Process positive videos
    for frames in train_positive:
        inputs = processor(
            text=f'{VLM_VIDEO_TOKENS[QWEN2_VL]} Represent the given video.',
            videos=[frames],
            return_tensors="pt"
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        inputs['pixel_values_videos'] = inputs['pixel_values_videos'].unsqueeze(0)
        inputs['video_grid_thw'] = inputs['video_grid_thw'].unsqueeze(0)
        with torch.no_grad():
            rep = model(qry=inputs)["qry_reps"]
        video_reps.append(rep)
        video_labels.append("positive")

    # Process negative videos
    for frames in train_negative:
        inputs = processor(
            text=f'{VLM_VIDEO_TOKENS[QWEN2_VL]} Represent the given video.',
            videos=[frames],
            return_tensors="pt"
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        inputs['pixel_values_videos'] = inputs['pixel_values_videos'].unsqueeze(0)
        inputs['video_grid_thw'] = inputs['video_grid_thw'].unsqueeze(0)
        with torch.no_grad():
            rep = model(qry=inputs)["qry_reps"]
        video_reps.append(rep)
        video_labels.append("negative")

    # Stack all video representations
    all_video_reps = torch.cat(video_reps, dim=0)

    # Prepare reasoning input: concatenate video labels as text
    reasoning_text = (
        f"Given these videos labeled as {', '.join(video_labels)}, "
        f"what is the common logic pattern in the positive videos for the principle '{principle}'?"
    )

    # Use the model to reason and generate the logic pattern
    reasoning_inputs = processor(
        text=reasoning_text,
        videos=None,
        return_tensors="pt"
    )
    reasoning_inputs = {key: value.to(device) for key, value in reasoning_inputs.items()}
    with torch.no_grad():
        output = model.generate(**reasoning_inputs)
    logic_text = processor.decode(output[0], skip_special_tokens=True)
    #
    # messages = conversations.vlm2vec_messages(train_positive, train_negative, principle)
    #
    # image_inputs, video_inputs = process_vision_info(messages)
    # inputs = processor(text=f'{VLM_VIDEO_TOKENS[QWEN2_VL]} Represent the given video.', videos=video_inputs, return_tensors="pt")
    # inputs = {key: value.to('cuda') for key, value in inputs.items()}
    # inputs['pixel_values_videos'] = inputs['pixel_values_videos'].unsqueeze(0)
    # inputs['video_grid_thw'] = inputs['video_grid_thw'].unsqueeze(0)
    # qry_output = model(qry=inputs)["qry_reps"]
    #
    # string = 'A man in a gray sweater plays fetch with his dog in the snowy yard, throwing a toy and watching it run.'
    # inputs = processor(text=string, images=None, return_tensors="pt")
    # inputs = {key: value.to('cuda') for key, value in inputs.items()}
    # tgt_output = model(tgt=inputs)["tgt_reps"]
    # print(string, '=', model.compute_similarity(qry_output, tgt_output))
    #
    # string = 'A person dressed in a blue jacket shovels the snow-covered pavement outside their house.'
    # inputs = processor(text=string, images=None, return_tensors="pt")
    # inputs = {key: value.to('cuda') for key, value in inputs.items()}
    # tgt_output = model(tgt=inputs)["tgt_reps"]
    # print(string, '=', model.compute_similarity(qry_output, tgt_output))

    return logic_text

def evaluate_vlm2vec(model, processor, test_images, logic_rules, device, principle, threshold=0.5):
    y_true = []
    y_pred = []
    similarities = []

    # Prepare logic rule representation
    logic_inputs = processor(
        text=logic_rules,
        images=None,
        return_tensors="pt"
    )
    logic_inputs = {key: value.to(device) for key, value in logic_inputs.items()}
    with torch.no_grad():
        logic_rep = model(tgt=logic_inputs)["tgt_reps"]

    for frames, label in test_images:
        # Prepare video representation
        video_inputs = processor(
            text=f'{VLM_VIDEO_TOKENS[QWEN2_VL]} Represent the given video.',
            videos=[frames],
            return_tensors="pt"
        )
        video_inputs = {key: value.to(device) for key, value in video_inputs.items()}
        video_inputs['pixel_values_videos'] = video_inputs['pixel_values_videos'].unsqueeze(0)
        video_inputs['video_grid_thw'] = video_inputs['video_grid_thw'].unsqueeze(0)
        with torch.no_grad():
            video_rep = model(qry=video_inputs)["qry_reps"]

        # Compute similarity
        similarity = model.compute_similarity(video_rep, logic_rep).item()
        similarities.append(similarity)
        pred = 1 if similarity >= threshold else 0
        y_pred.append(pred)
        y_true.append(label)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return accuracy, f1, precision, recall


def run_vlm2vec(data_path, principle, batch_size, device, img_num, epochs):
    init_wandb(batch_size)
    model, processor = load_vlm2vec_model(device)
    principle_path = Path(data_path)

    pattern_folders = sorted(
        [f for f in (principle_path / "train").iterdir() if f.is_dir() and not f.name.startswith('.')]
    )

    total_accuracy, total_f1 = [], []
    results = {}
    total_precision_scores = []
    total_recall_scores = []

    for pattern_folder in pattern_folders:
        train_positive_videos = load_videos(pattern_folder / "positive", img_num)
        train_negative_videos = load_videos(pattern_folder / "negative", img_num)
        test_positive_videos = load_videos((principle_path / "test" / pattern_folder.name) / "positive", img_num)
        test_negative_videos = load_videos((principle_path / "test" / pattern_folder.name) / "negative", img_num)

        # Flatten videos to list of frame paths for each video
        train_positive = [frames for frames in train_positive_videos]
        train_negative = [frames for frames in train_negative_videos]
        test_positive = [frames for frames in test_positive_videos]
        test_negative = [frames for frames in test_negative_videos]

        logic_rules = infer_logic_rules(model, processor, train_positive, train_negative, device, principle)

        test_images = [(frames, 1) for frames in test_positive] + [(frames, 0) for frames in test_negative]
        accuracy, f1, precision, recall = evaluate_vlm2vec(model, processor, test_images, logic_rules, device, principle)

        results[pattern_folder.name] = {
            "accuracy": accuracy,
            "f1_score": f1,
            "logic_rules": logic_rules,
            "precision": precision,
            "recall": recall
        }
        total_accuracy.append(accuracy)
        total_f1.append(f1)
        total_precision_scores.append(precision)
        total_recall_scores.append(recall)

    avg_accuracy = sum(total_accuracy) / len(total_accuracy) if total_accuracy else 0
    avg_f1 = sum(total_f1) / len(total_f1) if total_f1 else 0

    results["average"] = {"accuracy": avg_accuracy, "f1_score": avg_f1}
    results_path = Path(data_path) / f"deepseek_{principle}.json"
    with open(results_path, "w") as json_file:
        json.dump(results, json_file, indent=4)

    print("Evaluation complete. Results saved to evaluation_results.json.")
    print(f"Overall Average Accuracy: {avg_accuracy:.2f}% | Average F1 Score: {avg_f1:.4f}")
    wandb.finish()
    return avg_accuracy, avg_f1
