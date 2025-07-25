# Created by MacBook Pro at 16.07.25

import torch
import argparse
import json
import wandb
from pathlib import Path
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForCausalLM
# from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
# from deepseek_vl2.utils.io import load_pil_images

from src.models import conversations

from src.utils import data_utils


def init_wandb(batch_size):
    wandb.init(project="Gestalt-C-Baseline", config={"batch_size": batch_size})


def load_deepseek_model(device):
    model_name = "deepseek-ai/deepseek-vl2-small"
    cache_dir = "/models/deepseek_cache"  # Ensure this is mounted in Docker

    processor = DeepseekVLV2Processor.from_pretrained(model_name, cache_dir=cache_dir)
    model = DeepseekVLV2ForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if "cuda" in device else torch.float32,
        device_map=None,
        cache_dir=cache_dir
    ).to(device)
    # model = model.to(device).eval()
    tokenizer = processor.tokenizer
    return model, processor, tokenizer


def load_images(image_dir):
    # print("img dir " + str(image_dir))
    image_paths = sorted(Path(image_dir).glob("*.png"))
    return image_paths
    # return [Image.open(img_path).convert("RGB").resize((224, 224)) for img_path in image_paths]


def load_videos(video_dir, max_videos=None):
    video_folders = sorted([f for f in Path(video_dir).iterdir() if f.is_dir()])
    if max_videos is not None:
        video_folders = video_folders[:max_videos]
    videos = []
    for folder in video_folders:
        frame_paths = sorted(folder.glob("frame_*.png"))
        videos.append(frame_paths)
    return videos


def infer_logic_rules(model, processor, train_positive, train_negative, device, principle):
    # Prepare conversation as per official example
    # print("img path:" + str(train_negative[0]))
    conversation = conversations.deepseek_conversation(train_positive, train_negative, principle)
    pil_images = load_pil_images(conversation)
    prepare_inputs = processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    )

    model_device = next(model.parameters()).device

    model = model.to(device)
    # Move all tensors in prepare_inputs to model_device
    for attr in prepare_inputs.__dict__:
        v = getattr(prepare_inputs, attr)
        if isinstance(v, torch.Tensor):
            setattr(prepare_inputs, attr, v.to(model_device))

    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

    outputs = model.generate(inputs_embeds=inputs_embeds, attention_mask=prepare_inputs.attention_mask,
                             pad_token_id=processor.tokenizer.eos_token_id,
                             bos_token_id=processor.tokenizer.bos_token_id,
                             eos_token_id=processor.tokenizer.eos_token_id,
                             max_new_tokens=512, do_sample=False, use_cache=True
                             )
    answer = processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer


# def infer_logic_rules(model, processor, train_positive, train_negative, device, principle):
#     # Prepare conversation history
#     conversations = [
#         {
#             "role": "system",
#             "content": f"You are an AI analyzing Gestalt patterns. Principle: {principle}."
#         },
#     ]
#
#     # Add positive examples
#     for img in train_positive:
#         conversations.append({
#             "role": "user",
#             "content": [{"type": "image", "image": img}, {"type": "text", "text": "Positive example"}]
#         })
#
#     # Add negative examples
#     for img in train_negative:
#         conversations.append({
#             "role": "user",
#             "content": [{"type": "image", "image": img}, {"type": "text", "text": "Negative example"}]
#         })
#
#     # Final reasoning prompt
#     conversations.append({
#         "role": "user",
#         "content": "What rule distinguishes positive from negative examples?"
#     })
#
#     # Process and generate
#     inputs = processor(conversations).to(device)
#     inputs = {k: torch.tensor(v).to(device) for k, v in inputs.items()}
#     outputs = model.generate(**inputs, max_new_tokens=512)
#     return processor.decode(outputs[0], skip_special_tokens=True)


def evaluate_deepseek(model, processor, test_images, logic_rules, device, principle):
    model.eval()
    correct, total = 0, 0
    all_labels, all_predictions = [], []
    torch.cuda.empty_cache()

    for image, label in test_images:

        conversation = conversations.deepseek_eval_conversation(image, logic_rules)
        pil_images = load_pil_images(conversation)

        inputs = processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        )

        model_device = next(model.parameters()).device

        # Move all tensors in prepare_inputs to model_device
        for attr in inputs.__dict__:
            v = getattr(inputs, attr)
            if isinstance(v, torch.Tensor):
                setattr(inputs, attr, v.to(model_device))

        inputs_embeds = model.prepare_inputs_embeds(**inputs)

        outputs = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs.attention_mask,
            pad_token_id=processor.tokenizer.eos_token_id,
            bos_token_id=processor.tokenizer.bos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )
        answer = processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

        # # inputs = tokenizer.apply_chat_template(
        # #     conversation,
        # #     add_generation_prompt=True,
        # #     return_tensors="pt"
        # # ).to(device)
        #
        # generate_ids = model.generate(
        #     inputs,
        #     max_new_tokens=10,  # Short output expected
        #     do_sample=False
        # )
        #
        #
        # processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        # prediction_label = tokenizer.decode(generate_ids[0], skip_special_tokens=True)
        #
        #

        prediction_label = answer.split("response:")[-1].strip().lower()
        # prediction_label = prediction_label.split("response:")[-1].strip().lower()
        # print(f"({label}) evaluating answer: {prediction_label}")
        predicted_label = 1 if "positive" in prediction_label else 0
        all_labels.append(label)
        all_predictions.append(predicted_label)

        total += 1
        correct += (predicted_label == label)

    accuracy = 100 * correct / total if total > 0 else 0

    TN, FP, FN, TP = data_utils.confusion_matrix_elements(all_predictions, all_labels)
    precision, recall, f1_score = data_utils.calculate_metrics(TN, FP, FN, TP)

    wandb.log({
        f"{principle}/test_accuracy": accuracy,
        f"{principle}/f1_score": f1_score,
        f"{principle}/precision": precision,
        f"{principle}/recall": recall
    })

    print(f"({principle}) Test Accuracy: {accuracy:.2f}% | F1 Score: {f1_score:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    return accuracy, f1_score, precision, recall


def run_deepseek(data_path, principle, batch_size, device, img_num, epochs):
    init_wandb(batch_size)
    model, processor, tokenizer = load_deepseek_model(device)
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
        accuracy, f1, precision, recall = evaluate_deepseek(model, processor, test_images, logic_rules, device, principle)

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
