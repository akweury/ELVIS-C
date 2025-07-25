# Created by jing at 26.02.25
import random
import torch
import os
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import timm
import argparse
import json
import wandb
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
from rtpt import RTPT  # Real-Time Progress Tracker for PyTorch
from PIL import Image
from torch.utils.data import Dataset

from src import config
from src.utils import data_utils

# Configuration
# BATCH_SIZE = 8  # Increase batch size for better GPU utilization  # Reduce batch size dynamically
IMAGE_SIZE = 224  # ViT default input size
NUM_CLASSES = 2  # Positive and Negative
ACCUMULATION_STEPS = 1  # Reduce accumulation steps for faster updates  # Gradient accumulation steps


def init_wandb(batch_size, epochs, principle):
    # Initialize Weights & Biases (WandB)
    wandb.init(project=f"ELVIS-C-ViT-{principle}", config={
        "batch_size": batch_size,
        "image_size": IMAGE_SIZE,
        "num_classes": NUM_CLASSES,
        "epochs": epochs
    })


class VideoDataset(Dataset):
    def __init__(self, root_dir, frames_per_clip=16, transform=None):
        self.samples = []
        self.transform = transform
        self.frames_per_clip = frames_per_clip

        for label_dir in ['positive', 'negative']:
            label_path = root_dir / label_dir
            if not label_path.exists():
                continue
            label = 1 if label_dir == 'positive' else 0
            for example_dir in label_path.iterdir():
                frame_paths = sorted(example_dir.glob('frame_*.png'))
                if len(frame_paths) == 0:
                    continue
                self.samples.append((frame_paths, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]
        frames = []
        for i in range(self.frames_per_clip):
            if i < len(frame_paths):
                img = Image.open(frame_paths[i]).convert('RGB')
                if self.transform:
                    img = self.transform(img)
            else:
                img = torch.zeros(3, 224, 224)  # Pad missing frames
            frames.append(img)
        video_tensor = torch.stack(frames)  # (frames_per_clip, C, H, W)
        return video_tensor, label


# Python
def get_video_dataloader(data_dir, batch_size, frames_per_clip=16, num_workers=0):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = VideoDataset(data_dir, frames_per_clip=frames_per_clip, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers), len(dataset)


def get_dataloader(data_dir, batch_size, img_num, num_workers=2, pin_memory=True, prefetch_factor=None):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Group images by class
    class_to_indices = defaultdict(list)
    for idx, (image_path, label) in enumerate(dataset.samples):
        class_to_indices[label].append(idx)

    # Sample img_num images from each class
    selected_indices = []
    for label, indices in class_to_indices.items():
        if len(indices) < img_num:
            raise ValueError(f"Not enough images in class {label}. Required: {img_num}, Found: {len(indices)}")
        selected_indices.extend(random.sample(indices, img_num))

    subset_dataset = Subset(dataset, selected_indices)

    return DataLoader(subset_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                      pin_memory=pin_memory, prefetch_factor=prefetch_factor,
                      persistent_workers=(num_workers > 0)), len(subset_dataset)


# Load Pretrained ViT Model
class ViTClassifier(nn.Module):
    def save_checkpoint(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load_checkpoint(self, filepath, device):
        if Path(filepath).exists():
            self.to(device)
            self.load_state_dict(torch.load(filepath, map_location=device))
            print(f"Checkpoint loaded from {filepath}")
        else:
            print("No checkpoint found, starting from scratch.")

    def __init__(self, model_name, num_classes=NUM_CLASSES):
        super(ViTClassifier, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        self.model.set_grad_checkpointing(True)  # Enable gradient checkpointing

    def forward(self, x):
        return self.model(x)


# Training Function
def train_vit(model, train_loader, device, checkpoint_path, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-5, betas=(0.9, 0.999))  # Faster convergence
    scaler = torch.cuda.amp.GradScaler()  # Ensure AMP is enabled
    model.to(device)
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        for step, (videos, labels) in enumerate(train_loader):
            # videos: (B, T, C, H, W)
            B, T, C, H, W = videos.shape
            videos = videos.view(B * T, C, H, W).to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = model(videos)  # (B*T, num_classes)
                outputs = outputs.view(B, T, -1).mean(dim=1)  # (B, num_classes)
                loss = criterion(outputs, labels) / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                torch.cuda.empty_cache()


from sklearn.metrics import confusion_matrix


def evaluate_vit(model, test_loader, device, principle, pattern_name):
    model.to(device)
    print(f"[evaluate_vit] Model device: {next(model.parameters()).device}")

    model.eval()
    correct, total = 0, 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device != "cpu")):
                outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Accuracy Calculation
    accuracy = 100 * correct / total

    TN, FP, FN, TP = data_utils.confusion_matrix_elements(all_predictions, all_labels)
    precision, recall, f1_score = data_utils.calculate_metrics(TN, FP, FN, TP)

    print(f"TN: {TN}, FP: {FP}, FN: {FN}, TP: {TP}")
    print(
        f"({principle}) Test Accuracy for {pattern_name}: {accuracy:.2f}% | F1 Score: {f1_score:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

    print(f"True Negatives (TN): {TN}, False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}, True Positives (TP): {TP}")

    wandb.log({
        f"{principle}/test_accuracy": accuracy,
        f"{principle}/f1_score": f1_score,
        f"{principle}/precision": precision,
        f"{principle}/recall": recall
    })

    return accuracy, f1_score, precision, recall


def run_vit(data_path, principle, batch_size, device, img_num, epochs):
    init_wandb(batch_size, epochs, principle)
    model_name = "vit_base_patch16_224"
    checkpoint_path = config.output_dir / principle / f"{model_name}_{img_num}checkpoint.pth"
    model = ViTClassifier(model_name).to(device, memory_format=torch.channels_last)


    model.load_checkpoint(checkpoint_path, device)

    print(f"\n=== Training and Evaluating ViT Model on Gestalt ({principle}) Patterns ===")
    results = {}
    total_accuracy = []
    total_f1_scores = []
    total_precision_scores = []
    total_recall_scores = []

    principle_path = Path(data_path)
    results[principle] = {}

    pattern_folders = sorted([p for p in (principle_path / "train").iterdir() if p.is_dir()], key=lambda x: x.stem)

    rtpt = RTPT(name_initials='JS', experiment_name='ELVIS-C_ViT', max_iterations=len(pattern_folders))
    rtpt.start()
    print(f"Root config path: {config.root}")
    print(f"Found {len(pattern_folders)} pattern folders for training.")

    for idx, pattern_folder in enumerate(pattern_folders):
        print(f"\n--- [{idx + 1}/{len(pattern_folders)}] Training on pattern: {pattern_folder.stem} ---")
        rtpt.step(subtitle=f"")
        train_loader, num_train_images = get_video_dataloader(pattern_folder, batch_size, img_num)
        print(f"Number of training videos: {num_train_images}")
        wandb.log({f"{principle}/num_train_images": num_train_images})
        train_vit(model, train_loader, device, checkpoint_path, epochs)
        print(f"Finished training on {pattern_folder.stem}")

        torch.cuda.empty_cache()

        test_folder = Path(data_path) / "test" / pattern_folder.stem
        if test_folder.exists():
            print(f"Evaluating on test set: {test_folder}")
            test_loader, _ = get_video_dataloader(test_folder, batch_size, img_num)
            model.eval()
            correct, total = 0, 0
            all_labels, all_predictions = [], []
            with torch.no_grad():
                for videos, labels in test_loader:
                    B, T, C, H, W = videos.shape
                    videos = videos.view(B * T, C, H, W).to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    outputs = model(videos)
                    outputs = outputs.view(B, T, -1).mean(dim=1)
                    predicted = torch.argmax(outputs, dim=1)
                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            TN, FP, FN, TP = data_utils.confusion_matrix_elements(all_predictions, all_labels)
            precision, recall, f1_score = data_utils.calculate_metrics(TN, FP, FN, TP)
            print(f"Test results for {pattern_folder.stem}: Accuracy={accuracy:.2f}%, F1={f1_score:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")

            avg_f1_scores = sum(total_f1_scores) / len(total_f1_scores) if total_f1_scores else 0
            avg_accuracy = sum(total_accuracy) / len(total_accuracy) if total_accuracy else 0
            avg_precision = sum(total_precision_scores) / len(total_precision_scores) if total_precision_scores else 0
            avg_recall = sum(total_recall_scores) / len(total_recall_scores) if total_recall_scores else 0
            wandb.log({
                f"{principle}/test_accuracy": accuracy,
                f"{principle}/f1_score": f1_score,
                f"{principle}/precision": precision,
                f"{principle}/recall": recall,
                f"{principle}/average_f1_score": avg_f1_scores,
                f"{principle}/average_accuracy": avg_accuracy,
                f"{principle}/average_precision": avg_precision,
                f"{principle}/average_recall": avg_recall
            })

            results[principle][pattern_folder.stem] = {
                "accuracy": accuracy,
                "f1_score": f1_score,
                "precision": precision,
                "recall": recall
            }
            total_accuracy.append(accuracy)
            total_f1_scores.append(f1_score)
            total_precision_scores.append(precision)
            total_recall_scores.append(recall)
            torch.cuda.empty_cache()
        else:
            print(f"Test folder {test_folder} does not exist. Skipping evaluation.")

    # avg_f1_scores = sum(total_f1_scores) / len(total_f1_scores) if total_f1_scores else 0
    # avg_accuracy = sum(total_accuracy) / len(total_accuracy) if total_accuracy else 0
    # avg_precision = sum(total_precision_scores) / len(total_precision_scores) if total_precision_scores else 0
    # avg_recall = sum(total_recall_scores) / len(total_recall_scores) if total_recall_scores else 0
    #
    # wandb.log({
    #     f"average_f1_scores_{principle}": avg_f1_scores,
    #     f"average_test_accuracy_{principle}": avg_accuracy,
    #     f"average_precision_{principle}": avg_precision,
    #     f"average_recall_{principle}": avg_recall
    # })

    print(f"\n=== Average Metrics for {principle} ===")
    print(f"  - Accuracy: {avg_accuracy:.2f}%")
    print(f"  - F1 Score: {avg_f1_scores:.4f}")
    print(f"  - Precision: {avg_precision:.4f}")
    print(f"  - Recall: {avg_recall:.4f}")

    os.makedirs(config.output_dir / principle, exist_ok=True)
    results_path = config.output_dir / principle / f"{model_name}_{img_num}_evaluation_results.json"
    with open(results_path, "w") as json_file:
        json.dump(results, json_file, indent=4)

    print(f"\nTraining and evaluation complete. Results saved to {results_path}.")
    model.save_checkpoint(checkpoint_path)
    wandb.finish()


torch.set_num_threads(torch.get_num_threads())  # Utilize all available threads efficiently
os.environ['OMP_NUM_THREADS'] = str(torch.get_num_threads())  # Limit OpenMP threads
os.environ['MKL_NUM_THREADS'] = str(torch.get_num_threads())  # Limit MKL threads

torch.backends.cudnn.benchmark = True  # Optimize cuDNN for fixed image size

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate ViT model with CUDA support.")
    parser.add_argument("--device_id", type=int, help="Specify GPU device ID. If not provided, CPU will be used.")
    parser.add_argument("--img_num", type=int, default=5)
    args = parser.parse_args()

    device = f"cuda:{args.device_id}" if args.device_id is not None and torch.cuda.is_available() else "cpu"
    run_vit(config.raw_patterns, "proximity", 2, device)
