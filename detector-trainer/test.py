import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from train import build_model, get_binary_target_transform


def get_test_loader(test_dir: Path, image_size: int, batch_size: int, num_workers: int):
    transform_eval = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class_probe = datasets.ImageFolder(test_dir)
    target_transform = get_binary_target_transform(class_probe.class_to_idx)
    dataset = datasets.ImageFolder(
        test_dir,
        transform=transform_eval,
        target_transform=target_transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, class_probe.class_to_idx


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device, threshold: float):
    model.eval()
    total = 0
    correct = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Testing", unit="batch"):
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            correct += (preds == labels).sum().item()
            total += images.size(0)
            tp += ((preds == 1) & (labels == 1)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

    accuracy = correct / total
    precision_fake = tp / (tp + fp) if tp + fp else 0.0
    recall_fake = tp / (tp + fn) if tp + fn else 0.0
    f1_fake = (
        2 * precision_fake * recall_fake / (precision_fake + recall_fake)
        if precision_fake + recall_fake
        else 0.0
    )
    precision_real = tn / (tn + fn) if tn + fn else 0.0
    recall_real = tn / (tn + fp) if tn + fp else 0.0
    f1_real = (
        2 * precision_real * recall_real / (precision_real + recall_real)
        if precision_real + recall_real
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "threshold": threshold,
        "confusion_matrix": {
            "true_fake_pred_fake": tp,
            "true_fake_pred_real": fn,
            "true_real_pred_fake": fp,
            "true_real_pred_real": tn,
        },
        "fake": {
            "precision": precision_fake,
            "recall": recall_fake,
            "f1": f1_fake,
        },
        "real": {
            "precision": precision_real,
            "recall": recall_real,
            "f1": f1_real,
        },
        "total": total,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate an AI image detector on a held-out test set.")
    parser.add_argument("--test-dir", required=True, help="Test directory containing class subfolders.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--output-json", default=None, help="Optional path to save metrics as JSON.")
    parser.add_argument("--model", default="resnet50", help="Backbone model architecture.")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size.")
    parser.add_argument("--image-size", type=int, default=224, help="Image size for evaluation.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Fake probability threshold.")
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    checkpoint = Path(args.checkpoint)
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader, class_to_idx = get_test_loader(test_dir, args.image_size, args.batch_size, args.num_workers)

    model = build_model(args.model, pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model = model.to(device)

    metrics = evaluate(model, loader, device, args.threshold)
    metrics["class_to_idx"] = class_to_idx
    metrics["checkpoint"] = str(checkpoint)
    metrics["test_dir"] = str(test_dir)
    metrics["positive_class"] = "FAKE"
    metrics["score_meaning"] = "probability_fake_or_ai_generated"

    print(json.dumps(metrics, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
