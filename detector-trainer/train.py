import argparse
import json
from pathlib import Path
import time

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Subset
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from tqdm import tqdm


class BinaryTargetTransform:
    def __init__(self, fake_idx: int):
        self.fake_idx = fake_idx

    def __call__(self, label: int) -> int:
        return 1 if label == self.fake_idx else 0


def build_model(model_name: str, pretrained: bool = True) -> nn.Module:
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, 1))
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def get_binary_target_transform(class_to_idx):
    normalized = {name.lower(): idx for name, idx in class_to_idx.items()}
    fake_names = ["fake", "ai_generated", "ai-generated", "generated", "synthetic"]
    real_names = ["real", "authentic", "human", "natural"]

    fake_idx = next((normalized[name] for name in fake_names if name in normalized), None)
    real_idx = next((normalized[name] for name in real_names if name in normalized), None)

    if fake_idx is None or real_idx is None:
        raise ValueError(
            "Expected one real class and one fake/generated class. "
            f"Found classes: {sorted(class_to_idx.keys())}"
        )

    return BinaryTargetTransform(fake_idx)


def get_data_loaders(data_dir: Path, image_size: int, batch_size: int, num_workers: int, val_split: float):
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    transform_train = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    transform_eval = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class_probe = datasets.ImageFolder(train_dir)
    target_transform = get_binary_target_transform(class_probe.class_to_idx)

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=transform_train,
        target_transform=target_transform,
    )

    if val_dir.exists() and any(val_dir.iterdir()):
        val_dataset = datasets.ImageFolder(
            val_dir,
            transform=transform_eval,
            target_transform=target_transform,
        )
    else:
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        generator = torch.Generator().manual_seed(42)
        train_subset, val_subset = random_split(train_dataset, [train_size, val_size], generator=generator)
        eval_dataset = datasets.ImageFolder(
            train_dir,
            transform=transform_eval,
            target_transform=target_transform,
        )
        train_dataset = train_subset
        val_dataset = Subset(eval_dataset, val_subset.indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, train_dataset, val_dataset, class_probe.class_to_idx


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return loss_sum / total, correct / total


def save_metadata(output_dir: Path, class_to_idx):
    metadata = {
        "class_to_idx": class_to_idx,
        "output": {
            "activation": "sigmoid",
            "positive_class": "FAKE",
            "negative_class": "REAL",
            "score_meaning": "probability_fake_or_ai_generated",
            "threshold": 0.5,
        },
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train an AI image detector.")
    parser.add_argument("--data-dir", required=True, help="Root data directory containing train/ and optional val/ folders.")
    parser.add_argument("--output-dir", required=True, help="Directory to save model checkpoints.")
    parser.add_argument("--model", default="resnet50", help="Backbone model architecture.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split fraction if val/ is not provided.")
    parser.add_argument("--image-size", type=int, default=224, help="Image size for training.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, train_dataset, val_dataset, class_to_idx = get_data_loaders(
        data_dir, args.image_size, args.batch_size, args.num_workers, args.val_split
    )
    print(f"Detected classes: {class_to_idx}")
    print("Training target: REAL=0, FAKE=1. Sigmoid output is fake/AI-generated probability.")

    model = build_model(args.model, pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        for images, labels in loop:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            epoch_correct += (preds == labels).sum().item()
            epoch_total += images.size(0)

            loop.set_postfix(loss=epoch_loss / epoch_total, acc=epoch_correct / epoch_total)

        train_loss = epoch_loss / epoch_total
        train_acc = epoch_correct / epoch_total
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"Saved best model (val acc={val_acc:.4f})")

    torch.save(model.state_dict(), output_dir / "final_model.pt")
    save_metadata(output_dir, class_to_idx)

    elapsed = time.time() - start_time
    print(f"Training complete in {elapsed:.0f} seconds. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
