import argparse
import random
from pathlib import Path
import shutil


def split_class(src_dir: Path, train_dir: Path, val_dir: Path, val_fraction: float, seed: int) -> None:
    files = [f for f in sorted(src_dir.iterdir()) if f.is_file()]
    if not files:
        raise ValueError(f"No files found in {src_dir}")

    random.seed(seed)
    random.shuffle(files)

    split_index = int(len(files) * val_fraction)
    val_files = files[:split_index]
    train_files = files[split_index:]

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    for src in train_files:
        dest = train_dir / src.name
        if dest.exists():
            dest = train_dir / f"{src.stem}_{random.randint(0,9999)}{src.suffix}"
        shutil.move(str(src), str(dest))

    for src in val_files:
        dest = val_dir / src.name
        if dest.exists():
            dest = val_dir / f"{src.stem}_{random.randint(0,9999)}{src.suffix}"
        shutil.move(str(src), str(dest))


def main() -> None:
    parser = argparse.ArgumentParser(description="Split train data into train + val by class.")
    parser.add_argument("--source", required=True, help="Source train directory containing class subfolders.")
    parser.add_argument("--target", required=True, help="Target base directory for train/val folders.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of source images to move into validation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits.")
    args = parser.parse_args()

    source_dir = Path(args.source)
    target_dir = Path(args.target)
    val_root = target_dir / "val"
    train_root = target_dir / "train"

    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    for class_dir in sorted([p for p in source_dir.iterdir() if p.is_dir()]):
        train_class_dir = train_root / class_dir.name
        val_class_dir = val_root / class_dir.name
        split_class(class_dir, train_class_dir, val_class_dir, args.val_fraction, args.seed)
        print(f"Split {class_dir.name}: {sum(1 for _ in train_class_dir.iterdir())} train, {sum(1 for _ in val_class_dir.iterdir())} val")

    print("Done. Make sure to keep your existing test set unchanged.")


if __name__ == "__main__":
    main()
