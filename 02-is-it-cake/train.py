"""
Training script for "Is It Cake?" classifier.

Usage:
    python train.py --data_dir ../public_data --epochs 30

This script trains your model and saves the weights to model.pth.
Modify this script to experiment with different training strategies.
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

# Import your model - change this to use your own model.py
from model import create_model, count_parameters

IMG_SIZE = 64
BATCH_SIZE = 32


class CakeDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.samples = []

        cake_dir = os.path.join(data_dir, "cake")
        if os.path.exists(cake_dir):
            for f in os.listdir(cake_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(cake_dir, f), 1))

        obj_dir = os.path.join(data_dir, "object")
        if os.path.exists(obj_dir):
            for f in os.listdir(obj_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(obj_dir, f), 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(augment=False):
    """
    Get image transforms.

    You can modify the augmentation here to improve generalization.
    The evaluation will only use resize + normalize (no augmentation).
    """
    if augment:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            # Add additional augmentation here
            # ...
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


def train(args):
    # Check parameter count
    model = create_model()
    params = count_parameters(model)
    print(f"Model parameters: {params:,}")
    if params > 1_000_000:
        print("ERROR: Model exceeds 1M parameter limit!")
        return

    # Load data
    train_transform = get_transforms(augment=True)
    val_transform = get_transforms(augment=False)

    full_dataset = CakeDataset(args.data_dir)
    print(f"Total images: {len(full_dataset)}")

    # Split into train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(full_dataset), generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create datasets with appropriate transforms
    train_dataset = CakeDataset(args.data_dir, transform=train_transform)
    val_dataset = CakeDataset(args.data_dir, transform=val_transform)

    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)

    print(f"Train: {len(train_subset)}, Val: {len(val_subset)}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0
    best_state = None

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_correct = 0
        for imgs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_correct += (outputs.argmax(1) == labels).sum().item()

        # Validate
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                outputs = model(imgs)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = train_correct / len(train_subset) * 100
        val_acc = val_correct / len(val_subset) * 100

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: Train {train_acc:.1f}%, Val {val_acc:.1f}%")

    print(f"\nBest validation accuracy: {best_val_acc:.1f}%")

    # Save best model
    if best_state is not None:
        torch.save(best_state, "model.pth")
        print(f"Saved model to model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to training data")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(args)
