"""
Train the CNN tumor classifier on the Kaggle Brain Tumor MRI dataset.

Expected dataset layout (as provided):

    dataset/
        Training/
            glioma/*.jpg
            meningioma/*.jpg
            notumor/*.jpg
            pituitary/*.jpg
        Testing/
            glioma/*.jpg
            meningioma/*.jpg
            notumor/*.jpg
            pituitary/*.jpg

Run:
    python train_classifier.py --data_dir ../dataset --epochs 8

Designed for 8GB RAM / integrated graphics:
    - small batch size (16)
    - MobileNetV2 backbone (few params)
    - image size 224x224 (standard for MobileNet, keeps memory low)
    - trains fine on CPU in a reasonable time for a few epochs
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.cnn_classifier import TumorCNNClassifier, CLASS_NAMES


def get_dataloaders(data_dir, batch_size=16, img_size=224):
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # MobileNetV2 expects 3 channels
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(0.3),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dir = os.path.join(data_dir, "Training")
    test_dir = os.path.join(data_dir, "Testing")

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    test_ds = datasets.ImageFolder(test_dir, transform=test_tf)

    # Sanity check: ImageFolder sorts classes alphabetically, which happens
    # to match CLASS_NAMES already (glioma, meningioma, notumor, pituitary).
    assert train_ds.classes == CLASS_NAMES, (
        f"Class order mismatch: {train_ds.classes} vs expected {CLASS_NAMES}"
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../dataset")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", type=str, default="saved_models/cnn_classifier.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(args.data_dir, args.batch_size)
    print(f"Train samples: {len(train_loader.dataset)} | Test samples: {len(test_loader.dataset)}")

    model = TumorCNNClassifier(num_classes=len(CLASS_NAMES), pretrained=True).to(device)

    # Only the classifier head + last encoder blocks need aggressive
    # learning; keeping the whole encoder trainable but with a low LR
    # works fine at this dataset size (5600 training images).
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    best_acc = 0.0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        t0 = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total
        test_acc = evaluate(model, test_loader, device)
        dt = time.time() - t0

        print(f"Epoch {epoch}/{args.epochs} | "
              f"loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
              f"test_acc {test_acc:.4f} | {dt:.1f}s")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "model_state": model.state_dict(),
                "class_names": CLASS_NAMES,
                "test_acc": best_acc,
            }, args.out)
            print(f"  -> saved new best model (test_acc={best_acc:.4f}) to {args.out}")

    print(f"\nTraining done. Best test accuracy: {best_acc:.4f}")
    print(f"Model saved at: {args.out}")


if __name__ == "__main__":
    main()
