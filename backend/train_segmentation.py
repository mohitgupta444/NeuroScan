"""
Train the U-Net segmentation model on the pseudo-masks produced by
generate_pseudo_masks.py.

Prerequisite:
    python generate_pseudo_masks.py --data_dir ../dataset --split Training
    python generate_pseudo_masks.py --data_dir ../dataset --split Testing

Run:
    python train_segmentation.py --data_dir ../dataset --mask_dir pseudo_masks --epochs 12

Designed for 8GB RAM / integrated graphics:
    - trains at 256x256 (not 512x512) to keep memory + time reasonable
    - small batch size (8)
    - lightweight from-scratch U-Net (see models/unet_model.py), not a
      heavy pretrained segmentation backbone
"""

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.unet_model import LightUNet

IMG_SIZE = 256


class TumorSegDataset(Dataset):
    """Pairs each MRI image with its corresponding pseudo-mask."""

    def __init__(self, data_dir, mask_dir, split, class_names):
        self.samples = []
        split_img_dir = os.path.join(data_dir, split)
        split_mask_dir = os.path.join(mask_dir, split)

        for cname in class_names:
            img_class_dir = os.path.join(split_img_dir, cname)
            mask_class_dir = os.path.join(split_mask_dir, cname)
            for fname in os.listdir(img_class_dir):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img_path = os.path.join(img_class_dir, fname)
                mask_name = fname.rsplit(".", 1)[0] + "_mask.png"
                mask_path = os.path.join(mask_class_dir, mask_name)
                if os.path.exists(mask_path):
                    self.samples.append((img_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # [1, H, W]

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        mask = (mask > 127).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)  # [1, H, W]

        return torch.from_numpy(img), torch.from_numpy(mask)


def dice_coefficient(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()


def dice_loss(pred, target, eps=1e-6):
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../dataset")
    parser.add_argument("--mask_dir", type=str, default="pseudo_masks")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="saved_models/unet_segmentation.pth")
    args = parser.parse_args()

    class_names = ["glioma", "meningioma", "notumor", "pituitary"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = TumorSegDataset(args.data_dir, args.mask_dir, "Training", class_names)
    test_ds = TumorSegDataset(args.data_dir, args.mask_dir, "Testing", class_names)
    print(f"Train samples: {len(train_ds)} | Test samples: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = LightUNet(in_channels=1, out_channels=1, base=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCELoss()

    best_dice = 0.0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = bce(preds, masks) + dice_loss(preds, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_ds)

        # evaluate
        model.eval()
        dice_scores = []
        with torch.no_grad():
            for imgs, masks in test_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                dice_scores.append(dice_coefficient(preds, masks))
        mean_dice = float(np.mean(dice_scores))

        print(f"Epoch {epoch}/{args.epochs} | loss {train_loss:.4f} | val_dice {mean_dice:.4f}")

        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save({"model_state": model.state_dict(), "val_dice": best_dice}, args.out)
            print(f"  -> saved new best model (val_dice={best_dice:.4f}) to {args.out}")

    print(f"\nTraining done. Best val Dice: {best_dice:.4f}")
    print(f"Model saved at: {args.out}")
    print("\nNote: Dice here is measured against pseudo-masks (CNN Grad-CAM "
          "based), not radiologist ground truth -- treat it as a proxy for "
          "how well U-Net reproduces the CNN's attention region, not "
          "clinical segmentation accuracy.")


if __name__ == "__main__":
    main()
