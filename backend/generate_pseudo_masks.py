"""
Generate pseudo tumor-segmentation masks from the trained CNN classifier.

Why this exists
----------------
The Kaggle dataset only has classification labels (folder = class), NOT
pixel-level tumor masks. To train a real U-Net we still need masks, so we
manufacture them automatically in three steps:

  1. Grad-CAM: ask the trained CNN "which pixels made you predict this
     tumor class?" -> produces a coarse heatmap over the image.
  2. Otsu thresholding: automatically pick a cutoff on that heatmap to
     separate "tumor-likely" from "background" pixels (no manual tuning).
  3. Contour cleanup: keep only the largest connected blob and smooth it,
     so the mask looks like a single tumor region instead of scattered
     noisy pixels.

This mask is a weak label, not ground truth -- it teaches the U-Net
roughly WHERE the CNN is looking, which is a reasonable proxy for tumor
location given we have no radiologist annotations. Images labelled
"notumor" get an empty (all-zero) mask.

Run:
    python generate_pseudo_masks.py --data_dir ../dataset --split Training
    python generate_pseudo_masks.py --data_dir ../dataset --split Testing
"""

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.cnn_classifier import TumorCNNClassifier, CLASS_NAMES

IMG_SIZE = 224
MASK_SIZE = 256  # resolution the U-Net will train at


def load_classifier(weights_path, device):
    ckpt = torch.load(weights_path, map_location=device)
    model = TumorCNNClassifier(num_classes=len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


def preprocess(pil_img):
    tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tf(pil_img).unsqueeze(0)


def grad_cam(model, x, target_class, device):
    """Standard Grad-CAM on the last conv layer of the MobileNetV2 encoder."""
    x = x.to(device)
    x.requires_grad_(False)

    activations = {}
    gradients = {}

    def fwd_hook(module, inp, out):
        activations["value"] = out

    def bwd_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    last_conv = model.encoder[-1]  # last block of MobileNetV2 features
    h1 = last_conv.register_forward_hook(fwd_hook)
    h2 = last_conv.register_full_backward_hook(bwd_hook)

    logits = model(x)
    score = logits[0, target_class]
    model.zero_grad()
    score.backward()

    h1.remove()
    h2.remove()

    acts = activations["value"][0]      # [C, h, w]
    grads = gradients["value"][0]       # [C, h, w]
    weights = grads.mean(dim=(1, 2))    # [C]

    cam = torch.zeros(acts.shape[1:], device=device)
    for c, w in enumerate(weights):
        cam += w * acts[c]

    cam = F.relu(cam)
    cam = cam / (cam.max() + 1e-8)
    cam = cam.detach().cpu().numpy()
    cam = cv2.resize(cam, (MASK_SIZE, MASK_SIZE))
    return cam  # float32, 0..1


def cam_to_mask(cam):
    """Otsu threshold + largest-contour cleanup on a Grad-CAM heatmap."""
    cam_u8 = (cam * 255).astype(np.uint8)
    cam_u8 = cv2.GaussianBlur(cam_u8, (7, 7), 0)

    _, thresh = cv2.threshold(cam_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(thresh)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        # discard tiny specks (Grad-CAM noise), keep only meaningful blobs
        if cv2.contourArea(largest) > (MASK_SIZE * MASK_SIZE) * 0.01:
            cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)

    # smooth edges
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask


def process_split(data_dir, split, weights_path, out_dir, device):
    model = load_classifier(weights_path, device)
    class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}

    split_dir = os.path.join(data_dir, split)
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(split_dir, class_name)
        out_class_dir = os.path.join(out_dir, split, class_name)
        os.makedirs(out_class_dir, exist_ok=True)

        files = [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        print(f"[{split}/{class_name}] {len(files)} images")

        for i, fname in enumerate(files):
            img_path = os.path.join(class_dir, fname)
            pil_img = Image.open(img_path).convert("L")

            if class_name == "notumor":
                mask = np.zeros((MASK_SIZE, MASK_SIZE), dtype=np.uint8)
            else:
                x = preprocess(pil_img)
                target_idx = class_to_idx[class_name]
                cam = grad_cam(model, x, target_idx, device)
                mask = cam_to_mask(cam)

            out_path = os.path.join(out_class_dir, fname.rsplit(".", 1)[0] + "_mask.png")
            cv2.imwrite(out_path, mask)

            if (i + 1) % 200 == 0:
                print(f"  ...{i+1}/{len(files)} done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../dataset")
    parser.add_argument("--split", type=str, default="Training", choices=["Training", "Testing"])
    parser.add_argument("--weights", type=str, default="saved_models/cnn_classifier.pth")
    parser.add_argument("--out_dir", type=str, default="pseudo_masks")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("NOTE: this requires cnn_classifier.pth to already exist (run train_classifier.py first).")

    process_split(args.data_dir, args.split, args.weights, args.out_dir, device)
    print("Done. Pseudo-masks saved under:", os.path.join(args.out_dir, args.split))


if __name__ == "__main__":
    main()
