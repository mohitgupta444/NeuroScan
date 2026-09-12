"""
Combined inference pipeline: CNN classification -> U-Net segmentation ->
size/location extraction -> risk band.

This is the module app.py calls for every uploaded MRI slice.
"""

import base64
import io
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.cnn_classifier import TumorCNNClassifier, CLASS_NAMES
from models.unet_model import LightUNet

CNN_IMG_SIZE = 224
UNET_IMG_SIZE = 256

# Rough anatomical location lookup by quadrant of the mask centroid.
# This is a coarse heuristic, not real anatomical registration.
REGION_MAP = {
    ("left", "top"): "Left frontal lobe",
    ("right", "top"): "Right frontal lobe",
    ("left", "bottom"): "Left temporal / parietal region",
    ("right", "bottom"): "Right temporal / parietal region",
    ("center", "center"): "Central / sellar region",
}


class TumorPipeline:
    def __init__(self, cnn_path, unet_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cnn_ckpt = torch.load(cnn_path, map_location=self.device)
        self.cnn = TumorCNNClassifier(num_classes=len(CLASS_NAMES), pretrained=False)
        self.cnn.load_state_dict(cnn_ckpt["model_state"])
        self.cnn.to(self.device).eval()

        unet_ckpt = torch.load(unet_path, map_location=self.device)
        self.unet = LightUNet(in_channels=1, out_channels=1, base=16)
        self.unet.load_state_dict(unet_ckpt["model_state"])
        self.unet.to(self.device).eval()

        self.cnn_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((CNN_IMG_SIZE, CNN_IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # ---------- stage 1: classification ----------
    def classify(self, pil_img):
        x = self.cnn_tf(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.cnn(x)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        items = [{"name": CLASS_NAMES[i], "prob": float(probs[i])} for i in range(len(CLASS_NAMES))]
        items.sort(key=lambda d: d["prob"], reverse=True)
        return items

    # ---------- stage 2: segmentation ----------
    def segment(self, pil_img):
        gray = np.array(pil_img.convert("L"))
        resized = cv2.resize(gray, (UNET_IMG_SIZE, UNET_IMG_SIZE)).astype(np.float32) / 255.0
        x = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.unet(x)[0, 0].cpu().numpy()  # [H, W], 0..1

        mask = (pred > 0.5).astype(np.uint8) * 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {"present": False, "mask_base64": None}

        largest = max(contours, key=cv2.contourArea)
        area_px = cv2.contourArea(largest)

        if area_px < (UNET_IMG_SIZE * UNET_IMG_SIZE) * 0.003:
            return {"present": False, "mask_base64": None}

        x_, y_, w_, h_ = cv2.boundingRect(largest)
        cx, cy = x_ + w_ / 2, y_ + h_ / 2

        # crude mm calibration assumption: dataset slices are ~180mm field
        # of view across 256px after resize -> ~0.7 mm/px. This is an
        # approximation for demo purposes, not calibrated per-scan.
        mm_per_px = 0.7
        area_mm2 = area_px * (mm_per_px ** 2)
        diam_mm = max(w_, h_) * mm_per_px

        horiz = "left" if cx < UNET_IMG_SIZE * 0.42 else ("right" if cx > UNET_IMG_SIZE * 0.58 else "center")
        vert = "top" if cy < UNET_IMG_SIZE * 0.42 else ("bottom" if cy > UNET_IMG_SIZE * 0.58 else "center")
        region = REGION_MAP.get((horiz, vert), REGION_MAP[("center", "center")])

        mask_conf = float(pred[mask > 0].mean()) if (mask > 0).any() else 0.0

        # encode mask as base64 PNG so the frontend can overlay it directly
        _, buf = cv2.imencode(".png", mask)
        mask_b64 = base64.b64encode(buf).decode("utf-8")

        return {
            "present": True,
            "mask_base64": mask_b64,
            "bbox": {"x": x_, "y": y_, "w": w_, "h": h_},
            "centroid": {"x": cx, "y": cy},
            "area_mm2": round(area_mm2, 1),
            "diam_mm": round(diam_mm, 1),
            "region": region,
            "mask_confidence": round(mask_conf, 3),
            "mask_canvas_size": UNET_IMG_SIZE,
        }

    # ---------- stage 3: risk fusion ----------
    def risk_band(self, cls_items, seg_result):
        top = cls_items[0]

        if top["name"] == "notumor" or not seg_result.get("present"):
            return {"band": "low", "label": "No tumor detected",
                    "explanation": "Classifier and segmentation both found no significant tumor region."}

        diam = seg_result["diam_mm"]
        aggressive = top["name"] == "glioma"

        if diam < 20 and not aggressive:
            band, label = "low", "Low"
            note = "Small, well-defined region -- routine follow-up is typically suggested."
        elif diam < 35 or (aggressive and diam < 20):
            band, label = "moderate", "Moderate"
            note = "Size or tumor type warrants closer radiological review."
        else:
            band, label = "elevated", "Elevated"
            note = "Larger region and/or aggressive tumor type -- prioritize specialist review."

        return {"band": band, "label": label, "explanation": note}

    # ---------- full pipeline ----------
    def run(self, pil_img):
        cls_items = self.classify(pil_img)
        seg_result = self.segment(pil_img)
        risk = self.risk_band(cls_items, seg_result)

        return {
            "classification": cls_items,
            "segmentation": seg_result,
            "risk": risk,
        }
