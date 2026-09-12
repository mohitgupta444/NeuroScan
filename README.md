# NeuroScan — Brain Tumor Detection Pipeline

CNN classification + U-Net segmentation + Mistral-powered plain-language
explanation and follow-up chat, for brain MRI scans.

Built for a laptop with **8GB RAM and integrated graphics** (no dedicated
GPU required) — every model here is intentionally lightweight.

---

## 1. What's inside

```
brain-tumor-project/
├── backend/
│   ├── app.py                      Flask server (predict / explain / chat)
│   ├── inference.py                Combined CNN + U-Net pipeline + risk logic
│   ├── mistral_explainer.py        Mistral via LangChain (explain + follow-up chat)
│   ├── train_classifier.py         Trains the CNN on your dataset
│   ├── generate_pseudo_masks.py    Grad-CAM -> Otsu -> contour pseudo-masks
│   ├── train_segmentation.py       Trains U-Net on the pseudo-masks
│   ├── requirements.txt
│   ├── models/
│   │   ├── cnn_classifier.py       MobileNetV2-based classifier
│   │   └── unet_model.py           Lightweight from-scratch U-Net
│   ├── saved_models/               trained .pth files land here
│   └── pseudo_masks/                generated masks land here
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── dataset/                        your MRI dataset (Training/ + Testing/)
└── README.md
```

## 2. Why U-Net is trained on "pseudo-masks"

Your dataset (Kaggle-style: glioma / meningioma / notumor / pituitary) has
**classification labels only** — no pixel-level tumor masks. Training a real
U-Net needs a mask target, so this project builds one automatically:

1. Train the CNN classifier first.
2. For each tumor image, run **Grad-CAM** on the trained CNN to see which
   pixels drove its prediction.
3. Apply **Otsu thresholding** to the heatmap (auto-picks a cutoff, no manual
   tuning) and keep only the **largest contour** to get a clean blob.
4. That blob becomes the "ground truth" mask U-Net trains on.

This is a legitimate weakly-supervised technique, but be transparent about
it: the resulting segmentation is a proxy for "where the CNN is looking," not
a radiologist-verified tumor boundary. The app's disclaimer text says this
explicitly — keep it there if you present this project.

## 3. Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If `pip install torch` is slow/large on your machine, install the CPU-only
build directly from https://pytorch.org/get-started/locally/ (select CPU,
your OS, pip) — it's smaller and enough for this project.

## 4. Train the models (run once)

All commands run from inside `backend/`.

```bash
# Step 1: train the CNN classifier (~10-30 min on CPU depending on epochs)
python train_classifier.py --data_dir ../dataset --epochs 8

# Step 2: generate pseudo-masks using the trained CNN (Grad-CAM + Otsu)
python generate_pseudo_masks.py --data_dir ../dataset --split Training
python generate_pseudo_masks.py --data_dir ../dataset --split Testing

# Step 3: train U-Net on those pseudo-masks (~15-30 min on CPU)
python train_segmentation.py --data_dir ../dataset --mask_dir pseudo_masks --epochs 12
```

After this, `backend/saved_models/` should contain:
- `cnn_classifier.pth`
- `unet_segmentation.pth`

Tip: if training feels slow, lower `--epochs`, or reduce `--batch_size` if
you hit memory errors. 8 epochs for the CNN and 12 for U-Net is a reasonable
starting point on this dataset size (5600 train images).

## 5. Set your Mistral API key

```bash
# macOS/Linux
export MISTRAL_API_KEY="your-key-here"

# Windows (PowerShell)
$env:MISTRAL_API_KEY="your-key-here"
```

Or edit the placeholder directly in `backend/mistral_explainer.py`
(`MISTRAL_API_KEY = "PASTE_YOUR_MISTRAL_API_KEY_HERE"`) — the environment
variable is the safer option so you don't commit your key anywhere.

Get a key at https://console.mistral.ai/.

Note: `mistral_explainer.py` uses **LangChain** (`langchain-mistralai`'s
`ChatMistralAI` chat model, with `ChatPromptTemplate` / `MessagesPlaceholder`
for the follow-up chat history) rather than calling the Mistral REST API
directly. Both `explain_result()` and `follow_up()` keep the same function
signatures either way, so `app.py` didn't need to change.

## 6. Run the backend

```bash
cd backend
python app.py
```

Server starts at `http://localhost:5000`. Check `http://localhost:5000/health`
in a browser — it should say `"models_ready": true` once training is done.

## 7. Run the frontend

The frontend is plain HTML/CSS/JS with no build step. Simplest option:

```bash
cd frontend
python -m http.server 8000
```

Then open `http://localhost:8000` in your browser. (Opening `index.html`
directly by double-clicking also works in most browsers, but serving it
avoids occasional CORS quirks with local file uploads.)

## 8. Using the app

1. Upload an MRI slice (the `dataset/Testing/...` folders have sample images
   you can try).
2. Click **Run pipeline** — this calls `/predict`, which runs the CNN then
   U-Net and returns classification, mask, size, location, and risk band.
3. The result is automatically sent to **Mistral** via `/explain`, which
   returns a plain-language summary in the chat panel.
4. Type follow-up questions in the chat box (e.g. *"why is this considered
   moderate risk?"*, *"what does glioma mean?"*) — these go to `/chat`, which
   keeps the original result as context so answers stay grounded in the
   actual numbers.

## 9. Known limitations (be upfront about these)

- **Segmentation masks are pseudo-labels**, not clinically verified — see
  section 2. Treat area/diameter/location as approximate, not measurements.
- **mm calibration is an assumption** (`mm_per_px = 0.7` in `inference.py`),
  not derived from actual scan metadata (DICOM pixel spacing). Real deployment
  would need to read that from the scan file.
- **Risk band is a simple rule** (size + tumor type thresholds), not a
  validated clinical risk score.
- This is a **research/education prototype**, not a diagnostic device. The
  UI and Mistral prompt both say this — keep those disclaimers if you extend
  the project.

## 10. Extending it

- Swap MobileNetV2 for EfficientNet-B0 if you later get access to a GPU —
  `models/cnn_classifier.py` isolates the backbone so this is a one-line
  change.
- If you obtain a real tumor-mask dataset (e.g. BraTS) later, you can retrain
  `train_segmentation.py` directly on those masks instead of pseudo-masks —
  the `TumorSegDataset` class already expects `(image, mask)` pairs, so no
  architecture change is needed.
