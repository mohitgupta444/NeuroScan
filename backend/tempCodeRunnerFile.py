"""
Flask backend for the Brain Tumor Detection dashboard.

Endpoints:
  POST /predict            -> runs CNN + U-Net on an uploaded MRI image,
                               saves the scan + result to the database,
                               returns JSON result (includes scan_id)
  POST /explain             -> given a scan_id, asks Mistral to explain the
                               result, saves the exchange to chat history
  POST /chat                -> follow-up question about a scan, answered by
                               Mistral, saved to chat history
  GET  /report/<scan_id>    -> generates and returns a PDF report for a scan
  GET  /history             -> lists all past scans (id, date, top class, risk)
  GET  /history/<scan_id>   -> full detail for one past scan + its chat log

Run:
    python app.py
Server starts on http://localhost:5000 by default.
"""

import io
import os

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image

from inference import TumorPipeline
from mistral_explainer import explain_result, follow_up
from report_generator import build_report_pdf
import history_db as db

app = Flask(__name__)
CORS(app)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CNN_WEIGHTS = os.path.join(
    BASE_DIR, "saved_models", "cnn_classifier.pth"
)

UNET_WEIGHTS = os.path.join(
    BASE_DIR, "saved_models", "unet_segmentation.pth"
)

pipeline = None  # lazy-loaded on first request so the server still starts
                  # even if models haven't been trained yet


def get_pipeline():
    global pipeline
    if pipeline is None:
        if not (os.path.exists(CNN_WEIGHTS) and os.path.exists(UNET_WEIGHTS)):
            raise FileNotFoundError(
                "Model weights not found. Train them first:\n"
                "  python train_classifier.py\n"
                "  python generate_pseudo_masks.py --split Training\n"
                "  python generate_pseudo_masks.py --split Testing\n"
                "  python train_segmentation.py"
            )
        pipeline = TumorPipeline(CNN_WEIGHTS, UNET_WEIGHTS)
    return pipeline


@app.route("/health", methods=["GET"])
def health():
    cnn_exists = os.path.exists(CNN_WEIGHTS)
    unet_exists = os.path.exists(UNET_WEIGHTS)

    return jsonify({
        "status": "ok",
        "models_ready": cnn_exists and unet_exists,
        "cnn_model": cnn_exists,
        "unet_model": unet_exists,
        "cnn_path": CNN_WEIGHTS,
        "unet_path": UNET_WEIGHTS
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided (expected form field 'image')"}), 400

    try:
        pipe = get_pipeline()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    file = request.files["image"]
    image_bytes = file.read()
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        return jsonify({"error": f"Could not read image: {e}"}), 400

    result = pipe.run(pil_img)

    # Persist the scan so it can be re-fetched later for a PDF report or
    # revisited in scan history, with its own chat thread.
    scan_id = db.create_scan(image_bytes, result)
    result["scan_id"] = scan_id

    return jsonify(result)


@app.route("/explain", methods=["POST"])
def explain():
    """
    Expects JSON body: { "result": <the JSON returned by /predict> }
    (result must include "scan_id" so the explanation gets saved to history)
    Returns: { "error": bool, "message": str }
    """
    data = request.get_json(force=True, silent=True)
    if not data or "result" not in data:
        return jsonify({"error": True, "message": "Missing 'result' in request body"}), 400

    result = data["result"]
    response = explain_result(result)

    scan_id = result.get("scan_id")
    if scan_id and not response.get("error"):
        db.add_message(scan_id, "assistant", response["message"])

    return jsonify(response)


@app.route("/chat", methods=["POST"])
def chat():
    """
    Expects JSON body:
      {
        "result": <the JSON returned by /predict, including scan_id>,
        "history": [{"role": "user"|"assistant", "content": "..."}],
        "question": "user's follow-up question"
      }
    Returns: { "error": bool, "message": str }
    """
    data = request.get_json(force=True, silent=True)
    if not data or "result" not in data or "question" not in data:
        return jsonify({"error": True, "message": "Missing 'result' or 'question' in request body"}), 400

    result = data["result"]
    history = data.get("history", [])
    question = data["question"]

    response = follow_up(result, history, question)

    scan_id = result.get("scan_id")
    if scan_id:
        db.add_message(scan_id, "user", question)
        if not response.get("error"):
            db.add_message(scan_id, "assistant", response["message"])

    return jsonify(response)


@app.route("/report/<scan_id>", methods=["GET"])
def report(scan_id):
    """Generates a full clinical-style PDF report for a saved scan."""
    scan = db.get_scan(scan_id)
    if scan is None:
        return jsonify({"error": "Scan not found"}), 404

    import base64
    original_bytes = base64.b64decode(scan["original_image_b64"])
    result = scan["result"]

    # Pull the latest assistant explanation from chat history, if any,
    # to include in the PDF.
    messages = db.get_messages(scan_id)
    explanation_text = None
    for m in messages:
        if m["role"] == "assistant":
            explanation_text = m["content"]
            break  # first assistant message is the initial /explain output

    from datetime import datetime
    created_str = datetime.fromtimestamp(scan["created_at"]).strftime("%Y-%m-%d %H:%M")

    pdf_bytes = build_report_pdf(
        result, original_bytes,
        scan_id=scan_id, created_at_str=created_str,
        explanation_text=explanation_text,
    )

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"NeuroScan_Report_{scan_id}.pdf",
    )


@app.route("/history", methods=["GET"])
def history_list():
    """Lists past scans, most recent first."""
    scans = db.list_scans(limit=50)
    return jsonify({"scans": scans})


@app.route("/history/<scan_id>", methods=["GET"])
def history_detail(scan_id):
    """Full detail for one past scan, including its chat history."""
    scan = db.get_scan(scan_id)
    if scan is None:
        return jsonify({"error": "Scan not found"}), 404

    messages = db.get_messages(scan_id)
    return jsonify({
        "scan_id": scan_id,
        "created_at": scan["created_at"],
        "result": scan["result"],
        "original_image_b64": scan["original_image_b64"],
        "chat_history": messages,
    })


if __name__ == "__main__":
    print("Starting Brain Tumor Detection backend on http://localhost:5000")
    print("Endpoints: /health /predict /explain /chat /report/<id> /history /history/<id>")
    app.run(host="0.0.0.0", port=5000, debug=True)
