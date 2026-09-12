"""
Generates a clinical-style PDF report for one scan result, using ReportLab.

Report contents:
  - Header banner (NeuroScan branding, scan ID, date)
  - Side-by-side original MRI + segmentation overlay images
  - Classification results table (all class probabilities)
  - Tumor measurement table (location, area, diameter, mask confidence)
  - Risk indicator badge
  - Mistral's plain-language explanation (if available)
  - Standard disclaimer footer

Usage:
    from report_generator import build_report_pdf
    pdf_bytes = build_report_pdf(result_json, original_image_bytes, explanation_text=None)
"""
import re
from xml.sax.saxutils import escape

import base64
import io

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage,
    HRFlowable, KeepTogether,
)
from reportlab.pdfgen import canvas as pdfcanvas
from PIL import Image as PILImage
import cv2
import numpy as np

NAVY = colors.HexColor("#1B2A6B")
BLUE = colors.HexColor("#2E6FBF")
TEAL = colors.HexColor("#0F9B8E")
SLATE = colors.HexColor("#45526B")
MIST = colors.HexColor("#8A97AC")
CLOUD = colors.HexColor("#F4F6FA")
LINE = colors.HexColor("#E3E8F0")

RISK_COLORS = {
    "low": colors.HexColor("#3FA66E"),
    "moderate": colors.HexColor("#E8A33D"),
    "elevated": colors.HexColor("#E15759"),
}


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle(
        name="ReportTitle", fontName="Helvetica-Bold", fontSize=20,
        textColor=NAVY, alignment=TA_LEFT, spaceAfter=2,
    ))
    ss.add(ParagraphStyle(
        name="ReportSubtitle", fontName="Helvetica", fontSize=10,
        textColor=SLATE, alignment=TA_LEFT, spaceAfter=0,
    ))
    ss.add(ParagraphStyle(
        name="SectionHeader", fontName="Helvetica-Bold", fontSize=12.5,
        textColor=NAVY, spaceBefore=14, spaceAfter=6,
    ))
    ss.add(ParagraphStyle(
        name="BodyText2", fontName="Helvetica", fontSize=9.5,
        textColor=SLATE, leading=13.5,
    ))
    ss.add(ParagraphStyle(
        name="Disclaimer", fontName="Helvetica-Oblique", fontSize=7.5,
        textColor=MIST, leading=10.5,
    ))
    ss.add(ParagraphStyle(
        name="ImageCaption", fontName="Helvetica-Bold", fontSize=8.5,
        textColor=SLATE, alignment=TA_CENTER, spaceBefore=4,
    ))
    return ss

def _markdown_to_reportlab(text):
    """
    Convert basic Markdown from Mistral into ReportLab-compatible HTML.
    Supports:
      **bold**
      *italic*
      - bullet lists
      ### headings
      line breaks
    """
    if not text:
        return ""

    text = str(text).replace("\r\n", "\n").replace("\r", "\n")

    lines = text.split("\n")
    output = []

    for line in lines:
        line = line.strip()

        if not line:
            output.append("<br/>")
            continue

        # Headings
        if line.startswith("### "):
            content = escape(line[4:])
            output.append(f"<b>{content}</b><br/>")
            continue

        if line.startswith("## "):
            content = escape(line[3:])
            output.append(f"<b>{content}</b><br/>")
            continue

        if line.startswith("# "):
            content = escape(line[2:])
            output.append(f"<b>{content}</b><br/>")
            continue

        # Bullet points
        if line.startswith("- ") or line.startswith("* "):
            content = escape(line[2:])
            output.append(f"• {content}<br/>")
            continue

        # Escape HTML special characters first
        line = escape(line)

        # Bold: **text**
        line = re.sub(
            r"\*\*(.+?)\*\*",
            r"<b>\1</b>",
            line
        )

        # Italic: *text*
        line = re.sub(
            r"(?<!\*)\*([^*]+)\*(?!\*)",
            r"<i>\1</i>",
            line
        )

        # Inline code: `text`
        line = re.sub(
            r"`([^`]+)`",
            r"<font name='Courier'>\1</font>",
            line
        )

        output.append(line + "<br/>")

    return "".join(output)


def _make_overlay_image(original_bytes, seg_result):
    """Composite the segmentation mask (with bbox) onto the original MRI,
    matching what the frontend canvas shows, for use in the PDF."""
    pil_img = PILImage.open(io.BytesIO(original_bytes)).convert("RGB")
    size = seg_result.get("mask_canvas_size", 256)
    base = cv2.cvtColor(np.array(pil_img.resize((size, size))), cv2.COLOR_RGB2BGR)

    if seg_result.get("present") and seg_result.get("mask_base64"):
        mask_bytes = base64.b64decode(seg_result["mask_base64"])
        mask_arr = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        mask_arr = cv2.resize(mask_arr, (size, size))

        overlay = base.copy()
        overlay[mask_arr > 127] = (197, 209, 79)  # BGR cyan-ish tint
        base = cv2.addWeighted(overlay, 0.35, base, 0.65, 0)

        bbox = seg_result.get("bbox")
        if bbox:
            x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
            pad = int(max(w, h) * 0.15)
            cv2.rectangle(base, (x - pad, y - pad), (x + w + pad, y + h + pad), (60, 163, 232), 2)

    rgb = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)
    out = io.BytesIO()
    PILImage.fromarray(rgb).save(out, format="PNG")
    out.seek(0)
    return out


def _img_flowable(image_bytes_io, width_mm=70):
    img = RLImage(image_bytes_io, width=width_mm * mm, height=width_mm * mm)
    return img


def build_report_pdf(result_json, original_image_bytes, scan_id="N/A", created_at_str="N/A", explanation_text=None):
    """
    Returns PDF bytes for the full clinical-style report.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm, topMargin=14 * mm, bottomMargin=16 * mm,
    )
    ss = _styles()
    story = []

    # ---------------- Header ----------------
    header_table = Table(
        [[
            Paragraph("NeuroScan", ss["ReportTitle"]),
            Paragraph(f"Scan ID: {scan_id}<br/>Generated: {created_at_str}", ss["BodyText2"]),
        ]],
        colWidths=[110 * mm, 62 * mm],
    )
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
    ]))
    story.append(header_table)
    story.append(Paragraph("AI-Assisted Brain MRI Analysis Report", ss["ReportSubtitle"]))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.4, color=NAVY))
    story.append(Spacer(1, 10))

    # ---------------- Images ----------------
    classification = result_json.get("classification", [])
    segmentation = result_json.get("segmentation", {})
    risk = result_json.get("risk", {})

    story.append(Paragraph("Scan Imaging", ss["SectionHeader"]))

    orig_io = io.BytesIO(original_image_bytes)
    overlay_io = _make_overlay_image(original_image_bytes, segmentation)

    img_row = Table(
        [[_img_flowable(orig_io), _img_flowable(overlay_io)]],
        colWidths=[86 * mm, 86 * mm],
    )
    img_row.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    story.append(img_row)

    caption_row = Table(
        [[Paragraph("Original MRI", ss["ImageCaption"]), Paragraph("Segmentation Overlay", ss["ImageCaption"])]],
        colWidths=[86 * mm, 86 * mm],
    )
    story.append(caption_row)
    story.append(Spacer(1, 10))

    # ---------------- Classification table ----------------
    story.append(Paragraph("Classification Results (CNN)", ss["SectionHeader"]))
    cls_data = [["Tumor Type", "Confidence"]]
    for c in classification:
        cls_data.append([c["name"].capitalize(), f"{c['prob']*100:.1f}%"])

    cls_table = Table(cls_data, colWidths=[100 * mm, 72 * mm])
    cls_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, CLOUD]),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#EAF1FC")),  # highlight top prediction
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 1), (-1, 1), BLUE),
        ("GRID", (0, 0), (-1, -1), 0.5, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(cls_table)
    story.append(Spacer(1, 10))

    # ---------------- Segmentation / measurement table ----------------
    story.append(Paragraph("Tumor Measurements (U-Net Segmentation)", ss["SectionHeader"]))
    cell_style = ParagraphStyle("TableCell", fontName="Helvetica", fontSize=9.5, textColor=SLATE, leading=12)
    header_style = ParagraphStyle("TableHeader", fontName="Helvetica-Bold", fontSize=9.5, textColor=colors.white, leading=12)

    if segmentation.get("present"):
        seg_data = [
            [Paragraph("Metric", header_style), Paragraph("Value", header_style)],
            [Paragraph("Region", cell_style), Paragraph(segmentation.get("region", "N/A"), cell_style)],
            [Paragraph("Estimated Area", cell_style), Paragraph(f"{segmentation.get('area_mm2', 'N/A')} mm<super>2</super>", cell_style)],
            [Paragraph("Maximum Diameter", cell_style), Paragraph(f"{segmentation.get('diam_mm', 'N/A')} mm", cell_style)],
            [Paragraph("Segmentation Confidence", cell_style), Paragraph(f"{segmentation.get('mask_confidence', 0)*100:.1f}%", cell_style)],
        ]
    else:
        seg_data = [[Paragraph("Metric", header_style), Paragraph("Value", header_style)],
                    [Paragraph("Result", cell_style), Paragraph("No tumor region detected above threshold", cell_style)]]

    seg_table = Table(seg_data, colWidths=[100 * mm, 72 * mm])
    seg_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, CLOUD]),
        ("GRID", (0, 0), (-1, -1), 0.5, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(seg_table)
    story.append(Spacer(1, 10))

    # ---------------- Risk badge ----------------
    story.append(Paragraph("Risk Indicator", ss["SectionHeader"]))
    band = risk.get("band", "low")
    risk_color = RISK_COLORS.get(band, RISK_COLORS["low"])
    risk_table = Table(
        [[Paragraph(f"<b>{risk.get('label', 'N/A')}</b>", ParagraphStyle(
            "RiskLabel", fontName="Helvetica-Bold", fontSize=12, textColor=colors.white)),
          Paragraph(risk.get("explanation", ""), ParagraphStyle(
              "RiskExpl", fontName="Helvetica", fontSize=9, textColor=colors.white, leading=12))]],
        colWidths=[35 * mm, 137 * mm],
    )
    risk_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), risk_color),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (0, 0), 10),
        ("LEFTPADDING", (1, 0), (1, 0), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(risk_table)
    story.append(Spacer(1, 10))

    # ---------------- AI Explanation ----------------
    if explanation_text:
        story.append(
            Paragraph(
                "AI-Generated Explanation (Mistral)",
                ss["SectionHeader"]
            )
        )

        formatted_explanation = _markdown_to_reportlab(explanation_text)

        explanation_box = Table(
            [[
                Paragraph(
                    formatted_explanation,
                    ss["BodyText2"]
                )
            ]],
            colWidths=[172 * mm],
        )

        explanation_box.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), CLOUD),
            ("BOX", (0, 0), (-1, -1), 0.5, LINE),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))

    story.append(explanation_box)
    story.append(Spacer(1, 10))

    # ---------------- Disclaimer ----------------
    story.append(HRFlowable(width="100%", thickness=0.75, color=LINE))
    story.append(Spacer(1, 6))
    disclaimer = (
        "This report was generated by an automated research prototype (CNN classifier + U-Net "
        "segmentation, pseudo-mask trained) and an AI language model (Mistral). Segmentation area, "
        "diameter, and location are approximate and derived from Grad-CAM-based pseudo-masks, not "
        "radiologist-verified ground truth. Millimeter measurements assume an uncalibrated pixel-to-mm "
        "ratio and are illustrative only. This report is NOT a medical diagnosis and must not be used "
        "as the sole basis for clinical decisions. Please consult a qualified radiologist or physician "
        "for interpretation and next steps."
    )
    story.append(Paragraph(disclaimer, ss["Disclaimer"]))

    doc.build(story)
    buf.seek(0)
    return buf.read()
