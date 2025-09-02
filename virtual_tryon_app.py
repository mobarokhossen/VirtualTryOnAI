"""
Virtual Try‑On Streamlit App

Features
- Upload a person image
- Choose region: Upper or Lower
- Enter prompt or pick from preset clothing descriptions
- Detect clothing region via Gemini 1.5 Pro (bounding boxes constrained to allowed labels)
- Build a mask from detected boxes (with adjustable dilation and mask mode)
- Inpaint/insert new clothing using Vertex AI Imagen 3 (edit_image)
- Show final image and mask; allow downloads

──────────────
Quick Start
1) Python 3.10+ recommended
2) pip install -r requirements (suggested):
   - streamlit
   - pillow
   - numpy
   - google-generativeai
   - vertexai

   Example:
     pip install streamlit pillow numpy google-generativeai vertexai

3) Auth & Keys
   - Set environment variables (or Streamlit sidebar fields):
       GEMINI_API_KEY=<your_gemini_api_key>
       GCP_PROJECT_ID=<your_gcp_project_id>
       GCP_LOCATION=us-central1   # or your region
   - You must have Vertex AI API enabled in your GCP project and access to Imagen 3 Edit (imagen-3.0-capability-001).
   - Authenticate locally for Vertex AI (pick ONE):
       a) gcloud auth application-default login
          gcloud config set project <your_project_id>
       b) Or set GOOGLE_APPLICATION_CREDENTIALS to a service account JSON with Vertex permissions.

4) Run:
     streamlit run app.py

Notes
- If Gemini detection returns no boxes for the chosen region, try another image or adjust the region.
- The mask is built from detected rectangles for allowed labels. Dilation expands/shrinks the mask.
- This app uses rectangle masks; for higher fidelity, swap in a parsing/segmentation model.

"""

import io
import os
import re
import json
import tempfile
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image as PILImage, ImageDraw

import streamlit as st

# Google Gemini
import google.generativeai as genai

# Vertex AI Imagen 3 Edit
import vertexai
from vertexai.preview.vision_models import (
    Image as VtxImage,
    ImageGenerationModel,
    RawReferenceImage,
    MaskReferenceImage,
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
UPPER_LABELS = [
    'shirt', 'jacket', 'top', 't-shirt', 'dress', 'coat', 'blouse', 'sweater',
]
LOWER_LABELS = [
    'pants', 'skirt', 'shorts', 'jeans', 'trousers',
]

# Default presets (assignment-provided)
UPPER_PRESETS = [
    "A casual white cotton t-shirt",
    "A formal black blazer jacket",
    "A colorful tropical Hawaiian shirt",
    "A cozy knitted wool sweater",
    "A classic denim jacket",
    "An elegant silk blouse with floral patterns",
    "A striped navy polo shirt",
    "A luxurious leather jacket",
    "A professional button-down shirt",
    "A trendy crop top with polka dots",
]
LOWER_PRESETS = [
    "Classic blue denim jeans",
    "Elegant black dress pants",
    "Casual khaki chinos",
    "Trendy ripped skinny jeans",
    "Professional gray slacks",
    "Comfortable athletic shorts",
    "Stylish pleated skirt",
    "Casual cargo pants",
    "Formal suit trousers",
    "Vintage wide-leg pants",
]

# ──────────────────────────────────────────────────────────────────────────────
# Helper types
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Box:
    ymin: float
    xmin: float
    ymax: float
    xmax: float
    label: str

    def to_abs(self, w: int, h: int) -> Tuple[int, int, int, int]:
        """Convert normalized [0..1] box to absolute pixel coords (top, left, bottom, right)."""
        top = max(0, min(h, int(round(self.ymin * h))))
        left = max(0, min(w, int(round(self.xmin * w))))
        bottom = max(0, min(h, int(round(self.ymax * h))))
        right = max(0, min(w, int(round(self.xmax * w))))
        return top, left, bottom, right

# ──────────────────────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────────────────────

def get_env_or_default(name: str, default: str = "") -> str:
    val = os.environ.get(name)
    return val if val else default


def ensure_rgb(img: PILImage.Image) -> PILImage.Image:
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def parse_boxes_from_text(text: str) -> List[Box]:
    """
    Parse Gemini response expected to be either JSON or a Python-like list of lists.
    Target format: [[ymin, xmin, ymax, xmax, label], ...]
    Values are normalized floats within [0, 1]; label is a string.
    """
    if not text:
        return []

    # Try to extract a JSON array from the text
    cleaned = text.strip()

    # Some models wrap in markdown code fences; strip them
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    # Attempt direct JSON
    try:
        data = json.loads(cleaned)
        return _coerce_boxes(data)
    except Exception:
        pass

    # Attempt to locate first array in text
    match = re.search(r"\[[\s\S]*\]", cleaned)
    if match:
        snippet = match.group(0)
        try:
            data = json.loads(snippet)
            return _coerce_boxes(data)
        except Exception:
            # Last resort: safe eval style replacement of quotes
            try:
                snippet_jsonish = snippet.replace("'", '"')
                data = json.loads(snippet_jsonish)
                return _coerce_boxes(data)
            except Exception:
                return []
    return []


def _coerce_boxes(data) -> List[Box]:
    boxes: List[Box] = []
    if not isinstance(data, list):
        return boxes
    for item in data:
        if not isinstance(item, list) or len(item) < 5:
            continue
        try:
            ymin, xmin, ymax, xmax, label = item[:5]
            boxes.append(Box(float(ymin), float(xmin), float(ymax), float(xmax), str(label)))
        except Exception:
            continue
    return boxes


def filter_boxes_by_region(boxes: List[Box], region: str) -> List[Box]:
    allowed = UPPER_LABELS if region == 'Upper' else LOWER_LABELS
    allowed_set = set(a.lower() for a in allowed)
    out = [b for b in boxes if b.label.lower() in allowed_set]
    # If none match allowed labels, gracefully fall back to heuristics:
    if not out:
        # Heuristic: choose the largest box if any exist
        out = boxes
    return out


def boxes_union_mask(size: Tuple[int, int], boxes: List[Box], dilation: float) -> PILImage.Image:
    """Create a binary mask (white=region to edit) from boxes with fractional dilation."""
    w, h = size
    mask = PILImage.new("L", (w, h), color=0)
    draw = ImageDraw.Draw(mask)

    for b in boxes:
        top, left, bottom, right = b.to_abs(w, h)
        # Dilate bounding box by a fraction of image size
        pad_w = int(round(dilation * w))
        pad_h = int(round(dilation * h))
        l = max(0, left - pad_w)
        t = max(0, top - pad_h)
        r = min(w, right + pad_w)
        btm = min(h, bottom + pad_h)
        draw.rectangle([l, t, r, btm], fill=255)

    return mask


def draw_boxes_overlay(img: PILImage.Image, boxes: List[Box]) -> PILImage.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    for b in boxes:
        t, l, btm, r = b.to_abs(w, h)
        draw.rectangle([l, t, r, btm], outline=(255, 0, 0), width=3)
        draw.text((l + 4, t + 4), b.label, fill=(255, 0, 0))
    return out


def save_pil(temp_suffix: str, img: PILImage.Image) -> str:
    path = tempfile.NamedTemporaryFile(delete=False, suffix=temp_suffix).name
    img.save(path)
    return path


def gemini_detect(img_pil: PILImage.Image, api_key: str, allowed_labels: List[str]) -> List[Box]:
    """Call Gemini to detect bounding boxes for allowed labels. Returns list of Box."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name='gemini-1.5-pro',
        generation_config={
            # Ask for JSON to improve parseability
            'response_mime_type': 'application/json'
        },
    )

    instruction = (
        "You are a vision model. Detect 2D bounding boxes for clothing in the image.\n"
        f"Only consider the following labels: {allowed_labels}.\n"
        "Return a JSON array of items; each item must be:\n"
        "  [ymin, xmin, ymax, xmax, label]\n"
        "Coordinates are normalized floats in [0,1] relative to image height (y) and width (x).\n"
        "If multiple items match, include multiple entries. Do not add any explanatory text."
    )

    # Gemini Python SDK supports PIL images directly
    try:
        resp = model.generate_content([
            img_pil,
            instruction,
        ])
        text = resp.text
    except Exception as e:
        raise RuntimeError(f"Gemini detection failed: {e}")

    return parse_boxes_from_text(text)


def vertex_inpaint(
    base_img: PILImage.Image,
    mask_img: PILImage.Image,
    prompt: str,
    project_id: str,
    location: str,
    edit_mode: str = 'inpainting-insert',
    mask_mode: str = 'foreground',
) -> PILImage.Image:
    """Call Vertex Imagen 3 Edit to inpaint/insert within mask."""
    # Init Vertex
    try:
        vertexai.init(project=project_id, location=location)
    except Exception as e:
        raise RuntimeError(f"Vertex initialization failed: {e}")

    # Save to temp and load via Vertex Image API
    base_img = ensure_rgb(base_img)
    mask_img = mask_img.convert('L')

    base_path = save_pil('.png', base_img)
    mask_path = save_pil('.png', mask_img)

    try:
        v_base = VtxImage.load_from_file(location=base_path)
        v_mask = VtxImage.load_from_file(location=mask_path)

        raw_ref = RawReferenceImage(image=v_base, reference_id=0)
        mask_ref = MaskReferenceImage(
            reference_id=1,
            image=v_mask,
            mask_mode=mask_mode,
            dilation=0.0,  # we already handled dilation in mask creation
        )

        model = ImageGenerationModel.from_pretrained("imagen-3.0-capability-001")
        images = model.edit_image(
            prompt=prompt,
            edit_mode=edit_mode,
            reference_images=[raw_ref, mask_ref],
            number_of_images=1,
            safety_filter_level="block_some",
            person_generation="allow_adult",
        )

        # Try returning PIL
        # The SDK returns a list of Vertex Images; convert to bytes then PIL
        vimg = images[0]
        # Newer SDKs: vimg._image_bytes or vimg.to_bytes()
        img_bytes = None
        for attr in ["_image_bytes", "bytes", "to_bytes", "as_bytes"]:
            if hasattr(vimg, attr):
                val = getattr(vimg, attr)
                img_bytes = val() if callable(val) else val
                break
        if img_bytes is None:
            # Fallback: save to temp via SDK if available
            out_path = save_pil('.png', ensure_rgb(base_img))
            try:
                vimg.save(location=out_path)  # some SDKs support .save
                return PILImage.open(out_path).convert('RGB')
            except Exception as e:
                raise RuntimeError(f"Could not extract edited image: {e}")

        pil = PILImage.open(io.BytesIO(img_bytes)).convert('RGB')
        return pil

    finally:
        # Clean temp files
        for p in [locals().get('base_path'), locals().get('mask_path')]:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Virtual Try‑On (Gemini + Imagen 3)", page_icon="🧥", layout="wide")
st.title("🧥 Virtual Try‑On — Gemini + Vertex AI Imagen 3")
st.caption("Upload a photo, choose a region, describe the clothing, and generate a try‑on.")

# Sidebar: Keys / Config
with st.sidebar:
    st.header("🔑 Configuration")
    gemini_key = st.text_input("Gemini API Key", value=get_env_or_default("GEMINI_API_KEY"), type="password")
    gcp_project = st.text_input("GCP Project ID", value=get_env_or_default("GCP_PROJECT_ID"))
    gcp_location = st.text_input("GCP Location", value=get_env_or_default("GCP_LOCATION") or "us-central1")

    st.divider()
    st.subheader("🎛️ Edit Settings")
    region = st.radio("Region to edit", options=["Upper", "Lower"], horizontal=True)
    mask_mode = st.selectbox("Mask mode", options=["foreground", "background"], index=0, help="'foreground' edits inside the white mask; 'background' edits outside.")
    edit_mode = st.selectbox("Edit mode", options=["inpainting-insert", "outpainting", "inpainting-remove"], index=0)
    dilation_frac = st.slider("Mask dilation (fraction of image size)", min_value=0.0, max_value=0.10, value=0.02, step=0.005)

# Main controls
col_left, col_right = st.columns([1, 1])

with col_left:
    uploaded = st.file_uploader("Upload your image (person photo)", type=["png", "jpg", "jpeg"])

    preset_list = UPPER_PRESETS if region == 'Upper' else LOWER_PRESETS
    prompt_choice = st.selectbox("Pick a preset (optional)", options=["(custom)"] + preset_list, index=0)
    prompt_text = st.text_input("Or type your clothing description", value="A dark green hoodie, white shirt inside, waist length")
    clothing_prompt = prompt_text if prompt_choice == "(custom)" else prompt_choice

    run = st.button("Generate Try‑On", type="primary")

with col_right:
    st.write("\n")

# Process pipeline on click
if run:
    if not uploaded:
        st.error("Please upload an image first.")
        st.stop()
    if not gemini_key:
        st.error("Gemini API Key is required for detection.")
        st.stop()
    if not gcp_project:
        st.error("GCP Project ID is required for Imagen editing.")
        st.stop()

    # Load image
    img = ensure_rgb(PILImage.open(uploaded))
    st.subheader("1) Uploaded Image")
    st.image(img, use_container_width=True)

    # 1) Detect boxes
    st.subheader("2) Detect Clothing Region (Gemini)")
    allowed = UPPER_LABELS if region == 'Upper' else LOWER_LABELS
    try:
        boxes = gemini_detect(img, gemini_key, allowed)
    except Exception as e:
        st.error(str(e))
        st.stop()

    if not boxes:
        st.warning("No bounding boxes detected. Try a different image or region.")
        st.stop()

    filtered = filter_boxes_by_region(boxes, region)
    overlay = draw_boxes_overlay(img, filtered)
    st.image(overlay, caption="Detected boxes (filtered)", use_container_width=True)

    # 2) Build mask
    st.subheader("3) Mask for Inpainting")
    mask = boxes_union_mask(img.size, filtered, dilation=dilation_frac)
    st.image(mask, caption="Mask (white = edit region)", use_container_width=True)

    # 3) Edit via Vertex Imagen 3
    st.subheader("4) Virtual Try‑On Result (Imagen 3 Edit)")
    try:
        out = vertex_inpaint(
            base_img=img,
            mask_img=mask,
            prompt=clothing_prompt,
            project_id=gcp_project,
            location=gcp_location,
            edit_mode=edit_mode,
            mask_mode=mask_mode,
        )
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.image(out, caption="Final Try‑On Image", use_container_width=True)

    # Downloads
    out_buf = io.BytesIO()
    out.save(out_buf, format='PNG')
    st.download_button("Download Final Image (PNG)", data=out_buf.getvalue(), file_name="tryon_output.png", mime="image/png")

    mask_buf = io.BytesIO()
    mask.save(mask_buf, format='PNG')
    st.download_button("Download Mask (PNG)", data=mask_buf.getvalue(), file_name="tryon_mask.png", mime="image/png")

    st.success("Done! You can tweak prompts, mask mode, or dilation and run again.")
