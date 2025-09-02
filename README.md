# 🧥 Virtual Try‑On (Streamlit + Gemini 1.5 Pro + Vertex AI Imagen 3)

Turn a user’s photo into an interactive **virtual try‑on** experience: detect the clothing region with **Gemini 1.5 Pro**, build a mask, and inpaint new garments with **Vertex AI Imagen 3** — all wrapped in a simple **Streamlit** app.

---

## ✨ Features

* **Image upload** of a person photo
* **Region selection**: Upper (shirt/jacket/top) or Lower (pants/skirt)
* **Prompt presets** (assignment-provided) + custom prompt input
* **Bounding box detection** via Gemini 1.5 Pro, constrained to allowed clothing labels
* **Mask generation** from detected boxes with adjustable **dilation** and **mask mode** (foreground/background)
* **Virtual try‑on** using Vertex **Imagen 3** `edit_image` (inpainting‑insert / outpainting / inpainting‑remove)
* Shows both **final try‑on image** and the **mask**, with **download buttons**

---

## 🖼️ Demo

> Add a GIF or screenshot here (e.g., `assets/demo.gif`).

```
assets/
 └── demo.gif
```

---

## 🔧 Architecture

1. **User uploads** a portrait photo.
2. **Gemini 1.5 Pro** receives (image + instruction) and returns normalized boxes:
   `[[ymin, xmin, ymax, xmax, label], ...]`
3. App **filters** boxes by region (Upper/Lower) using allowed labels.
4. App **builds a binary mask** from selected boxes; optional **dilation** to expand the edit area.
5. **Vertex AI Imagen 3** performs **edit\_image** with the base image + mask + clothing prompt.
6. Streamlit displays **overlay, mask, and final result**, and provides download buttons.

---

## 📦 Requirements

* Python **3.10+**
* Packages (minimal):

  * `streamlit`
  * `pillow`
  * `numpy`
  * `google-generativeai`
  * `google-cloud-aiplatform` (Vertex AI Python SDK)

Create a `requirements.txt`:

```txt
streamlit
pillow
numpy
google-generativeai
google-cloud-aiplatform
```

Install:

```bash
pip install -r requirements.txt
```

> **Note:** The app uses classes like `RawReferenceImage`/`MaskReferenceImage` exposed via **`vertexai.preview.vision_models`** from the `google-cloud-aiplatform` package. Use a recent version (e.g., **1.95.x** or newer).

---

## 🔑 Credentials & Config

You need access to **both** Gemini and Vertex AI.

### Gemini API Key

* Create a key at **Google AI Studio** → copy value.
* Environment variable: `GEMINI_API_KEY`

### Vertex AI / GCP

* A **GCP project** with **Vertex AI API enabled** and access to **Imagen 3** (`imagen-3.0-capability-001`).
* Environment variables: `GCP_PROJECT_ID`, `GCP_LOCATION` (e.g., `us-central1`).
* Local auth (choose one):

  * **User ADC**: `gcloud auth application-default login` then `gcloud config set project <project>`
  * **Service Account**: set `GOOGLE_APPLICATION_CREDENTIALS=/path/key.json` with Vertex permissions

### Optional `.env`

```env
GEMINI_API_KEY=your_gemini_key
GCP_PROJECT_ID=your_project_id
GCP_LOCATION=us-central1
```

> You can also enter these directly in the Streamlit **sidebar** at runtime.

---

## ▶️ Run Locally

```bash
streamlit run app.py
```

Streamlit starts on `http://localhost:8501` by default.

---

## 🕹️ Usage

1. **Upload** a clear person image (torso visible for Upper, legs visible for Lower).
2. Pick **Upper** or **Lower** region.
3. Choose a **preset** or type a **custom clothing prompt**.
4. Adjust **mask mode** (foreground/background) and **dilation** if needed.
5. Click **Generate Try‑On**.
6. Review **boxes overlay**, **mask**, and **final image**. Use **Download** buttons as needed.

---

## 🎚️ Allowed Labels & Presets

* **Upper labels**: `['shirt', 'jacket', 'top', 't-shirt', 'dress', 'coat', 'blouse', 'sweater']`
* **Lower labels**: `['pants', 'skirt', 'shorts', 'jeans', 'trousers']`

**Upper presets** (examples)

* A casual white cotton t‑shirt
* A formal black blazer jacket
* A colorful tropical Hawaiian shirt
* A cozy knitted wool sweater
* A classic denim jacket
* An elegant silk blouse with floral patterns
* A striped navy polo shirt
* A luxurious leather jacket
* A professional button‑down shirt
* A trendy crop top with polka dots

**Lower presets** (examples)

* Classic blue denim jeans
* Elegant black dress pants
* Casual khaki chinos
* Trendy ripped skinny jeans
* Professional gray slacks
* Comfortable athletic shorts
* Stylish pleated skirt
* Casual cargo pants
* Formal suit trousers
* Vintage wide‑leg pants

---

## 📁 Project Structure

```
virtual-tryon/
├─ app.py                 # Streamlit application
├─ requirements.txt       # Python deps
├─ assets/                # (optional) demo GIFs, screenshots
└─ README.md              # this file
```

---

## 🩹 Troubleshooting

**ImportError: cannot import name `RawReferenceImage`**

* Ensure you’re importing from the correct module:

  ```python
  from vertexai.preview.vision_models import (
      Image as VtxImage,
      ImageGenerationModel,
      RawReferenceImage,
      MaskReferenceImage,
  )
  ```
* Upgrade the SDK (inside your venv):

  ```bash
  pip uninstall -y vertexai || true
  pip install -U google-cloud-aiplatform
  # e.g., pip install -U "google-cloud-aiplatform==1.95.1"
  ```
* Verify in Python:

  ```bash
  python -c "import vertexai, vertexai.preview.vision_models as v; print(vertexai.__version__); print([c for c in dir(v) if 'ReferenceImage' in c])"
  ```

**PermissionDenied / Quota issues**

* Check Vertex AI **API enabled**, project set, and credentials have the right roles.
* Some Imagen 3 capabilities require allow‑listing; confirm availability in your region.

**Gemini returns no boxes**

* Try a clearer image, different region, or adjust the prompt. The app falls back to largest box if no label matches.

**App runs but Imagen output looks off**

* Increase mask **dilation**, switch **mask mode**, or refine your clothing prompt.

---

## ⚠️ Limitations

* Box‑based masks are approximate; fine edges (e.g., sleeves) may need a parsing model for best results.
* Generation quality depends on input image clarity, pose, occlusions, and prompt specificity.

---

## 🗺️ Roadmap Ideas

* Swap rectangle masks for **human parsing** segmentation
* Add **pose‑aware** garment warping / physics
* Batch processing and prompt history
* One‑click deploy to **Streamlit Community Cloud**

---

## 🤝 Contributing

PRs welcome! Please open an issue first to discuss major changes.

---

## 📄 License

MIT — see `LICENSE` for details.

---

## 🙏 Acknowledgments

* Google **Gemini 1.5 Pro** for multi‑modal perception
* Google **Vertex AI Imagen 3** for image editing
* **Streamlit** for rapid prototyping
