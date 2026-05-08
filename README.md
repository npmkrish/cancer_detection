# 🧬 Cancer Detection System — AI-Powered Medical Imaging Platform

> An end-to-end deep learning web application for automated cancer detection in medical images, featuring YOLO-based object detection, geolocation tagging, PDF report generation, and a Flask-powered REST API backend.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Core Modules](#core-modules)
- [Model Details](#model-details)
- [Known Issues & Improvements](#known-issues--improvements)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This platform allows medical professionals and researchers to upload diagnostic images, run AI-based cancer detection inference, visualize results on a map, and generate structured PDF reports — all through a web interface. The backend uses a custom-trained YOLOv8 model (`best2.pt`) for tumor/lesion detection, combined with YOLOv5s as a secondary general-purpose detector.

---

## ✨ Features

- 🔍 **Cancer Detection via YOLO** — Custom-trained YOLOv8 model for identifying cancerous regions
- 🗺️ **Geolocation Tagging** — Embeds GPS metadata into prediction images using EXIF/piexif
- 📄 **PDF Report Generation** — Automated object count reports with annotated image previews
- 🧹 **Detection Box Removal** — OpenCV-based utility to clean prediction overlays from images
- 👤 **User Authentication** — MongoDB-backed login/register with session management
- 🖼️ **Image Gallery** — Browse uploaded and predicted image collections
- 💬 **Chat Interface** — NLP-based report interaction page
- 🔗 **REST API** — JSON endpoints for prediction, metadata extraction, and report generation

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| ML Inference | YOLOv8 (Ultralytics), YOLOv5 (PyTorch Hub) |
| Image Processing | OpenCV, Pillow, piexif |
| Database | MongoDB Atlas (pymongo) |
| Report Generation | FPDF |
| Frontend | Jinja2 Templates, HTML/CSS/JS |
| Deployment | Local (debug mode) |

---

## 📁 Project Structure

```
cancer-detection-system/
│
├── app.py                    # Main Flask application
├── best2.pt                  # Custom-trained YOLOv8 model weights
│
├── static/
│   ├── uploads/              # User-uploaded images
│   ├── predictions/          # YOLO annotated output images
│   ├── reports/              # Generated PDF reports
│   ├── files/                # Downloadable files (e.g., FUSIONAI.pdf)
│   └── fonts/                # Custom fonts for PDF generation
│
├── templates/
│   ├── landing.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── report.html
│   ├── map.html
│   ├── chat.html
│   ├── video.html
│   ├── route.html
│   ├── uploaded_images.html
│   ├── predicted_images.html
│   └── 405.html
│
└── utils/
    └── remove_detection_boxes.py   # OpenCV box removal utility
```

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.9+
pip
MongoDB Atlas account (or local MongoDB)
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/cancer-detection-system.git
cd cancer-detection-system

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install flask pymongo ultralytics torch torchvision opencv-python \
            pillow piexif fpdf2 numpy

# 4. Place your trained model weights
# Copy best2.pt to the project root (or update the path in app.py)

# 5. Configure MongoDB connection
# In app.py, update the MongoClient URI with your credentials

# 6. Run the application
python app.py
```

The server starts at `http://127.0.0.1:5000` by default.

---

## 📡 API Reference

### `POST /predict`

Run YOLO inference on an uploaded image with geolocation metadata.

**Request (multipart/form-data):**

| Field | Type | Description |
|---|---|---|
| `file` | File | Medical image (JPG/PNG) |
| `latitude` | float | GPS latitude |
| `longitude` | float | GPS longitude |

**Response:**
```json
{
  "image_url": "/static/predictions/annotated_image.jpg?t=abc123",
  "latitude": 28.6139,
  "longitude": 77.2090
}
```

---

### `POST /generate_report`

Generate a PDF report from uploaded image with YOLOv5 detection summary.

**Request (multipart/form-data):**

| Field | Type | Description |
|---|---|---|
| `image` | File | Image file for analysis |

**Response:** PDF file download

---

### `POST /upload_metadata_image`

Extract GPS geolocation from an image's EXIF metadata.

**Request (multipart/form-data):**

| Field | Type | Description |
|---|---|---|
| `file` | File | Image with EXIF GPS data |

**Response:**
```json
{
  "latitude": 51.5074,
  "longitude": -0.1278
}
```

---

### `POST /report_nlp`

Generate a structured NLP-style PDF analysis report from object count data.

**Request (JSON):**
```json
{
  "object_counts": {
    "tumor": 3,
    "lesion": 1
  }
}
```

**Response:** PDF file download

---

### `POST /upload`

Upload an image, run YOLOv8 detection, and return an HTML report.

**Request (multipart/form-data):**

| Field | Type | Description |
|---|---|---|
| `file` | File | Image file |

**Response:** Renders `report.html` with detected object counts

---

## 🔧 Core Modules

### Detection Box Removal Utility

This utility uses OpenCV's inpainting algorithm to cleanly remove bounding boxes and labels from YOLO prediction output images — useful when you want to present clean imagery in reports.

```python
import cv2
import numpy as np

def remove_detection_boxes(image_path, output_path):
    """
    Remove detection boxes and labels from predicted images using inpainting.

    Args:
        image_path (str): Path to input image with detection boxes
        output_path (str): Path to save cleaned image
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the image")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Build color masks for common YOLO box colors
    masks = []
    color_ranges = [
        (np.array([180, 180, 180]), np.array([255, 255, 255])),  # White
        (np.array([0, 0, 100]),     np.array([100, 100, 255])),  # Blue
        (np.array([100, 0, 0]),     np.array([255, 100, 100])),  # Red
        (np.array([100, 100, 0]),   np.array([255, 255, 100])),  # Cyan
    ]
    for lower, upper in color_ranges:
        masks.append(cv2.inRange(image_rgb, lower, upper))

    combined_mask = masks[0]
    for m in masks[1:]:
        combined_mask = cv2.bitwise_or(combined_mask, m)

    # Dilate to cover edges and text
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(combined_mask, kernel, iterations=2)

    # Inpaint using Telea algorithm
    cleaned = cv2.inpaint(image, dilated_mask, 3, cv2.INPAINT_TELEA)
    cv2.imwrite(output_path, cleaned)
    return cleaned


if __name__ == "__main__":
    cleaned = remove_detection_boxes("predicted_image.jpg", "cleaned_image.jpg")
    print("Cleaned image saved.")
```

---

### Geolocation EXIF Embedding

```python
import piexif

def convert_to_dms(decimal):
    """Convert decimal GPS coordinate to DMS tuple for EXIF."""
    degrees = int(decimal)
    minutes = int((decimal - degrees) * 60)
    seconds = int(((decimal - degrees) * 60 - minutes) * 60 * 100)
    return [(degrees, 1), (minutes, 1), (seconds, 100)]

def add_geolocation(image_path, latitude, longitude):
    """Embed GPS coordinates into a JPEG image's EXIF metadata."""
    from PIL import Image

    image = Image.open(image_path)
    exif_data = image.info.get("exif", None)
    exif_dict = piexif.load(exif_data) if exif_data else {
        "0th": {}, "Exif": {}, "GPS": {}, "Interop": {}, "1st": {}, "thumbnail": None
    }

    exif_dict["GPS"] = {
        piexif.GPSIFD.GPSLatitudeRef:  b'N' if latitude >= 0 else b'S',
        piexif.GPSIFD.GPSLatitude:     convert_to_dms(abs(latitude)),
        piexif.GPSIFD.GPSLongitudeRef: b'E' if longitude >= 0 else b'W',
        piexif.GPSIFD.GPSLongitude:    convert_to_dms(abs(longitude)),
    }

    image = image.convert("RGB")
    image.save(image_path, "JPEG", exif=piexif.dump(exif_dict))
```

---

### PDF Report Generation

```python
from fpdf import FPDF

def generate_pdf_report(report_path, object_counts, image_path):
    """Generate a detection summary PDF with embedded image."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Object Detection Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt="Uploaded Image:", ln=True)
    pdf.image(image_path, x=10, y=30, w=190)
    pdf.ln(80)
    pdf.cell(0, 10, txt="Detected Objects:", ln=True)
    for label, count in object_counts.items():
        pdf.cell(0, 10, txt=f"  - {label}: {count}", ln=True)
    pdf.output(report_path)
```

---

## 🤖 Model Details

| Model | Purpose | Source |
|---|---|---|
| `best2.pt` | Primary cancer/lesion detection | Custom-trained YOLOv8 |
| `yolov5s` | Secondary general object detection | PyTorch Hub |

The primary model was trained on a domain-specific medical imaging dataset. Inference runs at `conf=0.25` confidence threshold. Images are resized to a max dimension of 1024px before YOLOv5 inference to prevent memory issues with large scans.

---

## ⚠️ Known Issues & Improvements

**Security (Critical — fix before any deployment):**

- Passwords are stored in plaintext in MongoDB. Use `bcrypt` or `argon2` for hashing:
  ```python
  from bcrypt import hashpw, gensalt, checkpw
  hashed = hashpw(password.encode(), gensalt())
  ```
- MongoDB credentials and `secret_key` are hardcoded. Move to environment variables:
  ```python
  import os
  app.secret_key = os.environ.get("FLASK_SECRET_KEY")
  client = MongoClient(os.environ.get("MONGO_URI"))
  ```
- The `add_geolocation()` call at module level with a hardcoded path will crash on startup if the file doesn't exist. Move it inside a conditional block or remove it.

**Code Quality:**
- `@app.errorhandler(405)` is defined twice — remove the duplicate.
- `method_not_allowed` error handler uses `redirect(url_for('405.html', ...))` which is invalid; `url_for` takes view function names, not template names.
- Font path in `generate_report_nlp` is hardcoded to a specific local machine path — use `os.path.join(app.root_path, ...)` instead.

**Feature Improvements:**
- Add file type validation on upload (restrict to `.jpg`, `.jpeg`, `.png`)
- Add confidence threshold as a user-configurable parameter in `/predict`
- Implement async inference for large images to prevent request timeouts
- Add a results database to persist past predictions per user

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m 'Add your feature'`
4. Push to branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

----------------------------------------------------------------------

> Built as part of SIH (Smart India Hackathon) — FusionAI team.
