---
title: AI Facial Expression Detection Bot
emoji: 🤖
colorFrom: green
colorTo: blue
sdk: gradio
app_file: web_app.py
pinned: false
---

<div align="center">

# AI Facial Expression Detection Bot

**Detect a face. Estimate the expression. Get a short supportive reply.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![DeepFace](https://img.shields.io/badge/AI-DeepFace-0EA5E9?style=for-the-badge)](https://github.com/serengil/deepface)
[![OpenCV](https://img.shields.io/badge/Vision-OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-F97316?style=for-the-badge)](https://www.gradio.app/)
[![TensorFlow](https://img.shields.io/badge/Backend-TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)

[Features](#-features) ·
[Quick Start](#-quick-start) ·
[Usage](#-usage) ·
[How It Works](#-how-it-works) ·
[Project Structure](#-project-structure) ·
[Troubleshooting](#-troubleshooting)

</div>

---

## About

This project uses computer vision to estimate the **dominant facial expression** from a webcam photo or uploaded image.

It ships in two modes:

| Mode | File | Best for |
| --- | --- | --- |
| **Web app** | `web_app.py` | Browser demo, Hugging Face Spaces, Render |
| **Desktop app** | `app.py` | Live local webcam window |

> This is **not** a chatbot LLM. The AI is **DeepFace** facial expression recognition.

---

## Features

- Webcam snapshot **or** image upload
- Detects **7 expressions** with a confidence score
- Draws a face bounding box on the result
- Returns a short **bot response** for the detected expression
- Web UI built with Gradio
- Desktop live mode with OpenCV
- Production-friendly warmup so models load at startup

---

## Supported Expressions

| Expression | Label | Example bot tone |
| --- | --- | --- |
| Happy | `happy` | Encouraging |
| Sad | `sad` | Supportive |
| Angry | `angry` | Calming |
| Surprise | `surprise` | Curious |
| Fear | `fear` | Reassuring |
| Disgust | `disgust` | Neutral |
| Neutral | `neutral` | Steady / focused |

---

## Tech Stack

| Layer | Tools |
| --- | --- |
| Expression AI | [DeepFace](https://github.com/serengil/deepface) |
| Face detection / vision | [OpenCV](https://opencv.org/) |
| Deep learning backend | [TensorFlow](https://www.tensorflow.org/) + `tf-keras` |
| Web interface | [Gradio](https://www.gradio.app/) |
| Arrays / images | NumPy, Pillow |

---

## Quick Start

**Requirements:** Python **3.10+**, webcam optional (needed for live capture)

```bash
git clone https://github.com/AMAS-39/EmotionDetectionBot.git
cd EmotionDetectionBot

python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
# source venv/bin/activate

pip install -r requirements.txt
```

> First startup may download DeepFace model weights automatically.

---

## Usage

### 1) Web app (recommended)

```bash
python web_app.py
```

Open the local URL from the terminal (usually `http://127.0.0.1:7860`).

1. Allow camera access **or** upload an image  
2. Click **Analyze Expression**  
3. Check expression, confidence, and bot response  

### 2) Desktop webcam app

```bash
python app.py
```

- Analyzes your face about once per second  
- Shows live emotion + bot message on screen  
- Press **Q** to quit  

---

## How It Works

```text
Image / Webcam
      │
      ▼
 Face detection (OpenCV)
      │
      ▼
 Emotion classification (DeepFace)
      │
      ▼
 Confidence + bot response
```

1. **Capture** — webcam frame or uploaded image  
2. **Detect** — find the face region  
3. **Analyze** — classify expression with DeepFace  
4. **Respond** — map the top label to a short message  

The desktop app also averages recent predictions so the label does not jump every frame.

---

## Project Structure

```text
EmotionDetectionBot/
├── web_app.py           # Gradio web app (Spaces / Render entrypoint)
├── app.py               # Desktop OpenCV live app
├── requirements.txt     # Python dependencies
├── packages.txt         # System packages for Hugging Face Spaces
├── .python-version      # Preferred Python version
├── .gitignore
└── README.md
```

---

## Deployment Notes

| Platform | Entry file | Notes |
| --- | --- | --- |
| Hugging Face Spaces | `web_app.py` | Uses Gradio SDK metadata at the top of this README |
| Render | `web_app.py` | Needs enough RAM for TensorFlow (512MB free tier is often too small) |

Useful production behavior already included in `web_app.py`:

- model warmup at startup  
- writable DeepFace cache directory  
- Gradio queue for longer inference  

---

## Troubleshooting

| Problem | What to try |
| --- | --- |
| `Analysis failed` in production | Check deploy logs for TensorFlow / memory errors |
| `No face detected` | Better lighting, face the camera, try an uploaded photo |
| First click is very slow | Normal while models download; wait for warmup to finish |
| App crashes on free hosting | Move to a plan with more RAM, or use Hugging Face Spaces |
| Webcam blocked in browser | Allow camera permission, or use **Upload** instead |

---

## Disclaimer

This project estimates **visible facial expressions** from an image or video frame.

It does **not** diagnose emotions, mental health, or intent.  
Treat results as an approximate visual estimate only.

---

## License

Educational and personal use.