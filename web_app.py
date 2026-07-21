import os
import tempfile
import traceback

# Writable cache for DeepFace model weights (critical on Render / containers)
_deepface_home = os.path.join(tempfile.gettempdir(), ".deepface")
os.makedirs(_deepface_home, exist_ok=True)
os.environ.setdefault("DEEPFACE_HOME", _deepface_home)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Reduce TensorFlow memory pressure on small production instances
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import cv2
import gradio as gr
import numpy as np
from deepface import DeepFace


EMOTION_DETAILS = {
    "happy": {
        "emoji": "😊",
        "message": "You look happy! Keep smiling and share your positive energy."
    },
    "sad": {
        "emoji": "😢",
        "message": (
            "You appear sad. Take a moment, breathe, and remember that "
            "difficult feelings can pass."
        )
    },
    "angry": {
        "emoji": "😠",
        "message": (
            "You appear upset. Try taking a slow breath and giving yourself "
            "a moment to relax."
        )
    },
    "surprise": {
        "emoji": "😮",
        "message": (
            "You look surprised! Something seems to have caught your attention."
        )
    },
    "fear": {
        "emoji": "😨",
        "message": (
            "You appear worried. Try to stay calm and focus on your breathing."
        )
    },
    "disgust": {
        "emoji": "🤢",
        "message": "You appear uncomfortable with something."
    },
    "neutral": {
        "emoji": "😐",
        "message": "You look calm, neutral, and focused."
    }
}


def _to_bgr_uint8(image):
    """Normalize Gradio / browser image input into OpenCV BGR uint8."""
    if image is None:
        return None

    # Gradio may pass a PIL image depending on version / source
    if hasattr(image, "convert"):
        image = np.array(image.convert("RGB"))

    arr = np.asarray(image)

    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")

    if np.issubdtype(arr.dtype, np.floating):
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


def _run_deepface(bgr_image):
    """Run emotion analysis with a soft fallback if face detection is strict."""
    try:
        return DeepFace.analyze(
            img_path=bgr_image,
            actions=["emotion"],
            detector_backend="opencv",
            enforce_detection=True,
            align=True,
            silent=True,
        )
    except ValueError:
        # Browser webcam photos are often lower quality; retry softly
        return DeepFace.analyze(
            img_path=bgr_image,
            actions=["emotion"],
            detector_backend="opencv",
            enforce_detection=False,
            align=True,
            silent=True,
        )


def warmup_models():
    """
    Download / load DeepFace emotion weights at startup.

    Without this, the first user click in production often times out while
    models are downloading, and the UI looks like 'AI does not work'.
    """
    print("Warming up DeepFace emotion model...")
    dummy = np.zeros((224, 224, 3), dtype=np.uint8)
    try:
        DeepFace.analyze(
            img_path=dummy,
            actions=["emotion"],
            detector_backend="opencv",
            enforce_detection=False,
            align=False,
            silent=True,
        )
        print("DeepFace emotion model is ready.")
    except Exception as error:
        # Warmup failure should not crash the whole app
        print(f"Warmup warning (first real request may be slow): {error}")


def analyze_emotion(image):
    """
    Analyze a webcam snapshot or uploaded image.

    Gradio provides the image as an RGB NumPy array (or sometimes PIL).
    DeepFace and OpenCV normally work with BGR images.
    """

    if image is None:
        return (
            None,
            "No image provided",
            0,
            "Please take a webcam photo or upload an image.",
        )

    try:
        bgr_image = _to_bgr_uint8(image)
        results = _run_deepface(bgr_image)

        if isinstance(results, list):
            if len(results) == 0:
                raise ValueError("No face was detected.")

            result = max(
                results,
                key=lambda item: (
                    item.get("region", {}).get("w", 0)
                    * item.get("region", {}).get("h", 0)
                ),
            )
        else:
            result = results

        emotion_scores = result.get("emotion", {})
        dominant_emotion = result.get("dominant_emotion", "neutral").lower()

        confidence = float(emotion_scores.get(dominant_emotion, 0.0))

        details = EMOTION_DETAILS.get(
            dominant_emotion,
            {
                "emoji": "🤖",
                "message": "The system detected a facial expression.",
            },
        )

        output_image = bgr_image.copy()
        region = result.get("region", {})

        x = max(0, int(region.get("x", 0)))
        y = max(0, int(region.get("y", 0)))
        width = max(0, int(region.get("w", 0)))
        height = max(0, int(region.get("h", 0)))

        if width > 0 and height > 0:
            cv2.rectangle(
                output_image,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                3,
            )

            label = f"{dominant_emotion.title()} {confidence:.1f}%"
            label_y = max(30, y - 12)

            cv2.putText(
                output_image,
                label,
                (x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        emotion_text = f"{details['emoji']} {dominant_emotion.title()}"

        return (
            output_rgb,
            emotion_text,
            round(confidence, 1),
            details["message"],
        )

    except ValueError:
        return (
            image,
            "No face detected",
            0,
            (
                "No clear face was detected. Face the camera directly, "
                "use good lighting, and try again."
            ),
        )

    except Exception as error:
        print(f"Analysis error: {error}")
        traceback.print_exc()

        return (
            image,
            "Analysis failed",
            0,
            (
                "The image could not be analyzed. "
                f"Details: {type(error).__name__}: {error}"
            ),
        )


CUSTOM_CSS = """
.gradio-container {
    max-width: 1150px !important;
    margin: auto !important;
}
#main-title {
    text-align: center;
    margin-bottom: 5px;
}
#subtitle {
    text-align: center;
    color: #777;
    margin-bottom: 25px;
}
#emotion-result {
    text-align: center;
    font-size: 30px;
    font-weight: bold;
}
#response-box {
    min-height: 110px;
}
footer {
    display: none !important;
}
"""


with gr.Blocks(
    title="AI Facial Expression Detection Bot",
    css=CUSTOM_CSS,
) as demo:

    gr.Markdown(
        """
        # 🤖 AI Facial Expression Detection Bot
        """,
        elem_id="main-title",
    )

    gr.Markdown(
        """
        Take a webcam photo or upload an image. The AI will detect the
        dominant visible facial expression and provide a responsive message.
        """,
        elem_id="subtitle",
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Webcam or Image",
                sources=["webcam", "upload"],
                type="numpy",
                height=430,
            )

            analyze_button = gr.Button(
                "Analyze Expression",
                variant="primary",
                size="lg",
            )

            gr.ClearButton(
                value="Clear",
                components=[input_image],
            )

        with gr.Column(scale=1):
            output_image = gr.Image(
                label="Detection Result",
                height=430,
                interactive=False,
            )

    with gr.Row():
        emotion_output = gr.Textbox(
            label="Detected Expression",
            value="Waiting for an image...",
            elem_id="emotion-result",
            interactive=False,
        )

        confidence_output = gr.Slider(
            minimum=0,
            maximum=100,
            value=0,
            step=0.1,
            label="Confidence",
            interactive=False,
        )

    response_output = gr.Textbox(
        label="Bot Response",
        value="Take a photo and press Analyze Expression.",
        lines=3,
        elem_id="response-box",
        interactive=False,
    )

    gr.Markdown(
        """
        **Important:** This application estimates visible facial expressions.
        It does not determine a person's true emotional or mental state.
        """
    )

    analyze_button.click(
        fn=analyze_emotion,
        inputs=input_image,
        outputs=[
            output_image,
            emotion_output,
            confidence_output,
            response_output,
        ],
    )


# Allow long DeepFace inference without the request being dropped
demo.queue(default_concurrency_limit=1)

# Load weights before serving traffic
warmup_models()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
    )
