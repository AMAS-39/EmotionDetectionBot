import os

# Reduce unnecessary TensorFlow messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

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
        "message": "You appear sad. Take a moment, breathe, and remember that difficult feelings can pass."
    },
    "angry": {
        "emoji": "😠",
        "message": "You appear upset. Try taking a slow breath and giving yourself a moment to relax."
    },
    "surprise": {
        "emoji": "😮",
        "message": "You look surprised! Something seems to have caught your attention."
    },
    "fear": {
        "emoji": "😨",
        "message": "You appear worried. Try to stay calm and focus on your breathing."
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


def analyze_emotion(image):
    """
    Analyze a webcam snapshot or uploaded image.

    Gradio provides the image as an RGB NumPy array.
    DeepFace and OpenCV normally work with BGR images.
    """

    if image is None:
        return (
            None,
            "No image provided",
            0,
            "Please take a webcam photo or upload an image."
        )

    try:
        # Convert Gradio RGB image to OpenCV BGR
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        results = DeepFace.analyze(
            img_path=bgr_image,
            actions=["emotion"],
            detector_backend="opencv",
            enforce_detection=True,
            align=True,
            silent=True
        )

        # DeepFace normally returns a list of detected faces
        if isinstance(results, list):
            if len(results) == 0:
                raise ValueError("No face was detected.")

            # Select the largest face if several faces are present
            result = max(
                results,
                key=lambda item: (
                    item.get("region", {}).get("w", 0)
                    * item.get("region", {}).get("h", 0)
                )
            )
        else:
            result = results

        emotion_scores = result.get("emotion", {})
        dominant_emotion = result.get("dominant_emotion", "neutral").lower()

        confidence = float(
            emotion_scores.get(dominant_emotion, 0.0)
        )

        details = EMOTION_DETAILS.get(
            dominant_emotion,
            {
                "emoji": "🤖",
                "message": "The system detected a facial expression."
            }
        )

        # Copy image so the original input is not modified
        output_image = bgr_image.copy()

        region = result.get("region", {})

        x = max(0, int(region.get("x", 0)))
        y = max(0, int(region.get("y", 0)))
        width = max(0, int(region.get("w", 0)))
        height = max(0, int(region.get("h", 0)))

        # Draw face rectangle
        if width > 0 and height > 0:
            cv2.rectangle(
                output_image,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                3
            )

            label = (
                f"{dominant_emotion.title()} "
                f"{confidence:.1f}%"
            )

            label_y = max(30, y - 12)

            cv2.putText(
                output_image,
                label,
                (x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        # Convert the annotated image back to RGB for Gradio
        output_rgb = cv2.cvtColor(
            output_image,
            cv2.COLOR_BGR2RGB
        )

        emotion_text = (
            f"{details['emoji']} "
            f"{dominant_emotion.title()}"
        )

        return (
            output_rgb,
            emotion_text,
            round(confidence, 1),
            details["message"]
        )

    except ValueError:
        return (
            image,
            "No face detected",
            0,
            (
                "No clear face was detected. Face the camera directly, "
                "use good lighting, and try again."
            )
        )

    except Exception as error:
        print(f"Analysis error: {error}")

        return (
            image,
            "Analysis failed",
            0,
            (
                "The image could not be analyzed. Please take another "
                "photo with your face clearly visible."
            )
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
    css=CUSTOM_CSS
) as demo:

    gr.Markdown(
        """
        # 🤖 AI Facial Expression Detection Bot
        """,
        elem_id="main-title"
    )

    gr.Markdown(
        """
        Take a webcam photo or upload an image. The AI will detect the
        dominant visible facial expression and provide a responsive message.
        """,
        elem_id="subtitle"
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Webcam or Image",
                sources=["webcam", "upload"],
                type="numpy",
                height=430
            )

            analyze_button = gr.Button(
                "Analyze Expression",
                variant="primary",
                size="lg"
            )

            clear_button = gr.ClearButton(
                value="Clear",
                components=[input_image]
            )

        with gr.Column(scale=1):
            output_image = gr.Image(
                label="Detection Result",
                height=430,
                interactive=False
            )

    with gr.Row():
        emotion_output = gr.Textbox(
            label="Detected Expression",
            value="Waiting for an image...",
            elem_id="emotion-result",
            interactive=False
        )

        confidence_output = gr.Slider(
            minimum=0,
            maximum=100,
            value=0,
            step=0.1,
            label="Confidence",
            interactive=False
        )

    response_output = gr.Textbox(
        label="Bot Response",
        value="Take a photo and press Analyze Expression.",
        lines=3,
        elem_id="response-box",
        interactive=False
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
            response_output
        ]
    )


if __name__ == "__main__":
    demo.launch()