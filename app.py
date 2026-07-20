import cv2
import numpy as np
import time
from collections import deque
from deepface import DeepFace


# -------------------- SETTINGS --------------------

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Analyze every 1 second
ANALYSIS_INTERVAL = 1.0

# Use recent results to avoid jumping
HISTORY_SIZE = 4

# Minimum accepted confidence
MIN_CONFIDENCE = 30.0


# -------------------- BOT RESPONSES --------------------

def get_bot_message(emotion: str) -> str:
    messages = {
        "happy": "You look happy! Keep smiling.",
        "sad": "You seem sad. I hope your day gets better.",
        "angry": "You seem upset. Try taking a slow breath.",
        "neutral": "You look calm and focused.",
        "surprise": "You look surprised!",
        "fear": "You seem worried. Take a moment to relax.",
        "disgust": "Something seems uncomfortable.",
        "uncertain": "I am still reading your expression.",
        "no face": "Please look toward the camera."
    }

    return messages.get(emotion, "I am analyzing your expression.")


# -------------------- MAIN APPLICATION --------------------

def main():
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not camera.isOpened():
        print("Error: Could not open webcam.")
        return

    # Fast OpenCV face detector
    face_cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    face_detector = cv2.CascadeClassifier(face_cascade_path)

    if face_detector.empty():
        print("Error: Face detector could not be loaded.")
        camera.release()
        return

    emotion_history = deque(maxlen=HISTORY_SIZE)

    current_emotion = "Detecting..."
    current_confidence = 0.0
    current_face = None

    last_analysis_time = 0

    print("Emotion Detection Bot started.")
    print("Press Q to close.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Error: Could not read webcam frame.")
            break

        frame = cv2.flip(frame, 1)

        # Convert to grayscale for faster face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=6,
            minSize=(100, 100)
        )

        current_face = None

        if len(faces) > 0:
            # Choose the largest detected face
            x, y, width, height = max(
                faces,
                key=lambda face: face[2] * face[3]
            )

            current_face = (x, y, width, height)

            # Add padding around the face
            padding = 30

            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + width + padding)
            y2 = min(frame.shape[0], y + height + padding)

            face_crop = frame[y1:y2, x1:x2]

            current_time = time.time()

            # Analyze only once per interval
            if (
                current_time - last_analysis_time
                >= ANALYSIS_INTERVAL
            ):
                last_analysis_time = current_time

                try:
                    result = DeepFace.analyze(
                        img_path=face_crop,
                        actions=["emotion"],

                        # We already detected and cropped the face
                        detector_backend="skip",

                        enforce_detection=False,
                        align=False,
                        silent=True
                    )

                    if isinstance(result, list):
                        result = result[0]

                    scores = result["emotion"]

                    # Store scores from recent analyses
                    emotion_history.append(scores)

                    average_scores = {}

                    for emotion_name in scores:
                        values = [
                            prediction[emotion_name]
                            for prediction in emotion_history
                        ]

                        average_scores[emotion_name] = float(
                            np.mean(values)
                        )

                    best_emotion = max(
                        average_scores,
                        key=average_scores.get
                    )

                    best_confidence = average_scores[best_emotion]

                    if best_confidence >= MIN_CONFIDENCE:
                        current_emotion = best_emotion
                        current_confidence = best_confidence
                    else:
                        current_emotion = "uncertain"
                        current_confidence = best_confidence

                except Exception as error:
                    print(f"Emotion analysis error: {error}")

        else:
            current_emotion = "no face"
            current_confidence = 0
            emotion_history.clear()

        # -------------------- DRAW FACE BOX --------------------

        if current_face is not None:
            x, y, width, height = current_face

            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2
            )

        # -------------------- INFORMATION PANEL --------------------

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (10, 10),
            (620, 155),
            (0, 0, 0),
            -1
        )

        cv2.addWeighted(
            overlay,
            0.70,
            frame,
            0.30,
            0,
            frame
        )

        if current_emotion in [
            "Detecting...",
            "no face",
            "uncertain"
        ]:
            emotion_text = current_emotion.title()
        else:
            emotion_text = (
                f"{current_emotion.title()} "
                f"({current_confidence:.1f}%)"
            )

        cv2.putText(
            frame,
            f"Emotion: {emotion_text}",
            (30, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Bot: {get_bot_message(current_emotion)}",
            (30, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Press Q to exit",
            (30, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1
        )

        cv2.imshow("AI Emotion Detection Bot", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()