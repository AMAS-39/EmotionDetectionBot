import cv2


def main() -> None:
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        print("Error: Could not open the webcam.")
        print("Close Camera, Zoom, Teams, Discord, or other apps using it.")
        return

    print("Camera started successfully.")
    print("Press Q to close the application.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Error: Could not read a frame from the webcam.")
            break

        # Mirror the image naturally.
        frame = cv2.flip(frame, 1)

        cv2.imshow("AI Emotion Detection Bot - Camera Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()