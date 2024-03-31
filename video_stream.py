import cv2
from typing import Tuple, List, Final

BOX_THICKNESS: Final[int] = 4
TEXT_SIZE: Final[int] = 1
TEXT_THICKNESS: Final[int] = 2
CV_COLOR: Final[Tuple[int, int, int]] = (255, 255, 0)

def load_face_detector() -> cv2.CascadeClassifier:
    """
    Load the pre-trained face detection classifier.

    Returns:
        cv2.CascadeClassifier: The loaded face detection classifier.
    """
    haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(haar_cascade_path)

def draw_face_box(frame: cv2.Mat, face_box: Tuple[int, int, int, int]) -> None:
    """
    Draw a rectangle around a detected face and add coordinate labels.

    Args:
        frame (cv2.Mat): The input frame.
        face_box (Tuple[int, int, int, int]): The coordinates (x, y, w, h) of the detected face.
    """
    x, y, w, h = face_box

    # Draw the bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), CV_COLOR, BOX_THICKNESS)

    # Calculate corners of bounding box
    top_left = (x, y)
    top_right = (x + w, y)
    bottom_left = (x, y + h)
    bottom_right = (x + w, y + h)

    # Add labels for each corner
    cv2.putText(frame, str(top_left), top_left, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, CV_COLOR, TEXT_THICKNESS)
    cv2.putText(frame, str(top_right), top_right, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, CV_COLOR, TEXT_THICKNESS)
    cv2.putText(frame, str(bottom_left), bottom_left, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, CV_COLOR, TEXT_THICKNESS)
    cv2.putText(frame, str(bottom_right), bottom_right, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, CV_COLOR, TEXT_THICKNESS)



def detect_faces(frame: cv2.Mat, face_detector: cv2.CascadeClassifier) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces in the given frame and return their bounding boxes.

    Args:
        frame (cv2.Mat): The input frame.
        face_detector (cv2.CascadeClassifier): The face detection classifier.

    Returns:
        List[Tuple[int, int, int, int]]: A list of detected face bounding boxes represented as (x, y, w, h) tuples.
    """
    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=10, minSize=(40, 40))

    return faces

def main() -> None:
    """
    Main function to capture video stream and detect faces in real-time.
    """
    face_detector: cv2.CascadeClassifier = load_face_detector()
    video_capture: cv2.VideoCapture = cv2.VideoCapture(0)

    # Set the desired frame rate (frames per second)
    fps: float = 30.0
    video_capture.set(cv2.CAP_PROP_FPS, fps)

    while True:
        # Read a frame from the video stream
        ret: bool
        frame: cv2.Mat
        ret, frame = video_capture.read()

        # If the frame could not be read, break the loop
        if not ret:
            break

        # Detect faces and draw bounding boxes
        face_boxes: List[Tuple[int, int, int, int]] = detect_faces(frame, face_detector)
        for face_box in face_boxes:
            draw_face_box(frame, face_box)

        # Display the frame with bounding boxes
        cv2.imshow("Face Detection", frame)

        # Check if the 'q' key is pressed to quit the program
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture object and close all windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()