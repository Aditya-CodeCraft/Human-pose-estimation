"""MediaPipe Pose on a single image: prints the 33 landmarks and shows them drawn on the image.

Usage:
    python experiments/mediapipe_image_landmarks.py path/to/image.jpg

Experiment from the AICTE internship (day 2). The main project is the OpenCV DNN Streamlit app in app/.
"""
import sys

import cv2
import mediapipe as mp


def main(image_path: str) -> int:
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return 1

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
        print("No pose landmarks detected.")
        return 2

    height, width, _ = image.shape
    points_only = image.copy()
    for idx, landmark in enumerate(results.pose_landmarks.landmark):
        print(f"Landmark {idx}: (x: {landmark.x:.3f}, y: {landmark.y:.3f}, z: {landmark.z:.3f})")
        cv2.circle(points_only, (int(landmark.x * width), int(landmark.y * height)), 5, (0, 255, 0), -1)

    annotated = image.copy()
    mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Pose landmarks", points_only)
    cv2.imshow("Pose drawing", annotated)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
