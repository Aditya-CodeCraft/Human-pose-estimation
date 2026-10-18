"""MediaPipe Pose on a webcam feed or a video file, drawn frame by frame.

Usage:
    python experiments/mediapipe_webcam_pose.py            # webcam 0
    python experiments/mediapipe_webcam_pose.py video.mp4  # a video file

Press q to quit. Experiment from the AICTE internship (day 2). The main project is the OpenCV DNN Streamlit app in app/.
"""
import sys

import cv2
import mediapipe as mp


def main(source) -> int:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Could not open video source: {source}")
        return 1

    mirror = isinstance(source, int)  # mirror the webcam so it behaves like a mirror; leave files as recorded
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    landmark_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
    connection_style = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    ) as pose:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                print("Video capture ended.")
                break
            if mirror:
                frame = cv2.flip(frame, 1)

            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_style,
                    connection_style,
                )

            cv2.imshow("Pose estimation", cv2.resize(frame, (960, 540)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "0"
    sys.exit(main(int(arg) if arg.isdigit() else arg))
