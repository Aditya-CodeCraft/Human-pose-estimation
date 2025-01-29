import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils  # For drawing landmarks

# Capture Video: Replace with 'path_to_video.mp4' for a file or 0 for webcam

video_source=0
#video_source = "B:\google baba ka downloads\WhatsApp Video 2024-12-28 at 19.00.05.mp4"  #Replace with video file path or 0 for live webcam
cap = cv2.VideoCapture(video_source)

# Set video resolution for webcam (optional, adjust as needed for your laptop)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 350)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 700)

# Initialize MediaPipe Pose
with mp_pose.Pose(static_image_mode=False, model_complexity=0, enable_segmentation=False, min_detection_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video capture ended.")
            break
        # Flip frame horizontally for a mirror-like effect (optional)
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        results = pose.process(frame_rgb)

        # Draw landmarks on the frame
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  # Landmarks
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)   # Connections
            )

        # Resize frame to fit the screen (optional)
        display_frame = cv2.resize(frame, (960, 540))  # Adjust width and height as per your screen

        cv2.imshow('Pose Estimation',display_frame)#dispaly processed frame

        #beak loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
