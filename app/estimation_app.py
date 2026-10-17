"""Human pose estimation on a single image with OpenCV's DNN module and a pre-trained OpenPose-style model.

Run from the repository root:
    streamlit run app/estimation_app.py

The model (graph_opt.pb) and the demo image (stand.jpg) are loaded relative to this file, so the app works from any
working directory.
"""
import threading
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "graph_opt.pb"
DEMO_IMAGE = BASE_DIR / "stand.jpg"

BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18,
}

POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"],
]

IN_WIDTH = 368
IN_HEIGHT = 368

_NET_LOCK = threading.Lock()  # the cached network is shared by all sessions and cv2.dnn.Net is not thread-safe


@st.cache_resource
def load_network():
    """Load the TensorFlow graph once per server process."""
    return cv2.dnn.readNetFromTensorflow(str(MODEL_PATH))


def pose_detector(frame: np.ndarray, net, threshold: float) -> np.ndarray:
    """Return a copy of the frame with detected keypoints and limbs drawn on it.

    Each of the 19 heatmaps (18 body parts plus background) is reduced to its most confident point; points below
    `threshold` are dropped, and the remaining points are joined using the fixed list of limb pairs in POSE_PAIRS. The
    background point is never drawn because no limb pair uses it.
    """
    frame = frame.copy()
    frame_height, frame_width = frame.shape[:2]

    with _NET_LOCK:
        net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (IN_WIDTH, IN_HEIGHT), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out = net.forward()[:, :19, :, :]
    assert len(BODY_PARTS) == out.shape[1]

    points = []
    for i in range(len(BODY_PARTS)):
        heat_map = out[0, i, :, :]
        _, confidence, _, point = cv2.minMaxLoc(heat_map)
        x = (frame_width * point[0]) / out.shape[3]
        y = (frame_height * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if confidence > threshold else None)

    for part_from, part_to in POSE_PAIRS:
        id_from, id_to = BODY_PARTS[part_from], BODY_PARTS[part_to]
        if points[id_from] and points[id_to]:
            cv2.line(frame, points[id_from], points[id_to], (0, 255, 0), 3)
            cv2.ellipse(frame, points[id_from], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[id_to], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

    return frame


st.title("Human Pose Estimation (OpenCV)")
st.text("Use a clear, full-body image with all limbs visible.")

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()
net = load_network()

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
# Convert to RGB so PNG files with an alpha channel (4 channels) do not break the network input.
image = np.array(Image.open(uploaded if uploaded is not None else DEMO_IMAGE).convert("RGB"))

st.subheader("Original image")
st.image(image, caption="Original image", width="stretch")

threshold = st.slider("Confidence threshold for keypoints (%)", min_value=0, max_value=100, value=20, step=5) / 100

output = pose_detector(image, net, threshold)

st.subheader("Estimated pose")
st.image(output, caption="Estimated pose", width="stretch")
