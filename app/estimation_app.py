"""Human pose estimation on a single image with OpenCV's DNN module and a pre-trained OpenPose-style model.

Run from the repository root:
    streamlit run app/estimation_app.py

The model (graph_opt.pb) and the demo image (stand.jpg) are loaded relative to this file, so the app works from any
working directory.
"""
import io
import json
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


# Joints whose angle is useful for exercise-form feedback: (name, first point, vertex, second point).
JOINT_ANGLES = [
    ("Right elbow", "RShoulder", "RElbow", "RWrist"),
    ("Left elbow", "LShoulder", "LElbow", "LWrist"),
    ("Right knee", "RHip", "RKnee", "RAnkle"),
    ("Left knee", "LHip", "LKnee", "LAnkle"),
    ("Right hip", "RShoulder", "RHip", "RKnee"),
    ("Left hip", "LShoulder", "LHip", "LKnee"),
    ("Right shoulder", "RElbow", "RShoulder", "RHip"),
    ("Left shoulder", "LElbow", "LShoulder", "LHip"),
]


def joint_angle(a, b, c):
    """Angle in degrees at vertex b formed by points a-b-c, or None if any point is missing."""
    if a is None or b is None or c is None:
        return None
    ba = np.array(a, dtype=float) - np.array(b, dtype=float)
    bc = np.array(c, dtype=float) - np.array(b, dtype=float)
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return None
    cos_angle = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def midpoint(a, b):
    """Midpoint of two points, or None if either is missing."""
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def torso_lean(neck, mid_hip):
    """Lean of the neck-to-hip line from vertical, in degrees (0 = upright), or None if a point is missing.

    Image y grows downwards, so the vertical reference points from the hips up to the neck.
    """
    if neck is None or mid_hip is None:
        return None
    return joint_angle(neck, mid_hip, (mid_hip[0], mid_hip[1] - 1))


def pose_detector(frame: np.ndarray, net, threshold: float):
    """Return a copy of the frame with detected keypoints and limbs drawn on it, plus the list of keypoints.

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

    return frame, points


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

output, points = pose_detector(image, net, threshold)

st.subheader("Estimated pose")
st.image(output, caption="Estimated pose", width="stretch")

buffer = io.BytesIO()
Image.fromarray(output).save(buffer, format="PNG")
st.download_button("Download annotated image", data=buffer.getvalue(), file_name="pose.png", mime="image/png")

st.subheader("Joint angles")
rows = []
for name, a, b, c in JOINT_ANGLES:
    angle = joint_angle(points[BODY_PARTS[a]], points[BODY_PARTS[b]], points[BODY_PARTS[c]])
    rows.append({"Joint": name, "Angle (deg)": "not detected" if angle is None else f"{angle:.0f}"})
lean = torso_lean(points[BODY_PARTS["Neck"]], midpoint(points[BODY_PARTS["RHip"]], points[BODY_PARTS["LHip"]]))
rows.append({"Joint": "Torso lean from vertical", "Angle (deg)": "not detected" if lean is None else f"{lean:.0f}"})
st.table(rows)

keypoints = {name: (None if points[idx] is None else {"x": points[idx][0], "y": points[idx][1]})
             for name, idx in BODY_PARTS.items() if name != "Background"}
angles = {row["Joint"]: (None if row["Angle (deg)"] == "not detected" else float(row["Angle (deg)"])) for row in rows}
export = {"image_size": {"width": int(image.shape[1]), "height": int(image.shape[0])},
          "threshold": threshold, "keypoints": keypoints, "angles_deg": angles}
st.download_button("Download keypoints as JSON", data=json.dumps(export, indent=2),
                   file_name="pose.json", mime="application/json")
