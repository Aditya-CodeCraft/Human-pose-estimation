# 🧍 Human Pose Estimation

Detects and draws human body keypoints (shoulders, elbows, knees and more) in images and video, using machine learning.

**🔗 Live demo:** https://velvety-twilight-9e0b82.netlify.app/ (browser version built with ml5.js and p5.js)

---

## 💡 Motivation

People often do gym exercises with incorrect form because personal trainers are expensive, which raises the risk of injury and makes workouts less effective. Pose estimation can give automatic feedback on posture and form.

## 🎯 Applications

- **Healthcare:** patient monitoring, rehabilitation and physical therapy
- **Sports analytics:** performance analysis and injury prevention
- **Security:** understanding human behaviour to spot suspicious activity
- **Interactive apps:** natural, body-driven input for games and VR

## ⚙️ Implementation

| Part | Approach |
|---|---|
| Image pose estimation | Detects and annotates poses in still images |
| Video pose estimation | Detects and annotates poses frame by frame in video |
| Streamlit app (`estimation_app.py`) | OpenCV DNN with a pre-trained OpenPose-style TensorFlow model (`graph_opt.pb`) and an adjustable confidence threshold |
| Experiments (`Day2-P1.py`, `day2-p2.py`) | MediaPipe Pose |
| Web demo | ml5.js and p5.js, deployed on Netlify |

## 🚀 Run Locally

```bash
git clone https://github.com/Aditya-CodeCraft/Human-pose-estimation.git
cd Human-pose-estimation
pip install -r requirements.txt
cd "Human Pose Estimation using Machine Learning"
streamlit run estimation_app.py
```

Upload a clear, full-body image, or use the bundled `stand.jpg`, then adjust the threshold slider to control how confident a detection must be before it's drawn.

## 📸 Output

![Pose estimation output](Human%20Pose%20Estimation%20using%20Machine%20Learning/OutPut-image.png)

## 🛠️ Tech Stack

Python · OpenCV · MediaPipe · Streamlit · NumPy · ml5.js · p5.js

---

Built as part of an **AICTE internship** project.
