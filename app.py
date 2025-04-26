import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os
import pandas as pd

from detectors import MovementDetector


# --- Helper functions ---

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

def colorize_angle(angle, joint_name):
    color = "gray"
    if joint_name == "knee":
        if 160 <= angle <= 180:
            color = "green"
        elif 140 <= angle < 160:
            color = "orange"
        else:
            color = "red"
    elif joint_name in ["elbow", "shoulder"]:
        if 70 <= angle <= 110:
            color = "green"
        elif 50 <= angle < 70 or 110 < angle <= 130:
            color = "orange"
        else:
            color = "red"
    return f"<span style='color:{color}'>{int(angle)}°</span>"

def posture_score(avg_knee, avg_elbow, avg_shoulder):
    score = 0
    for angle, low, high in [(avg_knee, 160, 180), (avg_elbow, 70, 110), (avg_shoulder, 70, 110)]:
        if low <= angle <= high:
            score += 30
        elif (low-20) <= angle < low or high < angle <= (high+20):
            score += 15
    return score

def colorize_score(score):
    if score >= 75:
        color = "green"
    elif score >= 45:
        color = "orange"
    else:
        color = "red"
    return f"<span style='color:{color}; font-size:24px'><b>{score} / 90</b></span>"

def generate_report(avg_knee, avg_elbow, avg_shoulder, best_knee, best_knee_time, worst_elbow, worst_elbow_time, score):
    quality = []
    for avg, joint in [(avg_knee, "Knee"), (avg_elbow, "Elbow"), (avg_shoulder, "Shoulder")]:
        if 160 <= avg <= 180 or 70 <= avg <= 110:
            quality.append(f"Good {joint} Posture")
        else:
            quality.append("Needs Improvement")
    data = {
        "Metric": [
            "Avg Right Knee Angle", "Avg Right Elbow Angle", "Avg Right Shoulder Angle",
            "Best Right Knee Angle (max)", "Best Knee Timestamp (sec)",
            "Worst Right Elbow Angle (min)", "Worst Elbow Timestamp (sec)",
            "Posture Quality", "Athlete Score (out of 90)"
        ],
        "Value": [
            f"{int(avg_knee)}°", f"{int(avg_elbow)}°", f"{int(avg_shoulder)}°",
            f"{int(best_knee)}°", f"{best_knee_time:.2f} sec",
            f"{int(worst_elbow)}°", f"{worst_elbow_time:.2f} sec",
            ", ".join(quality), f"{score} / 90"
        ]
    }
    return pd.DataFrame(data)

# --- MediaPipe setup ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# --- Streamlit setup ---
st.title("Athlete Pose Analyzer")

movement_detector = MovementDetector()

mode = st.sidebar.radio("Select Mode:", ("Upload Image/Video", "Live Camera"))

if mode == "Upload Image/Video":
    st.write("Upload an image or video, and we'll detect the athlete's pose!")

    rotate_option = st.sidebar.selectbox("Rotate video (only if needed):", ("None", "90°", "180°", "270°"))
    allowed_image_types = ["jpg", "jpeg", "png"]
    allowed_video_types = ["mp4", "mov", "avi"]
    uploaded_file = st.file_uploader("Upload an image or video", type=allowed_image_types + allowed_video_types)

    if uploaded_file:
        file_extension = os.path.splitext(uploaded_file.name)[1][1:].lower()

        if file_extension in allowed_image_types:
            # Process Image
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                neck = [(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) / 2,
                        (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2]

                right_knee_angle = calculate_angle(hip, knee, ankle)
                right_elbow_angle = calculate_angle(shoulder, elbow, wrist)
                right_shoulder_angle = calculate_angle(neck, shoulder, elbow)

                score = posture_score(right_knee_angle, right_elbow_angle, right_shoulder_angle)

                st.subheader("📈 Athlete Stats:")
                st.markdown(f"""
                - **Right Knee Angle:** {colorize_angle(right_knee_angle, 'knee')}
                - **Right Elbow Angle:** {colorize_angle(right_elbow_angle, 'elbow')}
                - **Right Shoulder Angle:** {colorize_angle(right_shoulder_angle, 'shoulder')}
                """, unsafe_allow_html=True)

                st.subheader("🏅 Athlete Score")
                st.markdown(colorize_score(score), unsafe_allow_html=True)

        else:
            # Process Video
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_num = 0
            knee_angles, elbow_angles, shoulder_angles = [], [], []
            best_knee_angle, worst_elbow_angle = 0, 180
            best_knee_frame, worst_elbow_frame = 0, 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (640, 480))

                if rotate_option == "90°":
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif rotate_option == "180°":
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif rotate_option == "270°":
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                frame_num += 1

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    landmarks = results.pose_landmarks.landmark
                    hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    neck = [(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) / 2,
                            (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2]

                    right_knee_angle = calculate_angle(hip, knee, ankle)
                    right_elbow_angle = calculate_angle(shoulder, elbow, wrist)
                    right_shoulder_angle = calculate_angle(neck, shoulder, elbow)

                    knee_angles.append(right_knee_angle)
                    elbow_angles.append(right_elbow_angle)
                    shoulder_angles.append(right_shoulder_angle)

                    if right_knee_angle > best_knee_angle:
                        best_knee_angle = right_knee_angle
                        best_knee_frame = frame_num
                    if right_elbow_angle < worst_elbow_angle:
                        worst_elbow_angle = right_elbow_angle
                        worst_elbow_frame = frame_num

                    stframe.image(frame, channels="BGR")

            cap.release()

            if knee_angles and elbow_angles and shoulder_angles:
                avg_knee = np.mean(knee_angles)
                avg_elbow = np.mean(elbow_angles)
                avg_shoulder = np.mean(shoulder_angles)
                best_knee_time = best_knee_frame / fps
                worst_elbow_time = worst_elbow_frame / fps

                score = posture_score(avg_knee, avg_elbow, avg_shoulder)

                st.subheader("🏅 Athlete Score")
                st.markdown(colorize_score(score), unsafe_allow_html=True)

                report_df = generate_report(avg_knee, avg_elbow, avg_shoulder, best_knee_angle, best_knee_time, worst_elbow_angle, worst_elbow_time, score)
                csv = report_df.to_csv(index=False).encode('utf-8')
                st.download_button(label="📥 Download Performance Report", data=csv, file_name='performance_report.csv', mime='text/csv')

elif mode == "Live Camera":
    st.write("Start your webcam and we'll detect your pose live!")
    run = st.checkbox('Start Camera')

    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to grab frame from camera.")
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # 🆕 Improved detection
            detection = movement_detector.detect(results.pose_landmarks)
            if detection:
                st.success(detection)

        FRAME_WINDOW.image(frame, channels="BGR")


    camera.release()