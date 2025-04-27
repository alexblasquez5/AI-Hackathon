import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os
import pandas as pd
import time

# --- Import helper functions from detectors.py ---
from detectors import (
    calculate_angle,
    colorize_angle,
    posture_score,
    colorize_score,
    generate_posture_quality
)

# --- Constants ---
KNEE_THRESHOLD = 30
ELBOW_THRESHOLD = 30
HIP_MOVEMENT_THRESHOLD = 0.08
SHOULDER_MOVEMENT_THRESHOLD = 0.04

# --- Helper: Detect Movement Type ---
def detect_movement(knee_angles, elbow_angles, hip_ys, shoulder_ys):
    knee_variation = max(knee_angles) - min(knee_angles)
    elbow_variation = max(elbow_angles) - min(elbow_angles)
    hip_variation = max(hip_ys) - min(hip_ys)
    shoulder_variation = max(shoulder_ys) - min(shoulder_ys)

    if elbow_variation > ELBOW_THRESHOLD and shoulder_variation > SHOULDER_MOVEMENT_THRESHOLD:
        return "Push-ups"
    elif knee_variation > KNEE_THRESHOLD and hip_variation > HIP_MOVEMENT_THRESHOLD:
        return "Squats"
    else:
        return "Unknown Movement"

# --- MediaPipe setup ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# --- Streamlit App ---
st.title("Athlete Pose Analyzer + Smarter Movement Detection")

rotate_option = st.sidebar.selectbox("Rotate video (only if needed):", ("None", "90°", "180°", "270°"))
frame_skip = st.sidebar.slider("Frame Skip", 1, 5, 2)
allowed_image_types = ["jpg", "jpeg", "png"]
allowed_video_types = ["mp4", "mov", "avi"]
uploaded_file = st.file_uploader("Upload an image or video", type=allowed_image_types + allowed_video_types)

if uploaded_file:
    file_extension = os.path.splitext(uploaded_file.name)[1][1:].lower()

    if file_extension in allowed_image_types:
        # --- Process Image (Pose Analyzer) ---
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cv2.resize(image, (640, 480))

        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            landmarks = results.pose_landmarks.landmark

            detected_joints = {}

            # Safely calculate angles only if landmarks are visible and in reasonable position
            try:
                knee_landmark = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                if (knee_landmark.visibility > 0.5 and 0.3 <= knee_landmark.y <= 0.9):
                    hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    right_knee_angle = calculate_angle(hip, knee, ankle)
                    detected_joints['Right Knee'] = right_knee_angle
            except:
                pass

            try:
                if (landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility > 0.5 and
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility > 0.5 and
                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility > 0.5):
                    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    right_elbow_angle = calculate_angle(shoulder, elbow, wrist)
                    detected_joints['Right Elbow'] = right_elbow_angle
            except:
                pass

            try:
                if (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility > 0.5 and
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility > 0.5 and
                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility > 0.5):
                    neck = [(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) / 2,
                            (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2]
                    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_shoulder_angle = calculate_angle(neck, shoulder, elbow)
                    detected_joints['Right Shoulder'] = right_shoulder_angle
            except:
                pass

            st.image(image, channels='BGR')

            st.subheader("Detected Joint Angles:")
            for joint_name, angle in detected_joints.items():
                st.markdown(f"- **{joint_name}**: {colorize_angle(angle, joint_name.lower())}", unsafe_allow_html=True)

            if detected_joints:
                report_df = pd.DataFrame({
                    "Joint": list(detected_joints.keys()),
                    "Angle": [f"{int(angle)}°" for angle in detected_joints.values()]
                })

                csv = report_df.to_csv(index=False).encode('utf-8')
                st.download_button(label="Download Pose Report", data=csv, file_name='pose_report.csv', mime='text/csv')

        else:
            st.error("No pose landmarks detected. Please try another image.")

    elif file_extension in allowed_video_types:
        # --- Process Video (Movement Analyzer) ---
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_num = 0

        knee_angles, elbow_angles, shoulder_angles = [], [], []
        hip_ys, shoulder_ys = [], []
        best_knee_angle, worst_elbow_angle = 0, 180
        best_knee_frame, worst_elbow_frame = 0, 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1

            if frame_num % frame_skip != 0:
                continue

            frame = cv2.resize(frame, (640, 480))

            if rotate_option == "90°":
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotate_option == "180°":
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotate_option == "270°":
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

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
                hip_ys.append(hip[1])
                shoulder_ys.append(shoulder[1])

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

            raw_score = posture_score(avg_knee, avg_elbow, avg_shoulder)
            score = round((raw_score / 90) * 100)
            posture_quality = generate_posture_quality(avg_knee, avg_elbow, avg_shoulder)

            detected_movement = detect_movement(knee_angles, elbow_angles, hip_ys, shoulder_ys)

            # --- Display Results ---
            st.subheader("Posture Quality Assessment")
            for quality in posture_quality:
                st.write(quality)

            st.subheader("Athlete Score")
            st.markdown(colorize_score(score), unsafe_allow_html=True)

            st.subheader("Detected Movement Type")
            st.write(f"**Detected Movement:** {detected_movement}")

            report_df = pd.DataFrame({
                "Metric": [
                    "Avg Right Knee Angle", "Avg Right Elbow Angle", "Avg Right Shoulder Angle",
                    "Best Right Knee Angle", "Best Knee Timestamp (sec)",
                    "Worst Right Elbow Angle", "Worst Elbow Timestamp (sec)",
                    "Posture Quality", "Athlete Score", "Detected Movement"
                ],
                "Value": [
                    f"{int(avg_knee)}°", f"{int(avg_elbow)}°", f"{int(avg_shoulder)}°",
                    f"{int(best_knee_angle)}°", f"{best_knee_time:.2f} sec",
                    f"{int(worst_elbow_angle)}°", f"{worst_elbow_time:.2f} sec",
                    ", ".join(posture_quality), f"{score} / 100", detected_movement
                ]
            })

            csv = report_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Full Performance Report", data=csv, file_name='performance_report.csv', mime='text/csv')
