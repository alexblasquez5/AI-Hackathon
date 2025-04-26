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

# --- MediaPipe setup ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# --- Streamlit App ---
st.title("Athlete Pose Analyzer + Smarter Movement Detection")

rotate_option = st.sidebar.selectbox("Rotate video (only if needed):", ("None", "90°", "180°", "270°"))
allowed_image_types = ["jpg", "jpeg", "png"]
allowed_video_types = ["mp4", "mov", "avi"]
uploaded_file = st.file_uploader("Upload an image or video", type=allowed_image_types + allowed_video_types)

if uploaded_file:
    file_extension = os.path.splitext(uploaded_file.name)[1][1:].lower()

    if file_extension in allowed_image_types:
        st.info("Image uploads: Movement detection only works with videos.")

    elif file_extension in allowed_video_types:
        # --- Process Video ---
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
                hip_ys.append(hip[1])
                shoulder_ys.append(shoulder[1])

                if right_knee_angle > best_knee_angle:
                    best_knee_angle = right_knee_angle
                    best_knee_frame = frame_num
                if right_elbow_angle < worst_elbow_angle:
                    worst_elbow_angle = right_elbow_angle
                    worst_elbow_frame = frame_num

                stframe.image(frame, channels="BGR")
                time.sleep(1 / fps)

        cap.release()

        if knee_angles and elbow_angles and shoulder_angles:
            avg_knee = np.mean(knee_angles)
            avg_elbow = np.mean(elbow_angles)
            avg_shoulder = np.mean(shoulder_angles)
            best_knee_time = best_knee_frame / fps
            worst_elbow_time = worst_elbow_frame / fps

            score = posture_score(avg_knee, avg_elbow, avg_shoulder)
            posture_quality = generate_posture_quality(avg_knee, avg_elbow, avg_shoulder)

            # --- Smarter Movement Detection ---
            knee_variation = max(knee_angles) - min(knee_angles)
            elbow_variation = max(elbow_angles) - min(elbow_angles)
            hip_variation = max(hip_ys) - min(hip_ys)
            shoulder_variation = max(shoulder_ys) - min(shoulder_ys)

            knee_threshold = 30
            elbow_threshold = 30
            hip_movement_threshold = 0.08
            shoulder_movement_threshold = 0.04

            if elbow_variation > elbow_threshold and shoulder_variation > shoulder_movement_threshold:
                detected_movement = "Push-ups"
            elif knee_variation > knee_threshold and hip_variation > hip_movement_threshold:
                detected_movement = "Squats"
            else:
                detected_movement = "Unknown Movement"

            # --- Display Results ---
            st.subheader("Posture Quality Assessment")
            for quality in posture_quality:
                st.write(quality)

            st.subheader("Athlete Score")
            st.markdown(colorize_score(score), unsafe_allow_html=True)

            st.subheader("Detected Movement Type")
            st.write(f"**Detected Movement:** {detected_movement}")

            # --- Generate and Offer Report Download ---
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
                    ", ".join(posture_quality), f"{score} / 90", detected_movement
                ]
            })

            csv = report_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Full Performance Report", data=csv, file_name='performance_report.csv', mime='text/csv')
