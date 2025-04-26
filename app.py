# app.py
import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Streamlit page setup
st.title("Athlete Pose Analyzer")

# Sidebar to choose between Upload or Live Mode
mode = st.sidebar.radio("Select Mode:", ("Upload Image/Video", "Live Camera"))

if mode == "Upload Image/Video":
    st.write("Upload an image or video, and we'll detect the athlete's pose!")

    # Rotate option for uploaded videos
    rotate_option = st.sidebar.selectbox(
        "Rotate video (only if needed):",
        ("None", "90°", "180°", "270°")
    )

    # Allowed file types
    allowed_image_types = ["jpg", "jpeg", "png"]
    allowed_video_types = ["mp4", "mov", "avi"]

    # Upload a file
    uploaded_file = st.file_uploader("Upload an image or video", type=allowed_image_types + allowed_video_types)

    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1][1:].lower()  # Get file extension without dot

        if file_extension in allowed_image_types:
            # Process Image
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = pose.process(image_rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            st.image(image, channels="BGR", caption="Processed Image")

        elif file_extension in allowed_video_types:
            # Process Video
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())

            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (640, 480))

                # Rotate frame if needed
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

                stframe.image(frame, channels="BGR")

            cap.release()

        else:
            st.error("Unsupported file type.")

elif mode == "Live Camera":
    st.write("Start your webcam and we'll detect your pose live!")
    run = st.checkbox('Start Camera')

    FRAME_WINDOW = st.image([])

    # OpenCV setup for webcam
    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to grab frame from camera.")
            break

        frame = cv2.flip(frame, 1)  # Flip horizontally for natural interaction
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        FRAME_WINDOW.image(frame, channels="BGR")

    camera.release()
