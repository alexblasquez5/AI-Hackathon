# detectors.py
import numpy as np

class MovementDetector:
    def __init__(self):
        self.squat_down = False
        self.pushup_down = False

    @staticmethod
    def calculate_angle(a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def detect(self, landmarks):
        """Detect squat and pushup reps based on movement."""
        right_hip = [landmarks.landmark[24].x, landmarks.landmark[24].y]
        right_knee = [landmarks.landmark[26].x, landmarks.landmark[26].y]
        right_ankle = [landmarks.landmark[28].x, landmarks.landmark[28].y]
        right_shoulder = [landmarks.landmark[12].x, landmarks.landmark[12].y]
        right_elbow = [landmarks.landmark[14].x, landmarks.landmark[14].y]
        right_wrist = [landmarks.landmark[16].x, landmarks.landmark[16].y]

        knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
        elbow_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        hip_angle = self.calculate_angle(right_shoulder, right_hip, right_knee)

        detection = None

        # Squat detection
        if knee_angle < 90 and hip_angle < 140:
            self.squat_down = True
        elif knee_angle > 160 and self.squat_down:
            self.squat_down = False
            detection = "Squat completed!"

        # Pushup detection
        if elbow_angle < 90:
            self.pushup_down = True
        elif elbow_angle > 160 and self.pushup_down:
            self.pushup_down = False
            detection = "Pushup completed!"

        return detection