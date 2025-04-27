import numpy as np

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
    if score >= 85:
        color = "green"
    elif score >= 60:
        color = "orange"
    else:
        color = "red"
    return f"<span style='color:{color}; font-size:24px'><b>{score} / 100</b></span>"


def generate_posture_quality(avg_knee, avg_elbow, avg_shoulder):
    quality = []
    if 160 <= avg_knee <= 180:
        quality.append("✅ Good Knee Posture")
    else:
        quality.append("⚠️ Needs Improvement on Knee")
    if 70 <= avg_elbow <= 110:
        quality.append("✅ Good Elbow Posture")
    else:
        quality.append("⚠️ Needs Improvement on Elbow")
    if 70 <= avg_shoulder <= 110:
        quality.append("✅ Good Shoulder Posture")
    else:
        quality.append("⚠️ Needs Improvement on Shoulder")
    return quality
