"""
SmartBike - Real-Time Bicycle Safety using YOLOv8, MediaPipe Hand Tracking, and Streamlit

Author: Amir Mobasheraghdam
Website: https://www.nivta.de
"""

import cv2
import numpy as np
import pyttsx3
import threading
from ultralytics import YOLO
import time
import mediapipe as mp
import streamlit as st

# Initialize speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

# Load YOLO model
model = YOLO("models/yolov8n.pt")

# MediaPipe Hand Tracking initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Classes of interest and real-world widths for distance estimation
important_classes = ["person", "car", "bicycle", "motorcycle", "bus", "traffic light"]
REAL_WIDTHS = {"person": 0.5, "car": 1.8, "bicycle": 0.7, "motorcycle": 0.8, "bus": 2.5}
FOCAL_LENGTH = 600

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Streamlit UI setup
st.title("🚴‍♂️ SmartBike - Real-Time Safety")
run = st.checkbox('Start SmartBike System')
FRAME_WINDOW = st.image([])

# Setup video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Error: Camera not available.")
    exit()

while run:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    results = model.predict(frame, save=False, conf=0.5, verbose=False)[0]

    speech_text = ""
    red_light_detected = False

    # Hand gesture detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb_frame)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]

        if class_name not in important_classes:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center_x = int((x1 + x2) / 2)
        box_width = x2 - x1

        position = "Left" if center_x < width / 3 else "Right" if center_x > 2 * width / 3 else "Center"

        if class_name in REAL_WIDTHS and box_width > 0:
            distance = round((REAL_WIDTHS[class_name] * FOCAL_LENGTH) / box_width, 2)
            danger = distance < 1.0 and position in ["Left", "Right"]

            label = f"{class_name} - {distance}m"
            color = (0, 0, 255) if danger else (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if danger:
                scenario = "approaching from behind" if position == "Center" else f"on your {position.lower()}"
                speech_text += f"Warning: {class_name} {scenario}, too close. "
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 255, 100), 2)
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if class_name == "traffic light":
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                avg_color = roi.mean(axis=0).mean(axis=0)
                if avg_color[2] > avg_color[1] and avg_color[2] > avg_color[0]:
                    red_light_detected = True

    if red_light_detected:
        speech_text += "Red light ahead. Please stop. "

    if speech_text:
        threading.Thread(target=speak, args=(speech_text,)).start()

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()

if not run:
    st.info("SmartBike System is stopped.")
