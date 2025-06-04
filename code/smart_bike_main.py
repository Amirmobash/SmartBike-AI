"""
SmartBike - Real-Time Bicycle Safety using YOLOv8

Author: Amir Mobasheraghdam
Website: https://www.nivta.de
"""

import cv2
import numpy as np
import pyttsx3
import threading
from ultralytics import YOLO

engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

model = YOLO("models/yolov8n.pt")
important_classes = ["person", "car", "bicycle", "motorcycle", "bus", "traffic light"]
REAL_WIDTHS = {
    "person": 0.5,
    "car": 1.8,
    "bicycle": 0.7,
    "motorcycle": 0.8,
    "bus": 2.5
}
FOCAL_LENGTH = 600

def speak(text):
    engine.say(text)
    engine.runAndWait()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not available.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    results = model.predict(source=frame, save=False, conf=0.5, verbose=False)[0]

    speech_text = ""
    red_light_detected = False

    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]

        if class_name not in important_classes:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center_x = int((x1 + x2) / 2)
        box_width = x2 - x1
        position = "left" if center_x < width / 3 else "right" if center_x > 2 * width / 3 else "center"

        if class_name in REAL_WIDTHS and box_width > 0:
            distance = (REAL_WIDTHS[class_name] * FOCAL_LENGTH) / box_width
            distance = round(distance, 2)
            if distance < 1.0 and position in ["left", "right"]:
                speech_text += f"{class_name} too close on {position}. "
            label = f"{class_name} - {distance}m"
            color = (0, 255, 0) if distance > 1 else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 255, 100), 2)
            cv2.putText(frame, class_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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

    cv2.imshow("SmartBike YOLOv8", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
