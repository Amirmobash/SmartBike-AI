
# SmartBike-AI

**SmartBike-AI** is an AI-powered object detection assistant for cyclists, built using YOLOv8 and real-time voice feedback. It’s designed to enhance safety and awareness for all riders, especially those who are blind or visually impaired.

---

## ✨ Features
- Real-time object detection with YOLOv8
- Voice announcements for detected objects and their estimated distances
- Designed for street-relevant objects only (cars, people, bicycles, etc.)
- Lightweight interface, suitable for small helmet-mounted displays
- Full offline operation (no internet required)
- Customizable detection radius and speech patterns

---

## 💻 Requirements
- Python 3.8 or higher
- A working webcam (USB or helmet-mounted)

### Python packages
Use the following command to install dependencies:
```bash
pip install -r requirements.txt
```

### `requirements.txt` sample
```
ultralytics
opencv-python
numpy
pyttsx3
```

---

## ⚖️ How It Works
1. Run the script: `python bike_yolo8.py`
2. The webcam captures live video frames
3. YOLOv8 detects relevant objects in the frame
4. Their distance and location (left, center, right) are calculated
5. The system speaks the object and distance (e.g., "car ahead, right")
6. Results are shown full-screen for compact helmet displays

---

## 🛠️ Hardware Suggestions
- Jetson Orin Nano / Xavier NX / Orange Pi 5
- Helmet-mounted USB webcam
- Mini speaker or Bluetooth audio
- Power source (battery pack or USB-C power)

---

## 📸 Demo & Media
*Coming soon: Live street tests and demo video*

---

## 🌟 Credits
Developed by an independent creator to improve cycling safety and empower accessible urban mobility.

📲 Follow the project on Instagram: [@amirmobasher.ir](https://www.instagram.com/amirmobasher.ir/)

---

## 📌 Tags
`#AI` `#YOLOv8` `#SmartBike` `#ObjectDetection` `#BlindSupport` `#CyclingAI` `#Ultralytics`

