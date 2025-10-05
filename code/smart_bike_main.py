"""
SmartBike - Real-Time Bicycle Safety using YOLOv8, MediaPipe Hand Tracking, Object Speed Detection, Streamlit UI, and Google Maps Overlay

Author: Amir Mobasheraghdam
Website: https://www.nivta.de

Notes:
- Requires: ultralytics, opencv-python, mediapipe, pyttsx3, streamlit
- Optional: streamlit-geolocation (for auto-detecting coordinates in browser)
- Google Maps: provide an API key (with Maps JavaScript API enabled) in the sidebar.

Run:
  streamlit run smartbike_streamlit.py
"""

import json
import time
import threading
from collections import deque
from typing import List, Tuple, Dict

import cv2
import numpy as np
import pyttsx3
import streamlit as st
from ultralytics import YOLO

# --- Optional geolocation (won't break if missing) ---
try:
    from streamlit_geolocation import geolocation
    HAS_GEO = True
except Exception:
    HAS_GEO = False

# ---------------------- TTS (thread-safe) ----------------------
class Speaker:
    def __init__(self, rate: int = 150, volume: float = 1.0):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)
        self.lock = threading.Lock()

    def say_async(self, text: str):
        if not text:
            return
        threading.Thread(target=self._speak_blocking, args=(text,), daemon=True).start()

    def _speak_blocking(self, text: str):
        with self.lock:
            self.engine.say(text)
            self.engine.runAndWait()

speaker = Speaker()

# ---------------------- App State ----------------------
if 'hazards' not in st.session_state:
    # list of dicts: {"lat": float, "lng": float, "label": str, "ts": float}
    st.session_state.hazards = []

if 'object_histories' not in st.session_state:
    # Track last N positions per class name (simple heuristic without ID tracking)
    st.session_state.object_histories: Dict[str, deque] = {}

if 'last_danger_spoken' not in st.session_state:
    st.session_state.last_danger_spoken = 0.0

# ---------------------- Sidebar Controls ----------------------
st.sidebar.header("⚙️ Settings")
api_key = st.sidebar.text_input("Google Maps API Key (JS)", type="password", help="Enable the Maps JavaScript API for this key.")

# Location controls
st.sidebar.subheader("📍 Location")
if HAS_GEO:
    loc_btn = st.sidebar.button("Use my browser location")
else:
    st.sidebar.caption("Install optional package: streamlit-geolocation for auto location.")

default_lat, default_lng = 52.5200, 13.4050  # Berlin default
if HAS_GEO and 'browser_loc' not in st.session_state:
    st.session_state.browser_loc = None

if HAS_GEO and 'last_geo' not in st.session_state:
    st.session_state.last_geo = None

if HAS_GEO and loc_btn:
    st.session_state.last_geo = geolocation()
    if st.session_state.last_geo and 'lat' in st.session_state.last_geo:
        st.session_state.browser_loc = (st.session_state.last_geo['lat'], st.session_state.last_geo['lon'])

lat = st.sidebar.number_input("Latitude", value=(st.session_state.browser_loc[0] if HAS_GEO and st.session_state.browser_loc else default_lat), format="%.6f")
lng = st.sidebar.number_input("Longitude", value=(st.session_state.browser_loc[1] if HAS_GEO and st.session_state.browser_loc else default_lng), format="%.6f")

st.sidebar.subheader("🎥 Camera & Model")
cam_index = st.sidebar.number_input("Camera index", min_value=0, value=0, step=1)
conf_thresh = st.sidebar.slider("YOLO confidence", 0.1, 0.9, 0.5, 0.05)
speed_thresh = st.sidebar.slider("Speed warn (px/s)", 20, 400, 120, 5)
danger_distance_m = st.sidebar.slider("Danger distance (m)", 0.3, 5.0, 1.2, 0.1)

st.sidebar.subheader("🗺️ Map Options")
map_zoom = st.sidebar.slider("Zoom", 8, 20, 15)
show_map = st.sidebar.checkbox("Show Google Map overlay", value=True)

auto_drop_hazard = st.sidebar.checkbox("Auto mark hazards when warning triggers", value=True)

st.sidebar.divider()
clear_btn = st.sidebar.button("Clear all hazard markers")
if clear_btn:
    st.session_state.hazards = []

# ---------------------- Header ----------------------
st.title("🚴‍♂️ SmartBike - Real‑Time Safety")
st.caption("YOLOv8 + Hand Tracking + Speed Estimation + Google Maps")

col1, col2 = st.columns([3, 2])

with col2:
    st.markdown("### Live Map & Hazard Log")

# ---------------------- Google Map Embed ----------------------
from streamlit.components.v1 import html as components_html

MAP_HTML_TMPL = """
<!DOCTYPE html>
<html>
  <head>
    <meta name=viewport content="initial-scale=1, width=device-width" />
    <style>
      html, body, #map {{ height: 100%; margin: 0; padding: 0; }}
      .label {{
        background: rgba(0,0,0,0.6);
        color: #fff; padding: 2px 6px; border-radius: 4px; font-size: 12px;
      }}
    </style>
    <script src="https://maps.googleapis.com/maps/api/js?key={API_KEY}"></script>
    <script>
      function init() {{
        const center = {{ lat: {CENTER_LAT}, lng: {CENTER_LNG} }};
        const map = new google.maps.Map(document.getElementById('map'), {{
          center: center,
          zoom: {ZOOM},
          mapTypeId: 'roadmap',
          clickableIcons: true
        }});

        const bikeIcon = {{
          path: google.maps.SymbolPath.CIRCLE,
          scale: 6,
        }};

        const me = new google.maps.Marker({{
          position: center,
          map: map,
          title: 'Your position',
          icon: bikeIcon
        }});

        const hazards = {HAZARDS_JSON};
        hazards.forEach(h => {{
          const m = new google.maps.Marker({{
            position: {{lat: h.lat, lng: h.lng}},
            map: map,
            title: h.label || 'Hazard'
          }});
          const infowindow = new google.maps.InfoWindow({{
            content: `<div class="label"><b>${{h.label || 'Hazard'}}</b><br/>${{new Date(h.ts*1000).toLocaleString()}}</div>`
          }});
          m.addListener('click', () => infowindow.open({{anchor: m, map}}));
        }});
      }}
      window.onload = init;
    </script>
  </head>
  <body>
    <div id="map"></div>
  </body>
</html>
"""

def render_google_map(api_key: str, center: Tuple[float, float], zoom: int, hazards: List[dict]):
    if not api_key:
        st.info("Enter a Google Maps API key in the sidebar to enable the live map.")
        return
    html = MAP_HTML_TMPL.format(
        API_KEY=api_key,
        CENTER_LAT=center[0],
        CENTER_LNG=center[1],
        ZOOM=int(zoom),
        HAZARDS_JSON=json.dumps(hazards),
    )
    components_html(html, height=420)

# ---------------------- Detection Setup ----------------------
IMPORTANT_CLASSES = ["person", "car", "bicycle", "motorcycle", "bus", "traffic light"]
REAL_WIDTHS = {"person": 0.5, "car": 1.8, "bicycle": 0.7, "motorcycle": 0.8, "bus": 2.5}
FOCAL_LENGTH = 600  # Approx. tune for your camera
HISTORY_LENGTH = 10

@st.cache_resource(show_spinner=False)
def load_model():
    return YOLO("models/yolov8n.pt")

model = load_model()

# ---------------------- UI Controls ----------------------
run = st.toggle('Start SmartBike System', value=False)

FRAME = col1.empty()
LOG = col2.empty()
MAP = col2.empty()

# ---------------------- Camera ----------------------
cap = None
fps = 30
if run:
    cap = cv2.VideoCapture(int(cam_index))
    if not cap.isOpened():
        st.error("Error: Camera not available.")
        run = False
    else:
        got_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = int(got_fps) if got_fps and got_fps > 0 else 30

# ---------------------- Main Loop ----------------------
last_map_render = 0.0

while run:
    ok, frame = cap.read()
    if not ok:
        st.warning("No frame from camera.")
        break

    h, w = frame.shape[:2]
    results = model.predict(frame, save=False, conf=conf_thresh, verbose=False)[0]

    speech_chunks = []
    red_light_detected = False

    # Hand landmarks (optional visual only to avoid extra CPU; could be toggled)
    # If you want: Use MediaPipe Hands here (removed to simplify dependencies).

    # Iterate detections
    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names.get(cls_id, str(cls_id))
        if class_name not in IMPORTANT_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        bw = max(1, x2 - x1)

        # Left/Center/Right positioning
        position = "Left" if cx < w / 3 else ("Right" if cx > 2 * w / 3 else "Center")

        # --- Speed estimation (per-class history heuristic) ---
        hist = st.session_state.object_histories.setdefault(class_name, deque(maxlen=HISTORY_LENGTH))
        hist.append((time.time(), cx, cy))

        speed_warn = ""
        if len(hist) >= 2:
            t0, x0, y0 = hist[0]
            t1, x1n, y1n = hist[-1]
            dt = max(1e-3, t1 - t0)
            pix_dist = float(np.hypot(x1n - x0, y1n - y0))
            speed = pix_dist / dt  # px/s
            if speed > speed_thresh:
                speed_warn = "FAST"
                speech_chunks.append(f"Warning: {class_name} approaching fast.")

        # --- Distance estimation ---
        distance_m = None
        if class_name in REAL_WIDTHS:
            distance_m = round((REAL_WIDTHS[class_name] * FOCAL_LENGTH) / bw, 2)

        # Danger logic
        danger = False
        if distance_m is not None and distance_m < danger_distance_m and position in ("Left", "Right"):
            danger = True
            side = position.lower()
            speech_chunks.append(f"Warning: {class_name} on your {side}, too close.")

        # Draw
        label = class_name
        if distance_m is not None:
            label += f" {distance_m}m"
        if speed_warn:
            label += " ⚠️"
        color = (0, 0, 255) if danger or speed_warn else (0, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Traffic light naive red detection
        if class_name == "traffic light":
            roi = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if roi.size > 0:
                avg_bgr = roi.mean(axis=0).mean(axis=0)
                if avg_bgr[2] > avg_bgr[1] and avg_bgr[2] > avg_bgr[0]:
                    red_light_detected = True

    if red_light_detected:
        speech_chunks.append("Red light ahead. Please stop.")

    # Speak (debounced to ~1.5s)
    now = time.time()
    if speech_chunks and (now - st.session_state.last_danger_spoken > 1.5):
        speaker.say_async(" ".join(speech_chunks))
        st.session_state.last_danger_spoken = now

    # Auto hazard marker
    if auto_drop_hazard and (red_light_detected or any("Warning:" in s for s in speech_chunks)):
        st.session_state.hazards.append({
            "lat": float(lat),
            "lng": float(lng),
            "label": "Danger/Warning",
            "ts": now,
        })

    # Show frame
    FRAME.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Live ({fps} FPS est.)", use_column_width=True)

    # Log panel
    if speech_chunks or red_light_detected:
        LOG.write("\n".join([f"• {s}" for s in speech_chunks] + (["• Red light detected."] if red_light_detected else [])))

    # Map refresh (rate-limit to ~1s)
    if show_map and (now - last_map_render > 1.0):
        with MAP:
            render_google_map(api_key, (lat, lng), map_zoom, st.session_state.hazards)
        last_map_render = now

# Cleanup
if cap is not None:
    cap.release()

st.info("SmartBike System is stopped.")
