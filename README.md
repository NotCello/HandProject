# Two-Hand Finger Counter (MediaPipe + OpenCV)

**A Python project for real-time finger counting on both hands using OpenCV and MediaPipe.**
The code detects both hands, assigns handedness (“Left” or “Right”), counts extended fingers (with proper thumb logic), and stabilizes results using smoothing and debouncing. Includes FPS overlay and low-latency camera settings.

---

## ✨ Features

- **Real-time finger counting** for both hands via webcam
- **Handedness recognition**: distinguishes between left and right hands
- **Robust algorithm**: smoothing (majority over recent frames) and debouncing (updates only after N stable frames)
- **Live overlays**: FPS and “Left/Right” counters
- **Performance optimized**: low-latency capture, minimal buffering, MJPG compression

---

## 🚀 Quick Start

1. **Create and activate a virtual environment:**

   **Linux/macOS:**
   python3 -m venv .venv source .venv/bin/activate

**Windows:**
    py -m venv .venv .venv\Scripts\activate

2. **Install dependencies:**
     pip install -r requirements.txt
3. **Run the script:**
   python finger_count_bihand.py

4. **Controls:** Press **ESC** in the window to quit.

---

## 📝 requirements.txt
opencv-python>=4.8 mediapipe>=0.10 numpy>=1.22

---

## ⚙️ Configuration

Edit values at the top of `finger_count_bihand.py`:
- `WIDTH`, `HEIGHT`, `REQ_FPS`: camera resolution and FPS
- `MODEL_COMPLEXITY`: 0 (fastest), 1/2 (higher accuracy, lower FPS)
- `DRAW_LANDMARKS`: `True`/`False` to show/hide hand skeleton drawing
- `SMOOTH_N`: smoothing window (higher is more stable, but slower reaction)
- `DEBOUNCE_N`: show new value only after this many stable frames

**Make sure `max_num_hands=2`:**
python
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # allow detection of both hands
    model_complexity=MODEL_COMPLEXITY,
    min_detection_confidence=MIN_DET_CONF,
    min_tracking_confidence=MIN_TRK_CONF
)

🕹 ** How It Works**
Capture and preprocess

Grabs webcam frames with OpenCV
Flips horizontally for “selfie” view (cv2.flip(frame, 1))
Converts to RGB for MediaPipe processing
Hand and landmark detection

MediaPipe Hands returns for each detected hand:
21 landmarks
Handedness label ("Left" or "Right") with confidence
Loop example:

for lm, hh in zip(res.multi_hand_landmarks, res.multi_handedness):
    label = hh.classification[0].label
    score = hh.classification[0].score

    
Finger counting

Index/Middle/Ring/Pinky: “open” if tip.y < pip.y
Thumb: lateral logic based on handedness
Right hand: extended if tip.x < ip.x
Left hand: extended if tip.x > ip.x
Smoothing and debouncing

Circular buffer stores recent raw counts per hand
Majority (mode) over buffer smooths noise
Updates displayed value only after N stable frames
Overlay

Displays “Left: …”, “Right: …”, and “FPS: …” on video
Optional: draws landmarks/skeleton
For adding per-hand label overlay:

wx, wy = int(lm.landmark[0].x * w), int(lm.landmark[0].y * h)
color = (255, 0, 255) if label == "Left" else (0, 255, 255)
cv2.putText(frame, f"{label} {score:.2f}", (wx + 10, wy - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
🛠 Troubleshooting
Webcam won’t open: Close other apps using the webcam, try cv2.VideoCapture(1) (or 2), or remove the V4L2 backend for non-Linux.
Low FPS: Lower resolution, set MODEL_COMPLEXITY=0, set DRAW_LANDMARKS = False.
Flickering after occlusion: Increase SMOOTH_N or DEBOUNCE_N; filter low-handedness confidence.
Left/Right swapped: The view is mirrored, so labels fit visual orientation. For “physical” handedness, process the frame unflipped or adjust label before display.
Only one hand detected: Make sure max_num_hands = 2 in code.
📁 Suggested Folder Structure
.
├─ finger_count_bihand.py
├─ requirements.txt
├─ README.md
📜 License
MIT (or your preferred license)

🙏 Credits
MediaPipe Hands (Google)
OpenCV community
