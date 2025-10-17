Two‑Hand Finger Counter (MediaPipe + OpenCV)
Low‑latency, two‑hand finger counter using your webcam. Detects left vs right hand, counts extended fingers (with correct thumb logic), and stabilizes outputs with smoothing + debouncing. Includes FPS overlay and low‑latency camera settings.

Use it to learn real‑time vision fundamentals and to showcase problem‑solving and performance instrumentation.

Demo
Add your live recording here (replace with your files):
MP4: assets/demo.mp4
GIF preview:
Demo
Tip to convert MP4 → GIF:

ffmpeg -i assets/demo.mp4 -vf "fps=12,scale=640:-1:flags=lanczos" -loop 0 assets/demo.gif
Features
Detects up to two hands with handedness (“Left” / “Right”)
Counts fingers:
Index/Middle/Ring/Pinky: vertical test (tip above PIP)
Thumb: lateral, handedness‑aware test (different rule for left vs right)
Per‑hand smoothing (majority vote) + debounce (stable N frames before update)
Mirrored “selfie” view and FPS overlay
Low‑latency capture (MJPG, tiny buffer, single OpenCV thread)
Quick Start
Create a virtual environment
Linux/macOS:
python3 -m venv .venv && source .venv/bin/activate
Windows:
py -m venv .venv && .venv\Scripts\activate
Install dependencies
pip install -r requirements.txt
Run
python finger_count_bihand.py
Controls
Press ESC to quit
Requirements
Python 3.9–3.12
A webcam
OS: Linux, macOS, or Windows
Create requirements.txt:

opencv-python>=4.8
mediapipe>=0.10
numpy>=1.22
Configuration
Edit constants at the top of finger_count_bihand.py:

WIDTH, HEIGHT, REQ_FPS: requested capture settings
FOURCC = "MJPG": compressed stream for higher USB throughput
MODEL_COMPLEXITY = 0: 0 fastest; 1/2 more accurate but slower
DRAW_LANDMARKS = True: set False for a small FPS gain
MIN_DET_CONF, MIN_TRK_CONF: MediaPipe thresholds
SMOOTH_N: window for majority‑vote smoothing
DEBOUNCE_N: frames a new majority must persist before updating
Ensure two hands are allowed:

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,                 # allow up to two hands
    model_complexity=MODEL_COMPLEXITY,
    min_detection_confidence=MIN_DET_CONF,
    min_tracking_confidence=MIN_TRK_CONF
)
How It Works
Capture and pre‑process
OpenCV captures frames with low latency:
MJPG codec, CAP_PROP_BUFFERSIZE=1 (drop stale frames), cv2.setNumThreads(1)
Frame is mirrored (cv2.flip(frame, 1)) for a natural “selfie” view
Convert BGR→RGB for MediaPipe
MediaPipe Hands
Palm detector proposes hand ROIs; landmark model returns 21 keypoints per hand
Handedness (Left/Right) is predicted per hand with a confidence score
Access via:
for lm, hh in zip(res.multi_hand_landmarks, res.multi_handedness):
    label = hh.classification[0].label   # "Left" or "Right"
    score = hh.classification[0].score   # confidence 0..1
Finger counting logic
Convert normalized landmarks → pixel coordinates
Index/Middle/Ring/Pinky extended if tip.y < pip.y (y grows downward)
Thumb uses handedness (lateral test):
Right hand: extended if thumb_tip.x < thumb_ip.x (in mirrored view)
Left hand: extended if thumb_tip.x > thumb_ip.x
Smoothing + Debounce
Per hand, keep a circular buffer of recent raw counts
Majority (mode) over the buffer → stabilized value
Debounce updates “shown” only after the new majority persists DEBOUNCE_N frames
Display
Overlays “Left: …”, “Right: …”, and “FPS: …”
Optionally draws landmarks and connections:
mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
Handedness Notes
Because the frame is mirrored before inference, the label (“Left” / “Right”) matches what you see on screen.
If you want physical handedness without mirroring:
Process the unflipped frame, or flip only for display and adjust x‑coordinates accordingly.
Use score to filter very low‑confidence handedness if needed.
Suggested Repo Structure
.
├─ finger_count_bihand.py
├─ requirements.txt
├─ assets/
│  ├─ demo.mp4
│  └─ demo.gif
└─ README.md
Troubleshooting
Webcam won’t open:
Close other apps using the camera
Try a different index: cv2.VideoCapture(1) or 2
On non‑Linux, remove the V4L2 backend: cv2.VideoCapture(0) (no second arg)
Low FPS:
Use 640×480 or 320×240
MODEL_COMPLEXITY = 0
DRAW_LANDMARKS = False
Keep MJPG; if unsupported, try removing FOURCC line
Left/Right looks swapped:
This is expected in mirrored view; for physical handedness, don’t flip before inference or swap labels only for display
Flicker under occlusion:
Increase SMOOTH_N or DEBOUNCE_N
Ignore frames with handedness score < 0.5
Performance Tips
Camera I/O:
MJPG often stabilizes 30 FPS over USB
CAP_PROP_BUFFERSIZE=1 to minimize latency
CPU:
cv2.setNumThreads(1) to reduce scheduling overhead on small pipelines
MediaPipe:
static_image_mode=False to track between detections
Tune MODEL_COMPLEXITY only if accuracy requires it
Extensions
Draw per‑hand overlay near the wrist with label + confidence
Log CSV of timestamp, left/right counts, FPS
Unit tests that feed synthetic landmarks to validate counting logic
CLI flags for all config options
HALCON port for industrial‑style gauging or shape‑based matching (talking point for interviews)
Example per‑hand label overlay:

wx, wy = int(lm.landmark[0].x * w), int(lm.landmark[0].y * h)
color = (255,0,255) if label=="Left" else (0,255,255)
cv2.putText(frame, f"{label} {score:.2f}", (wx+10, wy-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
License
MIT (or choose your preferred license).

Acknowledgments
MediaPipe Hands (Google)
OpenCV community
