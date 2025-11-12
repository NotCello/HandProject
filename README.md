# Single-Hand Finger Counter (MediaPipe + OpenCV)

**A Python project for real-time single-hand finger counting using a webcam, OpenCV, and MediaPipe.**

This script detects one hand, counts the extended fingers (with proper thumb logic), and displays the count and FPS on a mirrored (selfie-view) video.

---

## Features

* **Real-time counting** for a **single hand** via webcam.
* **Orientation-aware thumb logic** (distinguishes right/left for correct thumb counting).
* **Selfie Mode** (`cv2.flip`): Video is mirrored, but the text remains upright.
* **Live Overlays**: Shows the **total finger count** and real-time **FPS**.

---

##  Quick Start

1.  **Create and activate a virtual environment:**

    **Linux/macOS:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

    **Windows:**
    ```bash
    py -m venv .venv
    .venv\Scripts\activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the script:**
    ```bash
    python finger_counter_flipped.py
    ```
4.  **Controls:** Press **'q'** in the window to quit.

---

##  requirements.txt
opencv-python mediapipe

---

## Configuration

You can modify the MediaPipe parameters directly in the code:

```python
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Ensures only one hand is detected
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
```
 How It Works
1. Capture and Preprocessing
```
Grabs frames from the webcam (cv2.VideoCapture(0)).

Converts the frame to RGB for MediaPipe (cv2.cvtColor).```
```
```
2. Hand and Landmark Detection
hands.process(frame_rgb) detects the hand and calculates the 21 landmarks.

It also returns the "Left" or "Right" label, which is crucial for the thumb logic.
```
3. Finger Counting
```
Index, Middle, Ring, Pinky: Considered "open" if their tip Y-coordinate is less than their lower joint (pip).

Thumb: Has lateral logic based on the hand label (to work correctly in the mirrored view):

Right hand: Extended if the tip's X is less than the joint's X.

Left hand: Extended if the tip's X is greater than the joint's X.
```
4. Overlay and Display
```
The entire video frame is flipped (cv2.flip(frame, 1)) to create a mirror effect.

After flipping, the text overlays (count and FPS) are drawn so they appear upright and un-mirrored.

A black rectangle is drawn to improve the count's readability.
```
--- 
Troubleshooting
Webcam won’t open:

Ensure no other program is using the webcam.

Try changing the index to cv2.VideoCapture(1) (or 2).

Low FPS:

Increase the min_detection_confidence values (e.g., 0.7) to filter weak detections and reduce workload.

Ensure you have good lighting.

Folder Structure
.
├── finger_counter_flipped.py
├── requirements.txt
└── README.md
License
MIT

Credits
MediaPipe Hands (Google)

OpenCV Community

