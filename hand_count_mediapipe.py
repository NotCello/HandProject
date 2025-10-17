# finger_count_bihand.py

import cv2                 # OpenCV for camera I/O, image processing, drawing
import time                # For timing and FPS estimation
from collections import deque  # Deque used as a fixed-size buffer for smoothing/debouncing
import mediapipe as mp     # MediaPipe for real-time hand detection/landmarks

# ---------- Config ----------
WIDTH, HEIGHT, REQ_FPS = 640, 480, 30  # Desired capture resolution and frame rate
FOURCC = "MJPG"                        # MJPG compressed stream (reduces USB bandwidth, often higher FPS)
MODEL_COMPLEXITY = 0                   # MediaPipe Hands model complexity (0 = fastest, 1/2 = more accurate, slower)
DRAW_LANDMARKS = True                  # Toggle drawing 21 landmarks and connections for visualization
MIN_DET_CONF = 0.6                     # Minimum detection confidence to start tracking a hand
MIN_TRK_CONF = 0.6                     # Minimum tracking confidence to keep tracking once detected
SMOOTH_N   = 7                         # Majority-vote window size for smoothing finger counts
DEBOUNCE_N = 3                         # Frames a new value must persist before updating "shown" value
# ----------------------------

def count_fingers(landmarks, handedness, w, h):
    """
    landmarks: list of 21 normalized landmarks (x,y in [0,1])
    handedness: "Left" or "Right" (used for thumb logic)
    w, h: frame width/height for converting normalized to pixel coordinates
    """

    # Convert normalized landmark coordinates to integer pixel coordinates
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    # Indices for fingertips (index, middle, ring, pinky) in MediaPipe hand model
    tips = [8, 12, 16, 20]
    # Indices for corresponding PIP joints (one joint below each tip)
    pips = [6, 10, 14, 18]

    fingers = 0  # Counter for extended fingers

    # For the four fingers (not thumb): "extended" if tip y is above PIP y (remember y grows downward in images)
    for tip, pip in zip(tips, pips):
        if pts[tip][1] < pts[pip][1]:
            fingers += 1

    # Thumb uses lateral check (x-direction), and needs handedness to know which way "outward" points
    thumb_tip, thumb_ip = 4, 3

    if handedness == "Right":
        # In a mirrored (flipped) view, right-hand thumb tip appears leftward when extended
        if pts[thumb_tip][0] < pts[thumb_ip][0]:
            fingers += 1
    else:  # "Left"
        # In mirrored view, left-hand thumb tip appears rightward when extended
        if pts[thumb_tip][0] > pts[thumb_ip][0]:
            fingers += 1

    # Clamp to 5 just in case (robustness)
    return min(fingers, 5)

def main():

    # Camera
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)       # Open default camera using V4L2 backend (Linux)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)      # Request width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)    # Request height
    cap.set(cv2.CAP_PROP_FPS, REQ_FPS)            # Request FPS (driver may not guarantee)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))  # Set codec to MJPG
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)           # Small buffer to reduce latency (drop late frames)
    cv2.setNumThreads(1)                           # Limit OpenCV CPU threads to reduce scheduling overhead

    if not cap.isOpened():                         # Verify the camera opened successfully
        print("ERROR: cannot open webcam")
        return

    # MediaPipe Hands setup
    mp_hands = mp.solutions.hands                 # Namespace for the Hands solution
    mp_draw  = mp.solutions.drawing_utils         # Drawing utilities (landmarks, connections)

    hands = mp_hands.Hands(
        static_image_mode=False,                  # Video stream mode (tracking after first detection)
        max_num_hands=2,                          # Allow up to two hands (your original comment said two)
        model_complexity=MODEL_COMPLEXITY,        # 0,1,2 trade speed/accuracy
        min_detection_confidence=MIN_DET_CONF,    # Threshold to start detecting a hand
        min_tracking_confidence=MIN_TRK_CONF      # Threshold to keep tracking an existing hand
    )

    # Per-hand smoothing/debounce state
    state = {
        # For each hand label, keep:
        # - buf: recent raw finger counts (for majority smoothing)
        # - shown: the current displayed count after debounce
        # - maj: current majority count within the buffer
        # - stable: how many consecutive frames the majority has stayed the same
        "Left":  {"buf": deque(maxlen=SMOOTH_N), "shown": None, "maj": None, "stable": 0},
        "Right": {"buf": deque(maxlen=SMOOTH_N), "shown": None, "maj": None, "stable": 0},
    }

    # FPS estimator (update every N frames)
    last = time.time()    # Last timestamp when FPS was updated
    cnt  = 0              # Frame counter since last update
    fps  = 0.0            # Smoothed FPS display value

    try:
        while True:                               # Main processing loop
            ok, frame = cap.read()                # Grab a frame from the camera
            if not ok:                            # If read fails, skip this iteration
                continue

            frame = cv2.flip(frame, 1)            # Mirror the frame horizontally (user-friendly "selfie" view)
            h, w = frame.shape[:2]                # Current frame height and width

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # MediaPipe expects RGB input
            res = hands.process(rgb)                        # Run hand detection + landmark + handedness

            # Track which hands are seen this frame (for optional buffer resets)
            seen = {"Left": False, "Right": False}

            # If any hands are detected and handedness info is available
            if res.multi_hand_landmarks and res.multi_handedness:

                # Pair the i-th landmarks with the i-th handedness (aligned lists)
                for lm, hh in zip(res.multi_hand_landmarks, res.multi_handedness):

                    label = hh.classification[0].label  # "Left" or "Right" as predicted by MediaPipe

                    if label not in state:              # Safety: if an unexpected label appears, initialize state
                        state[label] = {"buf": deque(maxlen=SMOOTH_N), "shown": None, "maj": None, "stable": 0}

                    seen[label] = True                  # Mark this hand label as present this frame

                    # Count raw fingers for this hand using landmarks, handedness, and image size
                    fingers_raw = count_fingers(lm.landmark, label, w, h)

                    # Draw landmarks (optional visualization)
                    if DRAW_LANDMARKS:
                        # FIX: use mp_hands.HAND_CONNECTIONS (your snippet had a typo)
                        mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

                    # Smoothing + Debounce for this hand
                    st = state[label]                   # Shorthand to this hand's state dict

                    if st["shown"] is None:            # Initialize "shown" on the very first observation
                        st["shown"] = fingers_raw

                    st["buf"].append(fingers_raw)      # Append the latest raw count to the circular buffer

                    # Compute the majority value within the buffer (mode); fallback to current raw if empty
                    maj = max(set(st["buf"]), key=st["buf"].count) if st["buf"] else fingers_raw

                    # If the majority changed, reset the stability counter; else increment it
                    if (st["maj"] is None) or (maj != st["maj"]):
                        st["maj"] = maj
                        st["stable"] = 1
                    else:
                        st["stable"] += 1

                    # Only update the displayed value if the new majority stayed for DEBOUNCE_N frames
                    if (maj != st["shown"]) and (st["stable"] >= DEBOUNCE_N):
                        st["shown"] = maj

            # Optional: if a hand is not seen this frame, you can reset its buffers to avoid stale influence
            # for k in ["Left", "Right"]:
            #     if not seen[k]:
            #         state[k]["buf"].clear()
            #         state[k]["maj"] = None
            #         state[k]["stable"] = 0
            #         # Keep "shown" as last good value so the on-screen number doesn't jump to "-"

            # FPS update logic
            cnt += 1                                 # Count this frame
            if cnt % 15 == 0:                        # Every 15 frames, recompute FPS
                now = time.time()
                fps = 15.0 / max(1e-6, (now - last))  # Frames per elapsed seconds (avoid division by zero)
                last = now

            # Overlays: show per-hand counts and FPS in the top-left region
            left_val  = state["Left"]["shown"]  if state["Left"]["shown"]  is not None else "-"
            right_val = state["Right"]["shown"] if state["Right"]["shown"] is not None else "-"
            cv2.putText(frame, f"Left: {left_val}",   (20,  40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Right: {right_val}", (20,  80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"FPS: {fps:.1f}",     (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),   2, cv2.LINE_AA)

            cv2.imshow("Two-Hand Finger Count (ESC to quit)", frame)  # Show the annotated frame

            if (cv2.waitKey(1) & 0xFF) == 27:        # Exit when ESC key is pressed
                break

    finally:
        cap.release()               # Always release the camera
        cv2.destroyAllWindows()     # Close any OpenCV windows

if __name__ == "__main__":          # Standard Python entry-point guard
    main()                          # Run the main function
