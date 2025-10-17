# finger_count_bihand.py
import cv2
import time
from collections import deque
import mediapipe as mp

# ---------- Config ----------
WIDTH, HEIGHT, REQ_FPS = 640, 480, 30
FOURCC = "MJPG"
MODEL_COMPLEXITY = 0    # 0 = fastest
DRAW_LANDMARKS = True   # set False to gain FPS if needed
MIN_DET_CONF = 0.6
MIN_TRK_CONF = 0.6

SMOOTH_N   = 7   # majority window size
DEBOUNCE_N = 3   # frames required before updating
# ----------------------------

def count_fingers(landmarks, handedness, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    fingers = 0
    for tip, pip in zip(tips, pips):
        if pts[tip][1] < pts[pip][1]:
            fingers += 1
    thumb_tip, thumb_ip = 4, 3
    if handedness == "Right":
        if pts[thumb_tip][0] < pts[thumb_ip][0]:
            fingers += 1
    else:  # "Left"
        if pts[thumb_tip][0] > pts[thumb_ip][0]:
            fingers += 1
    return min(fingers, 5)

def main():
    # Camera
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, REQ_FPS)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cv2.setNumThreads(1)
    if not cap.isOpened():
        print("ERROR: cannot open webcam")
        return

    # MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,                 # enable two hands
        model_complexity=MODEL_COMPLEXITY,
        min_detection_confidence=MIN_DET_CONF,
        min_tracking_confidence=MIN_TRK_CONF
    )

    # Per-hand smoothing/debounce state
    state = {
        "Left":  {"buf": deque(maxlen=SMOOTH_N), "shown": None, "maj": None, "stable": 0},
        "Right": {"buf": deque(maxlen=SMOOTH_N), "shown": None, "maj": None, "stable": 0},
    }

    # FPS estimator
    last = time.time()
    cnt  = 0
    fps  = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            # Track which hands are seen this frame
            seen = {"Left": False, "Right": False}

            if res.multi_hand_landmarks and res.multi_handedness:
                for lm, hh in zip(res.multi_hand_landmarks, res.multi_handedness):
                    label = hh.classification[0].label  # "Left" or "Right"
                    if label not in state:
                        # Initialize if a new label appears (rare)
                        state[label] = {"buf": deque(maxlen=SMOOTH_N), "shown": None, "maj": None, "stable": 0}
                    seen[label] = True

                    fingers_raw = count_fingers(lm.landmark, label, w, h)

                    # Draw landmarks (optional)
                    if DRAW_LANDMARKS:
                        mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

                    # Smoothing + Debounce for this hand
                    st = state[label]
                    if st["shown"] is None:
                        st["shown"] = fingers_raw
                    st["buf"].append(fingers_raw)
                    maj = max(set(st["buf"]), key=st["buf"].count) if st["buf"] else fingers_raw
                    if (st["maj"] is None) or (maj != st["maj"]):
                        st["maj"] = maj
                        st["stable"] = 1
                    else:
                        st["stable"] += 1
                    if (maj != st["shown"]) and (st["stable"] >= DEBOUNCE_N):
                        st["shown"] = maj

            # Optional: if a hand is not seen, you can reset its buffers
            # for k in ["Left", "Right"]:
            #     if not seen[k]:
            #         state[k]["buf"].clear()
            #         state[k]["maj"] = None
            #         state[k]["stable"] = 0
            #         # keep shown as last good value so UI doesn't jump

            # FPS
            cnt += 1
            if cnt % 15 == 0:
                now = time.time()
                fps = 15.0 / max(1e-6, (now - last))
                last = now

            # Overlays
            left_val  = state["Left"]["shown"]  if state["Left"]["shown"]  is not None else "-"
            right_val = state["Right"]["shown"] if state["Right"]["shown"] is not None else "-"
            cv2.putText(frame, f"Left: {left_val}",  (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Right: {right_val}",(20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"FPS: {fps:.1f}",    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)

            cv2.imshow("Two-Hand Finger Count (ESC to quit)", frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
