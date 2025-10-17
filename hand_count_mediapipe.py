# finger_count_basic.py
import cv2, time
import mediapipe as mp
from collecions import deque

def count_fingers(landmarks, handedness, w, h):
    # Convert normalized coords -> pixels
    pts = [(int(lm.x*w), int(lm.y*h)) for lm in landmarks]

    # Non-thumb fingers: tip above PIP
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    fingers = 0
    for tip, pip in zip(tips, pips):
        if pts[tip][1] < pts[pip][1]:
            fingers += 1

    # Thumb uses x (image is mirrored)
    thumb_tip, thumb_ip = 4, 3
    if handedness == "Right":
        if pts[thumb_tip][0] < pts[thumb_ip][0]:
            fingers += 1
    else:  # "Left"
        if pts[thumb_tip][0] > pts[thumb_ip][0]:
            fingers += 1

    return min(fingers, 5)

def main():
    # Camera config: same as when it ti funzionava
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cv2.setNumThreads(1)

    if not cap.isOpened():
        print("ERROR: cannot open webcam"); return

    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=1,
                           min_detection_confidence=0.6,
                           min_tracking_confidence=0.6)  # questi valori ti andavano bene

    last = time.time(); cnt = 0; fps = 0.0
    # Smoothing (majority over last N) + Debounce (require N_stable frames)
    SMOOTH_N = 7        # rolling window size
    DEBOUNCE_N = 3      # require 3 consecutive frames of same majority
    buf = deque(maxlen=SMOOTH_N)
    shown = None        # value we display (starts as None, set on first frame)
    stable_for = 0      # consecutive frames matching 'shown'



    while True:
        ok, frame = cap.read()
        if not ok: continue

        # Specchio per usabilità
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        fingers = 0
        if res.multi_hand_landmarks and res.multi_handedness:
            lm = res.multi_hand_landmarks[0]
            label = res.multi_handedness[0].classification[0].label  # "Left" or "Right"
            fingers = count_fingers(lm.landmark, label, w, h)
	    # Smoothing + Debounce
	    if shown is None:
	    shown = fingers  # seed on first valid frame

	    buf.append(fingers)
	    # majority vote over the buffer
	    maj = max(set(buf), key=buf.count) if buf else fingers

# debounce: only update 'shown' after DEBOUNCE_N stable frames
	    if maj == shown:
    	    	stable_for += 1
	    else:
	    	stable_for = 0

	    if stable_for >= DEBOUNCE_N:
	    	shown = maj

	    display_val = shown
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        # FPS semplice (rolling su 15 frame)
        cnt += 1
        if cnt % 15 == 0:
            now = time.time()
            fps = 15.0 / max(1e-6, (now - last))
            last = now

        # Overlay
	cv2.putText(frame, f"Fingers: {display_val}", (20,40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 2, cv2.LINE_AA)
#        cv2.putText(frame, f"Fingers: {fingers}", (20,40),
 #                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)

        cv2.imshow("MediaPipe Finger Count (ESC to quit)", frame)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
