import cv2, time
import mediapipe as mp
from collections import deque
import time, csv, os, numpy as np

buf = deque(maxlen=7)      # rolling votes
shown = 0                  # number displayed
stable_for = 0             # consecutive frames with same majority

last_t = time.perf_counter()
frame_ms = []
log_path = "finger_metrics.csv"
csv_f = open(log_path, "w", newline=""); w = csv.writer(csv_f)
w.writerow(["t_sec","fps","frame_ms","fingers_raw","fingers_shown"])
fps_est = 0.0

def count_fingers(landmarks, handedness, w, h):
    # landmarks: 21 punti in coordinate normalizzate [0..1]
    # Converti in pixel
    pts = [(int(lm.x*w), int(lm.y*h)) for lm in landmarks]
    # Indici utili (MediaPipe):
    # Pollice: 4 (tip), 3 (IP)
    # Indice Medio Anulare Mignolo: tip = [8,12,16,20]; PIP = [6,10,14,18]
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]

    fingers = 0
    # Dita (escl. pollice): dito “aperto” se tip è più in alto (y più piccolo) del PIP
    for tip, pip in zip(tips, pips):
        if pts[tip][1] < pts[pip][1]:
            fingers += 1

    # Pollice: usa asse x (dipende da mano sinistra/destra e dal mirroring)
    # Nota: stiamo FLIPPANDO l’immagine, quindi dx/sx risultano invertiti.
    # Per semplicità: se è Right, consideriamo "aperto" se tip.x < ip.x (perché immagine specchiata)
    # Se è Left, l'opposto.
    thumb_tip, thumb_ip = 4, 3
    if handedness == "Right":
        if pts[thumb_tip][0] < pts[thumb_ip][0]:
            fingers += 1
    else:  # Left
        if pts[thumb_tip][0] > pts[thumb_ip][0]:
            fingers += 1

    return min(fingers, 5)

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cv2.setNumThreads(1)

    if not cap.isOpened():
        print("ERROR: webcam"); return

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=1,
                           min_detection_confidence=0.6,
                           min_tracking_confidence=0.6)

    last = time.time(); cnt = 0; fps = 0.0
    while True:
        ok, frame = cap.read()
        if not ok: continue
        # Flip per effetto specchio
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        fingers = 0
        if res.multi_hand_landmarks and res.multi_handedness:
            lm = res.multi_hand_landmarks[0]
            label = res.multi_handedness[0].classification[0].label  # "Left" o "Right"
            fingers = count_fingers(lm.landmark, label, w, h)
            # smoothing + debounce
	    buf.append(fingers)
	    maj = max(set(buf), key=buf.count) if buf else fingers
	    stable_for = stable_for + 1 if maj == shown else 0
	    if stable_for >= 3:  # require 3 stable frames before updating
		    shown = maj

	   # timing + FPS + log
	   now = time.perf_counter()
           dt_ms = (now - last_t)*1000.0
	   last_t = now
   	   frame_ms.append(dt_ms)
	   if len(frame_ms) >= 15:
	   	fps_est = 1000.0 / (sum(frame_ms[-15:]) / 15.0)
	   w.writerow([f"{time.time():.3f}", f"{fps_est:.2f}", f"{dt_ms:.2f}", int(fingers), int(shown)])

	   # overlays
	   cv2.putText(frame, f"Fingers: {shown}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 2)
	   cv2.putText(frame, f"FPS: {fps_est:.1f}", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        # FPS semplice
        cnt += 1
        if cnt % 15 == 0:
            now = time.time()
            fps = 15.0 / max(1e-6, (now - last))
            last = now

        cv2.putText(frame, f"Fingers: {fingers}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("MediaPipe Finger Count (ESC to quit)", frame)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
csv_f.close()
if frame_ms:
    a = np.array(frame_ms, dtype=np.float64)
    p50, p95, p99 = np.percentile(a, [50,95,99])
    print("\n=== RUN SUMMARY ===")
    print(f"Frames: {len(a)} | Avg FPS~{(1000.0/np.mean(a)):.1f}")
    print(f"Frame ms p50/p95/p99: {p50:.1f}/{p95:.1f}/{p99:.1f}")
    print(f"CSV saved: {os.path.abspath(log_path)}")

if __name__ == "__main__":
    main()
