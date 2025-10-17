import cv2, time
import mediapipe as mp

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 424)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cv2.setNumThreads(1)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       model_complexity=0, min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

last = time.time(); cnt = 0; fps = 0.0
while True:
    ok, frame = cap.read()
    if not ok: continue
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    detected = bool(res.multi_hand_landmarks)

    cnt += 1
    if cnt % 15 == 0:
        now = time.time()
        fps = 15.0 / max(1e-6, (now - last))
        last = now

    cv2.putText(frame, f"DETECTED: {detected}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.imshow("Hands quick test", frame)
    if (cv2.waitKey(1) & 0xFF) == 27: break

cap.release(); cv2.destroyAllWindows()
