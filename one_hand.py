#!/usr/bin/env python3
"""
finger_counter_flipped.py
Counts raised fingers in real time using the laptop webcam (Ubuntu 24).
Mirror-like video, but text (count & FPS) not flipped.
"""

import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]


def count_fingers(hand_landmarks, handedness_label, img_w, img_h):
    fingers_up = []
    lm = [(int(l.x * img_w), int(l.y * img_h)) for l in hand_landmarks.landmark]

    # thumb
    thumb_tip_x = lm[4][0]
    thumb_ip_x = lm[3][0]
    if handedness_label == "Right":
        thumb_is_up = thumb_tip_x < thumb_ip_x
    else:
        thumb_is_up = thumb_tip_x > thumb_ip_x
    fingers_up.append(thumb_is_up)

    # other fingers
    for tip_idx, pip_idx in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        tip_y = lm[tip_idx][1]
        pip_y = lm[pip_idx][1]
        fingers_up.append(tip_y < pip_y)

    return sum(fingers_up)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: impossible to open the webcam.")
        return

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_h, img_w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        finger_count = 0

        if results.multi_hand_landmarks and results.multi_handedness:
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0].classification[0].label
            finger_count = count_fingers(hand_landmarks, handedness, img_w, img_h)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Flip frame horizontally for natural mirror effect
        flipped = cv2.flip(frame, 1)

        # --- draw text after flipping ---
        # solid black background for visibility
        cv2.rectangle(flipped, (10, 10), (190, 70), (0, 0, 0), -1)
        cv2.putText(flipped, f"FINGERS: {finger_count}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(flipped, f"FPS: {int(fps)}", (img_w - 140, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Finger Counter - press 'q' to quit", flipped)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
