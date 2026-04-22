import cv2
import numpy as np
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

mp_draw = mp.solutions.drawing_utils

canvas = np.zeros((480, 640, 3), dtype=np.uint8)

prev_x, prev_y = 0, 0
prev_ex, prev_ey = 0, 0

def fingers_up(lm):
    fingers = []

    # Thumb (simple check)
    fingers.append(lm[4][1] > lm[3][1])

    # Index
    fingers.append(lm[8][2] < lm[6][2])

    # Middle
    fingers.append(lm[12][2] < lm[10][2])

    # Ring
    fingers.append(lm[16][2] < lm[14][2])

    # Pinky
    fingers.append(lm[20][2] < lm[18][2])

    return fingers

while True:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    mode = "IDLE"

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            h, w, _ = img.shape
            lm = []

            for id, pt in enumerate(handLms.landmark):
                lm.append((id, int(pt.x * w), int(pt.y * h)))

            fingers = fingers_up(lm)

            index = lm[8][1], lm[8][2]
            middle = lm[12][1], lm[12][2]

            # ----------------------
            # ERASE MODE (Index + Middle up)
            # ----------------------
            if fingers[1] and fingers[2]:
                mode = "ERASE"

                cv2.circle(canvas, index, 25, (0, 0, 0), -1)

                if prev_ex == 0:
                    prev_ex, prev_ey = index

                cv2.line(canvas, (prev_ex, prev_ey), index, (0, 0, 0), 30)
                prev_ex, prev_ey = index

                prev_x, prev_y = 0, 0

            # ----------------------
            # DRAW MODE (Only Index up)
            # ----------------------
            elif fingers[1] and not fingers[2]:
                mode = "DRAW"

                if prev_x == 0:
                    prev_x, prev_y = index

                cv2.line(canvas, (prev_x, prev_y), index, (0, 255, 255), 5)
                cv2.line(canvas, (prev_x, prev_y), index, (255, 255, 255), 2)

                prev_x, prev_y = index

            else:
                prev_x, prev_y = 0, 0
                prev_ex, prev_ey = 0, 0

    output = cv2.add(img, canvas)

    cv2.putText(output, f"MODE: {mode}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("🔥 High Accuracy Air Drawing", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()