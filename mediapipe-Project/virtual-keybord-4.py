import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Define Keyboard Layout
keyboard_keys = [
    "Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P",
    "A", "S", "D", "F", "G", "H", "J", "K", "L",
    "Z", "X", "C", "V", "B", "N", "M"
]
key_positions = {}

# Open Webcam
cap = cv2.VideoCapture(0)
frame_width, frame_height = 800, 600

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror the frame
    frame = cv2.resize(frame, (frame_width, frame_height))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect Hands
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get Finger Tip Positions
            index_finger_tip = hand_landmarks.landmark[8]  # Index finger tip
            thumb_tip = hand_landmarks.landmark[4]  # Thumb tip

            ix, iy = int(index_finger_tip.x * frame_width), int(index_finger_tip.y * frame_height)
            tx, ty = int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height)

            # Draw Point on Index Finger Tip
            cv2.circle(frame, (ix, iy), 10, (0, 255, 0), -1)

            # Check Thumb and Index Finger Distance
            distance = ((tx - ix) ** 2 + (ty - iy) ** 2) ** 0.5

            # If fingers are too close, do not press key
            if distance < 40:
                cv2.putText(frame, "Fingers Too Close", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # Check If Index Finger is Over a Key
                for key, pos in key_positions.items():
                    kx, ky = pos
                    if abs(ix - kx) < 20 and abs(iy - ky) < 20:
                        cv2.rectangle(frame, (kx - 20, ky - 20), (kx + 20, ky + 20), (0, 255, 0), 2)
                        cv2.putText(frame, key, (kx - 10, ky + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        pyautogui.press(key)

    # Draw Keyboard Layout
    for i, key in enumerate(keyboard_keys):
        x, y = (i % 10) * 70 + 50, (i // 10) * 70 + 200
        key_positions[key] = (x, y)
        cv2.rectangle(frame, (x - 20, y - 20), (x + 20, y + 20), (255, 0, 0), 2)
        cv2.putText(frame, key, (x - 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show Frame
    cv2.imshow("AI Virtual Keyboard", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
