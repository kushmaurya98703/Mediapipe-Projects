import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV Video Capture
cap = cv2.VideoCapture(0)

prev_x, prev_y = None, None  # Store previous hand position
gesture_text = ""  # To display detected gesture
last_update_time = time.time()  # For controlling text display duration

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process Hand Landmarks
    hand_result = hands.process(rgb_frame)

    if hand_result.multi_hand_landmarks:
        for hand_landmarks in hand_result.multi_hand_landmarks:
            # Get the X, Y coordinate of the index finger tip (landmark 8)
            index_finger_tip = hand_landmarks.landmark[8]
            h, w, _ = frame.shape
            x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            # Draw landmarks on normal frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))

            # Detect swipe direction
            if prev_x is not None and prev_y is not None:
                dx = x - prev_x
                dy = y - prev_y

                if abs(dx) > abs(dy):  # Horizontal Movement
                    if dx > 30:
                        gesture_text = "RIGHT"
                    elif dx < -30:
                        gesture_text = "LEFT"
                else:  # Vertical Movement
                    if dy > 30:
                        gesture_text = "DOWN"
                    elif dy < -30:
                        gesture_text = "UP"

                last_update_time = time.time()  # Update text time

            prev_x, prev_y = x, y  # Store last position

    # Display Gesture Text
    if time.time() - last_update_time < 1.5:  # Display text for 1.5 sec
        cv2.putText(frame, gesture_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

    # Show Output
    cv2.imshow("Hand Swipe Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
