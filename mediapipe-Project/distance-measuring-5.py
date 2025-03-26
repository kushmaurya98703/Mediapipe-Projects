import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

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

            # Get Thumb Tip and Index Finger Tip Positions
            thumb_tip = hand_landmarks.landmark[4]   # Thumb Tip
            index_tip = hand_landmarks.landmark[8]   # Index Finger Tip

            tx, ty = int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height)
            ix, iy = int(index_tip.x * frame_width), int(index_tip.y * frame_height)

            # Draw Circles on Thumb and Index Finger
            cv2.circle(frame, (tx, ty), 10, (0, 0, 255), -1)  # Red Circle on Thumb
            cv2.circle(frame, (ix, iy), 10, (0, 255, 0), -1)  # Green Circle on Index Finger

            # Draw Line Between Thumb and Index Finger
            cv2.line(frame, (tx, ty), (ix, iy), (255, 255, 0), 2)

            # Calculate Distance
            distance = np.sqrt((tx - ix) ** 2 + (ty - iy) ** 2)

            # Convert Distance to Object Size (Scale it to cm)
            object_size = int(distance / 5)  # Adjust Scale Factor as Needed

            # Display Distance
            cv2.putText(frame, f"Distance: {int(distance)} px", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            # Display Object Size Prediction
            cv2.putText(frame, f"Size: {object_size} cm", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show Frame
    cv2.imshow("Hand Distance Measurement", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
