# Detect hand ,face and draw circle from them.

import cv2
import mediapipe as mp
import math

# Initialize MediaPipe for Face and Hand detection
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.7)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Open Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # Flip for mirror effect
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    # Face Detection & Circle Drawing
    face_results = face_detection.process(rgb_frame)
    if face_results.detections:
        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            cx = int((bboxC.xmin + bboxC.width / 2) * w)
            cy = int((bboxC.ymin + bboxC.height / 2) * h)
            radius = int(min(bboxC.width * w, bboxC.height * h) / 2)
            cv2.circle(frame, (cx, cy), radius, (0, 255, 0), 3)  # Green circle around face

    # Hand Detection & Circle Drawing
    hand_results = hands.process(rgb_frame)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            x_list = [int(lm.x * w) for lm in hand_landmarks.landmark]
            y_list = [int(lm.y * h) for lm in hand_landmarks.landmark]

            if x_list and y_list:
                center_x = int((min(x_list) + max(x_list)) / 2)
                center_y = int((min(y_list) + max(y_list)) / 2)
                radius = int(math.hypot(max(x_list) - min(x_list), max(y_list) - min(y_list)) / 2)
                cv2.circle(frame, (center_x, center_y), radius, (255, 0, 0), 3)  # Blue circle around full hand

    # Show the output
    cv2.imshow("Face & Hand Circle Detection", frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
