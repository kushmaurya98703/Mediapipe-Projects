import cv2
import mediapipe as mp
import numpy as np

# Initialize Hand and Face Detection
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh()
mp_drawing = mp.solutions.drawing_utils

# Initialize Camera
cap = cv2.VideoCapture(0)
prev_x, prev_y = None, None
filter_index = 0
zoom_factor = 1.0  # Initial Zoom Factor
filters = ['normal', 'cartoon', 'big_eyes', 'slim_face', 'swap_face']


def apply_filter(frame, filter_type, zoom_factor):
    modified_frame = frame.copy()
    height, width, _ = modified_frame.shape

    if filter_type == 'cartoon':
        gray = cv2.cvtColor(modified_frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(modified_frame, 9, 250, 250)
        modified_frame = cv2.bitwise_and(color, color, mask=edges)

    elif filter_type == 'big_eyes':
        M = cv2.getRotationMatrix2D((width // 2, height // 2), 0, zoom_factor)
        modified_frame = cv2.warpAffine(modified_frame, M, (width, height))

    elif filter_type == 'slim_face':
        M = cv2.getRotationMatrix2D((width // 2, height // 2), 0, 0.9 * zoom_factor)
        modified_frame = cv2.warpAffine(modified_frame, M, (width, height))

    elif filter_type == 'swap_face':
        modified_frame = cv2.GaussianBlur(modified_frame, (15, 15), 0)

    return modified_frame


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_result = hands.process(rgb_frame)

    if hand_result.multi_hand_landmarks:
        for hand_landmarks in hand_result.multi_hand_landmarks:
            index_tip = hand_landmarks.landmark[8]  # Index Finger Tip
            thumb_tip = hand_landmarks.landmark[4]  # Thumb Tip
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]

            h, w, _ = frame.shape
            x, y = int(index_tip.x * w), int(index_tip.y * h)

            # Detect swipe for filter change
            if prev_x is not None:
                dx = x - prev_x
                if abs(dx) > 30:
                    if dx > 0:
                        filter_index = (filter_index + 1) % len(filters)
                    else:
                        filter_index = (filter_index - 1) % len(filters)
            prev_x, prev_y = x, y

            # Detect Palm Squeeze (If all fingers close together except thumb)
            if (abs(index_tip.y - middle_tip.y) < 0.05 and abs(middle_tip.y - ring_tip.y) < 0.05 and
                    abs(ring_tip.y - pinky_tip.y) < 0.05 and abs(index_tip.x - thumb_tip.x) > 0.1):
                zoom_factor += 0.02  # Zoom In
                if zoom_factor > 2.0:
                    zoom_factor = 2.0
            else:
                zoom_factor = 1.0  # Reset Zoom when hand is open

    modified_frame = apply_filter(frame, filters[filter_index], zoom_factor)

    # 🔹 Side-by-side display of Original and Modified
    combined = np.hstack((frame, modified_frame))

    cv2.putText(combined, f"Filter: {filters[filter_index]}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined, f"Zoom: {zoom_factor:.2f}x", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("Magic Mirror AI (Left: Original | Right: M