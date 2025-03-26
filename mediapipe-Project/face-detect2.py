import cv2
import mediapipe as mp

# MediaPipe Face Detection module initialize
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Webcam se video capture karega
cap = cv2.VideoCapture(0)

# Face detection model ko initialize karna
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR se RGB me convert karna (MediaPipe ke liye)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face detect karna
        results = face_detection.process(rgb_frame)

        # Agar face detect hota hai to uspe bounding box draw karega
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)

        # Output window me frame show karna
        cv2.imshow("Face Detection", frame)

        # 'q' dabaane par loop break hoga
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Resources ko release karna
cap.release()
cv2.destroyAllWindows()
