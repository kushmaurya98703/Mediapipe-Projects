import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 Face Detection Model
model = YOLO("yolov8m-face-lindevs.pt")

# Load Sunglasses Image (PNG with Alpha)
sunglasses = cv2.imread("sunglasses_black.png", cv2.IMREAD_UNCHANGED)

# Check if sunglasses image has alpha channel
if sunglasses.shape[2] != 4:
    raise ValueError("Sunglasses image must have 4 channels (RGBA)")

# **Scaling Factor (Adjust for Sunglasses Size)**
SCALING_FACTOR = 1.2  # Increase for larger sunglasses

# **Webcam Initialization**
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)  # Set higher FPS for smooth performance

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip for mirror effect
    h, w, _ = frame.shape

    # **Step 1: Detect Faces Using YOLO (Optimized for Speed)**
    results = model.predict(frame, conf=0.5, iou=0.4, verbose=False)  # Lower IOU & Confidence for faster detection
    detections = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes

    for box in detections:
        x1, y1, x2, y2 = map(int, box)  # Convert bounding box coordinates to integers

        face_width = x2 - x1
        face_height = y2 - y1

        # **Ignore Very Small Faces**
        if face_width < 80 or face_height < 80:
            continue

        # **Step 2: Resize Sunglasses with Scaling Factor**
        sunglasses_width = int(face_width * 0.75 * SCALING_FACTOR)
        sunglasses_height = int(sunglasses_width * (sunglasses.shape[0] / sunglasses.shape[1]))

        resized_sunglasses = cv2.resize(sunglasses, (sunglasses_width, sunglasses_height), interpolation=cv2.INTER_LINEAR)

        # **Step 3: Position Sunglasses**
        x_offset = x1 + int(0.12 * face_width)
        y_offset = y1 + int(0.35 * face_height)

        # **Ensure ROI Stays Within Frame**
        y1, y2 = max(0, y_offset), min(h, y_offset + sunglasses_height)
        x1, x2 = max(0, x_offset), min(w, x_offset + sunglasses_width)

        if x1 >= w or y1 >= h or x2 <= 0 or y2 <= 0:
            continue

        roi = frame[y1:y2, x1:x2]

        # **Step 4: Apply Sunglasses Overlay**
        if roi.shape[:2] == resized_sunglasses.shape[:2]:
            alpha_s = resized_sunglasses[:, :, 3] / 255.0  # Normalize alpha
            alpha_l = 1.0 - alpha_s

            for c in range(3):  # Apply for R, G, B channels
                roi[:, :, c] = (alpha_s * resized_sunglasses[:, :, c] + alpha_l * roi[:, :, c]).astype(np.uint8)

    # **Show Output**
    cv2.imshow("YOLOv8 Face Filter (Optimized)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
