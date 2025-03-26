import cv2
import numpy as np
import requests
import json
import time
from rembg import remove

# Step 1: Capture image from the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access webcam!")
    exit()

ret, frame = cap.read()
cap.release()  # Close the webcam

if not ret:
    print("Error: Failed to capture image!")
    exit()

# Save the captured image
cv2.imwrite("captured_image.png", frame)
print("Image saved as captured_image.png")

# Step 2: Remove background
input_image = cv2.imread("captured_image.png")
output_image = remove(input_image)

#  Step 3: Add a custom background
background = cv2.imread("background.jpg")  # Provide your background image path
if background is None:
    print("Error: Background image not found!")
    exit()

# Resize background to match the foreground size
background = cv2.resize(background, (output_image.shape[1], output_image.shape[0]))

# Convert transparent areas to background
mask = output_image[:, :, 3] > 0
final_image = background.copy()
final_image[mask] = output_image[mask][:, :3]

# Save the final image
cv2.imwrite("final_image.png", final_image)
print("Final image saved as final_image.png")

# Step 4: Upload to Google Drive
ACCESS_TOKEN = "ya29.a0AeXRPp643j-XCpoyDnSDrGF3za6ZyINHlmDQ8cOKf-P3FBBdI1AJQy4zyu6RKY-haSit0Pubaf31y5LV-rLLJSYwaeNcT_Yla8SnXnTIzEEdf-cUrfUhX0MY_RghGUb3o09L6CLo0zYlkPfI64BT4wlXpiLyCgfIwnIRIhEjaCgYKAVESARMSFQHGX2MiJs0Z4H33BXNr0DLk4zVWQg0175"  # Replace with your access token

# File metadata (Change the name dynamically)
file_name = f"photo_{int(time.time())}.jpg"  # Unique file name with timestamp
file_path = "final_image.png"  # Path to the final image

url = "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart"

metadata = {
    "name": file_name,  # Unique name for each upload
    "parents": ["1i6h6y7uPPkwF0i9TX8JNKf43pehA5PxX"]  # Replace with your Google Drive folder ID
}

files = {
    "metadata": ("metadata.json", json.dumps(metadata), "application/json"),
    "file": open(file_path, "rb"),
}

headers = {
    "Authorization": f"Bearer {ACCESS_TOKEN}"
}

# Send the request
response = requests.post(url, headers=headers, files=files)

# Print response
print(response.json())
