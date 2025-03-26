import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import Button, Label
from PIL import Image, ImageTk
import threading
import time

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Load Sunglasses Images
sunglasses_options = {
    "Black": cv2.imread("sunglasses_black.png", cv2.IMREAD_UNCHANGED),
    "Red": cv2.imread("sunglasses_red.png", cv2.IMREAD_UNCHANGED),
    "Green": cv2.imread("sunglasses_green.png", cv2.IMREAD_UNCHANGED),
}

selected_sunglasses = sunglasses_options["Black"]
cap = cv2.VideoCapture(0)

def update_frame():
    global selected_sunglasses, frame
    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)  # Flip horizontally
    h, w, _ = frame.shape

    # Convert to 9:16 aspect ratio by adding padding
    target_width = int(h * (9 / 16))
    if w > target_width:
        # Crop horizontally to maintain 9:16
        crop = (w - target_width) // 2
        frame = frame[:, crop:w - crop]
    else:
        # Add black padding to maintain 9:16
        pad = (target_width - w) // 2
        frame = cv2.copyMakeBorder(frame, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    h, w, _ = frame.shape  # Update dimensions after aspect ratio adjustment

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]

            x1, y1 = int(left_eye.x * w), int(left_eye.y * h)
            x2, y2 = int(right_eye.x * w), int(right_eye.y * h)

            eye_center_x = (x1 + x2) // 2
            eye_center_y = (y1 + y2) // 2

            width = abs(x2 - x1) * 2

            if selected_sunglasses is sunglasses_options["Black"]:
                height = int(width * 0.4)  # Reduced height for Black sunglasses
            else:
                aspect_ratio = selected_sunglasses.shape[0] / selected_sunglasses.shape[1]
                height = int(width * aspect_ratio)

            x_offset = eye_center_x - width // 2
            y_offset = eye_center_y - height // 2 + 10

            resized_sunglasses = cv2.resize(selected_sunglasses, (width, height), interpolation=cv2.INTER_AREA)

            y1, y2 = max(0, y_offset), min(h, y_offset + height)
            x1, x2 = max(0, x_offset), min(w, x_offset + width)

            roi = frame[y1:y2, x1:x2]

            if roi.shape[:2] == resized_sunglasses.shape[:2]:
                alpha_s = resized_sunglasses[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s

                for c in range(3):
                    roi[:, :, c] = (alpha_s * resized_sunglasses[:, :, c] + alpha_l * roi[:, :, c]).astype(np.uint8)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Resize frame to match window size while maintaining 9:16 aspect ratio
    screen_height = root.winfo_screenheight()
    screen_width = int(screen_height * (9 / 16))
    frame = cv2.resize(frame, (screen_width, screen_height))

    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    lbl_video.imgtk = imgtk
    lbl_video.configure(image=imgtk)
    lbl_video.after(10, update_frame)

def select_sunglasses(color):
    global selected_sunglasses
    selected_sunglasses = sunglasses_options[color]

def download_image():
    if frame is not None:
        timestamp = int(time.time())
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB before saving
        filename = f"captured_image_{timestamp}.png"
        cv2.imwrite(filename, rgb_frame)
        print(f"Image saved as {filename}")

def start_app():
    btn_start.place_forget()  # Hide start button
    lbl_video.pack(fill="both", expand=True)

    # Show buttons with equal spacing (adjusted for 9:16 ratio)
    button_spacing = 0.05  # Adjust this value for more or less spacing
    btn_black.place(relx=0.3, rely=0.92, anchor="center")  # 30% from left
    btn_red.place(relx=0.3 + button_spacing, rely=0.92, anchor="center")  # 35% from left
    btn_green.place(relx=0.3 + 2 * button_spacing, rely=0.92, anchor="center")  # 40% from left
    btn_download.place(relx=0.3 + 3 * button_spacing, rely=0.92, anchor="center")  # 45% from left

    threading.Thread(target=update_frame, daemon=True).start()

def exit_fullscreen(event=None):
    root.attributes("-fullscreen", False)

# Tkinter GUI
root = tk.Tk()
root.title("Face Filter App - Mediapipe")
root.configure(bg="black")

# Set window to 9:16 aspect ratio
screen_height = root.winfo_screenheight()
screen_width = int(screen_height * (9 / 16))
root.geometry(f"{screen_width}x{screen_height}")

lbl_video = Label(root)
lbl_video.pack(fill="both", expand=True)

# Load button images with adjusted height for black sunglasses
def load_button_image(img_path, new_height):
    img = Image.open(img_path)
    aspect_ratio = img.width / img.height
    new_width = int(new_height * aspect_ratio)  # Maintain aspect ratio
    img = img.resize((new_width, new_height), Image.LANCZOS)
    return ImageTk.PhotoImage(img)

# Load button images with adjusted heights
black_img = load_button_image("sunglasses_black.png", 25)  # Reduced height
red_img = load_button_image("sunglasses_red.png", 30)
green_img = load_button_image("sunglasses_green.png", 30)

# Create buttons with margins
button_spacing = 5  # Adjust this value for more or less spacing
btn_black = Button(root, image=black_img, command=lambda: select_sunglasses("Black"), borderwidth=0, bg="grey")
btn_red = Button(root, image=red_img, command=lambda: select_sunglasses("Red"), borderwidth=0, bg="grey")
btn_green = Button(root, image=green_img, command=lambda: select_sunglasses("Green"), borderwidth=0, bg="grey")

# Download Button
btn_download = Button(root, text="Download", font=("Arial", 12, "bold"), fg="white", bg="green", command=download_image)

# Place buttons with spacing
def start_app():
    btn_start.place_forget()  # Hide start button
    lbl_video.pack(fill="both", expand=True)

    # Show buttons with equal spacing (adjusted for 9:16 ratio)
    btn_black.place(relx=0.1, rely=0.92, anchor="center")  # 30% from left
    btn_red.place(relx=0.3 + (btn_black.winfo_width() + button_spacing) / root.winfo_width(), rely=0.92, anchor="center")  # 35% from left
    btn_green.place(relx=0.5 + 2 * (btn_black.winfo_width() + button_spacing) / root.winfo_width(), rely=0.92, anchor="center")  # 40% from left
    btn_download.place(relx=0.8 + 3 * (btn_black.winfo_width() + button_spacing) / root.winfo_width(), rely=0.92, anchor="center")  # 45% from left

    threading.Thread(target=update_frame, daemon=True).start()

# Download Button
# btn_download = Button(root, text="Download", font=("Arial", 12, "bold"), fg="white", bg="green", command=download_image)

# START Button (Centered Before Fullscreen)
btn_start = Button(root, text="START", font=("Arial", 20, "bold"), fg="white", bg="blue", command=start_app)
btn_start.place(relx=0.5, rely=0.5, anchor="center")  # Centered Start Button

# Bind ESC key to exit fullscreen
root.bind("<Escape>", exit_fullscreen)

root.mainloop()
cap.release()
cv2.destroyAllWindows()