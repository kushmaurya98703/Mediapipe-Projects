import cv2
import mediapipe as mp
import numpy as np
import random
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# OpenCV Video Capture
cap = cv2.VideoCapture(0)

# Questions and Answers (MCQs)
questions = [
    {"q": "What is 2 + 2?", "options": ["3", "4", "5", "6"], "correct": 1},
    {"q": "Capital of India?", "options": ["Mumbai", "Delhi", "Kolkata", "Chennai"], "correct": 1},
    {"q": "Largest planet?", "options": ["Mars", "Venus", "Jupiter", "Saturn"], "correct": 2},
    {"q": "H2O is?", "options": ["Oxygen", "Water", "Hydrogen", "Acid"], "correct": 1},
    {"q": "5 x 3?", "options": ["8", "15", "10", "20"], "correct": 1},
    {"q": "CPU full form?", "options": ["Central Process Unit", "Central Processing Unit", "Core Process Unit", "None"], "correct": 1},
    {"q": "2^3?", "options": ["6", "8", "4", "9"], "correct": 1},
    {"q": "Who invented Python?", "options": ["Guido", "Dennis", "James", "Elon"], "correct": 0},
    {"q": "Which is a programming language?", "options": ["HTML", "CSS", "Python", "Photoshop"], "correct": 2},
    {"q": "Which is an OS?", "options": ["Windows", "Chrome", "Google", "Edge"], "correct": 0},
]

# Quiz Variables
score = 0
wrong = 0
question_index = 0
show_result = False
result_text = ""

# Hand Tracking
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Flip for mirror effect
        h, w, _ = frame.shape

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        # Draw question and options
        current_question = questions[question_index]
        cv2.putText(frame, current_question["q"], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        option_positions = [(100, 150), (100, 250), (100, 350), (100, 450)]
        option_colors = [(0, 0, 255)] * 4  # By default sabka red border hoga

        # Hand Landmarks Detection
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Index Finger Tip Coordinates
                index_finger = hand_landmarks.landmark[8]
                fx, fy = int(index_finger.x * w), int(index_finger.y * h)

                cv2.circle(frame, (fx, fy), 10, (255, 0, 0), -1)

                # Check if index finger is pointing at an option
                for i, (x, y) in enumerate(option_positions):
                    if x < fx < x+300 and y < fy < y+50:
                        if i == current_question["correct"]:
                            result_text = "RIGHT!"
                            option_colors[i] = (0, 255, 0)  # Green border for correct answer
                            score += 1
                        else:
                            result_text = "WRONG!"
                        show_result = True
                        time.sleep(1)  # Delay to prevent multiple selection
                        question_index += 1
                        break  # Stop checking further options

        # Draw Options
        for i, (x, y) in enumerate(option_positions):
            cv2.rectangle(frame, (x, y), (x+300, y+50), option_colors[i], 2)
            cv2.putText(frame, current_question["options"][i], (x+10, y+35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show result (RIGHT or WRONG)
        if show_result:
            cv2.putText(frame, result_text, (400, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            show_result = False

        # End of Quiz
        if question_index >= len(questions):
            frame.fill(0)
            cv2.putText(frame, f"Quiz Completed!", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.putText(frame, f"Correct: {score}", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            cv2.putText(frame, f"Wrong: {wrong}", (150, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.imshow("Quiz Game", frame)
            cv2.waitKey(5000)
            break

        cv2.imshow("Quiz Game", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
