import cv2
import numpy as np
import face_recognition
import os
import pickle
from datetime import datetime
import pandas as pd
import pyttsx3
import pythoncom

# Initialize variables
attendance_log = []

# File paths for saving/loading encodings and names
ENCODINGS_FILE = 'face_encodings.pkl'
ATTENDANCE_FILE = 'attendance.csv'

# Initialize the text-to-speech engine
pythoncom.CoInitialize()
engine = pyttsx3.init()

# Function to speak text
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Load previously saved encodings and names if they exist
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, 'rb') as f:
        known_encodings, known_names = pickle.load(f)
else:
    known_encodings = []
    known_names = []

# Function to show the welcome screen
def show_welcome_screen():
    # Create a black image to display the welcome screen
    welcome_screen = np.zeros((500, 800, 3), dtype=np.uint8)

    # Define the text to be displayed
    text = "Welcome to the Face Recognition Attendance App"

    # Set font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255)  # White color
    thickness = 2

    # Calculate the position of the text to center it on the screen
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (welcome_screen.shape[1] - text_size[0]) // 2
    text_y = (welcome_screen.shape[0] + text_size[1]) // 2

    # Add the text to the welcome screen
    cv2.putText(welcome_screen, text, (text_x, text_y), font, font_scale, color, thickness)

    # Display the welcome screen
    cv2.imshow('Welcome Screen', welcome_screen)

    # Speak the welcome message
    speak("Welcome to the Face Recognition Attendance App")

    # Wait for 5 seconds (5000 milliseconds) before closing the window
    cv2.waitKey(5000)  # This ensures the window stays open for 5 seconds

    # Destroy the window after the wait time
    cv2.destroyAllWindows()  # This ensures all windows are closed properly


# Function to mark attendance
def mark_attendance(name):
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')
    if not any(entry[0] == name and entry[2] == date_str for entry in attendance_log):
        attendance_log.append([name, time_str, date_str])
        print(f"Attendance marked for {name} on {date_str} at {time_str}")
        speak(f"Attendance marked for {name}. Thank you!")
    else:
        print(f"{name} is already marked present on {date_str}.")

# Function to save attendance
def save_attendance():
    attendance_df = pd.DataFrame(attendance_log, columns=["Name", "Time", "Date"])
    if os.path.exists(ATTENDANCE_FILE):
        existing_df = pd.read_csv(ATTENDANCE_FILE)
        attendance_df = pd.concat([existing_df, attendance_df]).drop_duplicates()
    attendance_df.to_csv(ATTENDANCE_FILE, index=False)
    print(f"Attendance saved to: {os.path.abspath(ATTENDANCE_FILE)}")

# Blink detection parameters
EAR_THRESHOLD = 0.2
CONSECUTIVE_FRAMES = 3
blink_counters = {}
blink_states = {}

# Load Haar Cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Main application logic
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Show the welcome screen before starting the main loop
show_welcome_screen()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        if matches and any(matches):
            best_match_index = np.argmin(face_distances)
            name = known_names[best_match_index]
            top, right, bottom, left = face_location
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4

            # Draw rectangle around face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Detect eyes within the face region
            roi_gray = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
            eyes = eye_cascade.detectMultiScale(roi_gray)

            # Blink detection logic
            if len(eyes) >= 2:
                if name not in blink_counters:
                    blink_counters[name] = 0
                    blink_states[name] = False

                blink_counters[name] += 1
                if blink_counters[name] >= CONSECUTIVE_FRAMES and not blink_states[name]:
                    print(f"Blink detected for {name}. Marking attendance...")
                    mark_attendance(name)
                    blink_states[name] = True
            else:
                blink_counters[name] = 0
                blink_states[name] = False
        else:
            top, right, bottom, left = face_location
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Attendance System', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release the video capture and save attendance
cap.release()
cv2.destroyAllWindows()
save_attendance()
