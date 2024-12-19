import cv2
import numpy as np
import face_recognition
import os
import pickle
from datetime import datetime
import pandas as pd
import pyttsx3

# Initialize variables
attendance_log = []

# File paths for saving/loading encodings and names
ENCODINGS_FILE = 'face_encodings.pkl'
ATTENDANCE_FILE = 'attendance.csv'

# Initialize the text-to-speech engine
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

# Function to mark attendance
def markAttendance(name):
    now = datetime.now()
    dtString = now.strftime('%H:%M:%S')
    if name not in [entry[0] for entry in attendance_log]:
        attendance_log.append([name, dtString])
        print(f"Attendance marked for {name} at {dtString}")
    else:
        print(f"{name} is already marked present.")

# Function to register a new face
def register_face(frame, face_location):
    top, right, bottom, left = face_location
    # Scale back up to the original frame size
    top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4

    # Extract the face image
    face_image = frame[top:bottom, left:right]

    # Convert the face image to RGB
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    # Get face encodings
    face_encodings = face_recognition.face_encodings(face_image)

    if face_encodings:  # Check if encoding was successful
        known_encodings.append(face_encodings[0])
        name = input("Enter the name of the person: ")
        known_names.append(name.capitalize())
        print(f"Face registered for {name}.")

        # Speak the registered name and thank you message
        speak(f"Face registered for {name}. Thank you!")

        # Save the encodings and names to a file
        with open(ENCODINGS_FILE, 'wb') as f:
            pickle.dump((known_encodings, known_names), f)
        print("Encodings saved.")
    else:
        print("No face encoding found. Ensure the face is clearly visible.")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Speak the welcome message when the interface opens
speak("Welcome to the Face Recognition Attendance App")

print("Press 'r' to register a new face, or 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Resize the frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces and compute encodings
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Check if the face is already registered
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        if matches and any(matches):
            best_match_index = np.argmin(face_distances)
            name = known_names[best_match_index]

            # Mark attendance
            markAttendance(name)

            # Draw rectangle and name on the face
            top, right, bottom, left = face_location
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            # If the face is not registered, prompt to register
            top, right, bottom, left = face_location
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, "Press 'r' to Register", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the video feed
    cv2.imshow('Attendance System', frame)

    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('r'):  # Register new face
        if face_locations:
            register_face(frame, face_locations[0])
        else:
            print("No face detected for registration. Try again.")

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()

# Save attendance to a CSV file
attendance_df = pd.DataFrame(attendance_log, columns=["Name", "Time"])
attendance_df.to_csv(ATTENDANCE_FILE, index=False)
print(f"Attendance saved to: {os.path.abspath(ATTENDANCE_FILE)}")
