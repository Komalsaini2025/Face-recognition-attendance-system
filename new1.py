import streamlit as st
import face_recognition
import numpy as np
import pandas as pd
import pickle
import cv2
from datetime import datetime
import os
import pyttsx3
import threading

ENCODINGS_FILE = 'face_encodings.pkl'
ATTENDANCE_FILE = 'attendance.csv'

# Function to speak text
def speak(text):
    def run_speak():
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            st.error(f"Error in text-to-speech: {e}")

    # Run the speaking function in a separate thread
    thread = threading.Thread(target=run_speak, daemon=True)
    thread.start()

# Initialize attendance log and spoken names
attendance_log = []
spoken_names = set()  # Track names for which speech has been made

# Load known encodings and names if available
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, 'rb') as f:
        known_encodings, known_names = pickle.load(f)
else:
    known_encodings = []
    known_names = []

# Function to mark attendance
def mark_attendance(name):
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')

    # Check if attendance is already marked for the person on the same date
    if not any(entry[0] == name and entry[2] == date_str for entry in attendance_log):
        attendance_log.append([name, time_str, date_str])
        st.success(f"Attendance marked for {name}.")
    else:
        st.info(f"{name} is already marked present today.")

# Function to save attendance to a CSV file
def save_attendance():
    if not attendance_log:
        st.warning("No attendance to save.")
        return

    # Create DataFrame
    attendance_df = pd.DataFrame(attendance_log, columns=["Name", "Time", "Date"])

    # Check if the attendance file exists
    if os.path.exists(ATTENDANCE_FILE):
        existing_df = pd.read_csv(ATTENDANCE_FILE)
        attendance_df = pd.concat([existing_df, attendance_df]).drop_duplicates()

    # Save the updated DataFrame
    attendance_df.to_csv(ATTENDANCE_FILE, index=False)
    st.success("Attendance saved successfully!")
    speak(f"Attendance marked for {name}. Thank you!")

# Function to register a new face
def register_new_face(face_encoding, name):
    known_encodings.append(face_encoding)
    known_names.append(name)
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump((known_encodings, known_names), f)
    st.success(f"New face registered for {name}.")

# Streamlit UI
st.title("Face Recognition Attendance System")
st.write("Welcome to the Face Recognition Attendance App")

# Use session state to ensure the welcome message is spoken only once
if "has_spoken_welcome" not in st.session_state:
    st.session_state.has_spoken_welcome = False

if not st.session_state.has_spoken_welcome:
    speak("Welcome to the Face Recognition Attendance App")
    st.session_state.has_spoken_welcome = True

if "known_encodings" not in st.session_state:
    st.session_state.known_encodings = known_encodings
if "known_names" not in st.session_state:
    st.session_state.known_names = known_names

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Error: Could not open camera.")
else:
    ret, frame = cap.read()

    if ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if face_encodings:
            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(st.session_state.known_encodings, face_encoding)
                face_distances = face_recognition.face_distance(st.session_state.known_encodings, face_encoding)

                if matches and any(matches):
                    best_match_index = np.argmin(face_distances)
                    name = st.session_state.known_names[best_match_index]
                    mark_attendance(name)

                    # Draw a rectangle around the face
                    top, right, bottom, left = face_location
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        name,
                        (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )
                else:
                    st.warning("Unknown face detected.")
                    new_name = st.text_input("Enter your name to register:")

                    if new_name and st.button("Register New Face"):
                        register_new_face(face_encoding, new_name)
                        mark_attendance(new_name)

            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Processed Image")
        else:
            st.warning("No faces detected. Please try again.")

        cap.release()

if st.button("Save Attendance"):
    save_attendance()

if st.button("View Attendance Log"):
    if os.path.exists(ATTENDANCE_FILE):
        attendance_df = pd.read_csv(ATTENDANCE_FILE)
        st.dataframe(attendance_df)
    else:
        st.info("No attendance records found.")
