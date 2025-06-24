import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from datetime import datetime
import threading

# Load model
model = tf.keras.models.load_model('sign_model.h5')

# Class labels
class_labels = ['good_luck', 'i_love_you', 'no', 'stop', 'yes']

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# GUI Setup
window = Tk()
window.title("Sign Language Detection")
window.geometry("900x700")

label_result = Label(window, text="", font=("Arial", 24))
label_result.pack(pady=20)

canvas = Canvas(window, width=640, height=480)
canvas.pack()

cap = None
running = False

# Time restriction
def is_time_allowed():
    current_time = datetime.now().time()
    return current_time >= datetime.strptime("06:00:00", "%H:%M:%S").time() and \
           current_time <= datetime.strptime("22:00:00", "%H:%M:%S").time()

# Preprocess cropped hand
def preprocess_hand(img):
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Predict from image file
def predict_image():
    if not is_time_allowed():
        messagebox.showwarning("Warning", "This system works only between 6 PM and 10 PM.")
        return

    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    img = cv2.imread(file_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            h, w, _ = img.shape
            x_min = int(min(x_coords) * w) - 40
            y_min = int(min(y_coords) * h) - 40
            x_max = int(max(x_coords) * w) + 40
            y_max = int(max(y_coords) * h) + 40

            x_min, y_min = max(x_min, 0), max(y_min, 0)
            x_max, y_max = min(x_max, w), min(y_max, h)

            cropped_hand = img[y_min:y_max, x_min:x_max]
            processed = preprocess_hand(cropped_hand)
            prediction = model.predict(processed)
            confidence = np.max(prediction)

            if confidence < 0.5:
                label_result.config(text="Not Confident")
            else:
                predicted_class = class_labels[np.argmax(prediction)]
                label_result.config(text=f'Prediction: {predicted_class} ({confidence:.2f})')
            break
    else:
        label_result.config(text="No hand detected.")
        print("⚠️ No hand detected in image.")

# Real-time webcam prediction
def start_webcam():
    if not is_time_allowed():
        messagebox.showwarning("Warning", "This system works only between 6 PM and 10 PM.")
        return

    global cap, running
    cap = cv2.VideoCapture(0)
    running = True
    threading.Thread(target=process_webcam, daemon=True).start()

def process_webcam():
    global cap, running

    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]

                h, w, _ = frame.shape
                x_min = int(min(x_coords) * w) - 40
                y_min = int(min(y_coords) * h) - 40
                x_max = int(max(x_coords) * w) + 40
                y_max = int(max(y_coords) * h) + 40

                x_min, y_min = max(x_min, 0), max(y_min, 0)
                x_max, y_max = min(x_max, w), min(y_max, h)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                cropped_hand = frame[y_min:y_max, x_min:x_max]
                processed = preprocess_hand(cropped_hand)
                prediction = model.predict(processed)
                confidence = np.max(prediction)

                if confidence < 0.5:
                    label_result.config(text="Not Confident")
                else:
                    predicted_class = class_labels[np.argmax(prediction)]
                    label_result.config(text=f'Prediction: {predicted_class} ({confidence:.2f})')

                break
        else:
            label_result.config(text="No hand detected.")
            print("⚠️ No hand detected in webcam frame.")

        # Display frame in Tkinter
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        canvas.create_image(0, 0, anchor=NW, image=imgtk)
        canvas.imgtk = imgtk

    cap.release()

def stop_webcam():
    global running
    running = False
    label_result.config(text="Webcam Stopped")

# GUI Buttons
btn_upload = Button(window, text="Upload Image", command=predict_image, width=20, height=2)
btn_upload.pack(pady=10)

btn_start = Button(window, text="Start Webcam", command=start_webcam, width=20, height=2)
btn_start.pack(pady=10)

btn_stop = Button(window, text="Stop Webcam", command=stop_webcam, width=20, height=2)
btn_stop.pack(pady=10)

window.mainloop()
