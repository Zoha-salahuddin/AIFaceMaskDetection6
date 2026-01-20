import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from tensorflow.keras.models import load_model

# ======================
# THEME COLORS
# ======================
BG_COLOR = "#FFD6F5"
PURPLE = "#50207A"
PINK = "#FF48B9"
GREEN = "#12CE6A"
WHITE = "#FFFFFF"

# ---------------- Load Model ----------------
model = load_model("model/mask_detector_model.h5")
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- Mask Detection Function ----------------
def detect_mask(frame):
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (224, 224))
        face_resized = face_resized / 255.0
        face_resized = np.reshape(face_resized, (1, 224, 224, 3))

        pred = model.predict(face_resized)

        if pred.shape[1] == 1:
            prob = float(pred[0][0])
            label = "MASK" if prob > 0.5 else "NO MASK"
        else:
            label = "MASK" if pred[0][0] > pred[0][1] else "NO MASK"

        color = (18, 206, 106) if label == "MASK" else (255, 72, 185)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )

    return frame

# ---------------- Webcam Function ----------------
def start_webcam():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_mask(frame)
        cv2.imshow("AI Face Mask Detection (Press 'q' to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ======================
# GUI WINDOW
# ======================
root = tk.Tk()
root.title("🌸✨ AI Face Mask Detection ✨🌸")
root.geometry("600x420")
root.configure(bg=BG_COLOR)
root.resizable(False, False)

# ======================
# TITLE
# ======================
title = tk.Label(
    root,
    text="AI FACE MASK DETECTION",
    bg=BG_COLOR,
    fg=PURPLE,
    font=("Poppins", 26, "bold")
)
title.pack(pady=(25, 10))

# ======================
# SUBTITLE
# ======================
subtitle = tk.Label(
    root,
    text="AI-powered real-time face mask detection",
    bg=BG_COLOR,
    fg=PURPLE,
    font=("Poppins", 12)
)
subtitle.pack(pady=(0, 30))

# ======================
# BUTTON
# ======================
start_btn = tk.Button(
    root,
    text="START WEBCAM",
    command=start_webcam,
    bg=GREEN,
    fg=WHITE,
    activebackground=GREEN,
    font=("Poppins", 14, "bold"),
    padx=40,
    pady=15,
    bd=0,
    relief="flat"
)
start_btn.pack()

# ======================
# FOOTER
# ======================
footer = tk.Label(
    root,
    text="Press 'q' to stop webcam",
    bg=BG_COLOR,
    fg=PURPLE,
    font=("Poppins", 10)
)
footer.pack(side="bottom", pady=20)

# ======================
# RUN GUI
# ======================
root.mainloop()
