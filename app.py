import cv2
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
from tensorflow.keras.models import load_model
from plyer import notification

# -------------------- CONFIG --------------------

MODEL_PATH = "model/emotion_model.h5"
CASCADE_PATH = "haarcascade_frontalface_default.xml"
LOG_FILE = "data/mood_log.csv"

EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

LOG_INTERVAL_SEC = 6
ROLLING_WINDOW = 10
WARNING_LIMIT = 7
WARNING_GAP_SEC = 60
MESSAGE_VISIBLE_SEC = 5

# -------------------- LOAD MODEL --------------------

model = load_model(MODEL_PATH, compile=False)
face_detector = cv2.CascadeClassifier(CASCADE_PATH)

# -------------------- STORAGE --------------------

def write_log(emotion):
    os.makedirs("data", exist_ok=True)

    entry = {
        "time": datetime.now(),
        "emotion": emotion
    }

    if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
        df = pd.read_csv(LOG_FILE)
    else:
        df = pd.DataFrame(columns=entry.keys())

    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    df.to_csv(LOG_FILE, index=False)

# -------------------- FEEDBACK --------------------

def emotion_message(emotion):
    if emotion in ("Sad", "Fear", "Disgust", "Angry"):
        return "Hey, you've been feeling this for a while. Take a breath. Be chill 🤍"
    if emotion == "Happy":
        return "Hey cutie 😊 keep smiling — you look wonderful"
    if emotion == "Neutral":
        return "Embrace yourself. Collect moments, not things 🌱"
    return ""

def notify_user(emotion):
    msg = emotion_message(emotion)
    if not msg:
        return

    notification.notify(
        title="You're Not Alone",
        message=msg,
        timeout=6
    )

# -------------------- CAMERA --------------------

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not camera.isOpened():
    print("Camera not available.")
    exit()

print("Camera running. Press Q to exit.")

# -------------------- STATE --------------------

current_emotion = None
last_log_time = time.time()
last_warning_time = 0
message_expiry = 0

recent_emotions = []
active_message = ""

# -------------------- MAIN LOOP --------------------

while True:
    ok, frame = camera.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        if face.size == 0:
            continue

        face = cv2.resize(face, (64, 64))
        face = face / 255.0
        face = face.reshape(1, 64, 64, 1)

        prediction = model.predict(face, verbose=0)
        current_emotion = EMOTIONS[int(np.argmax(prediction))]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            current_emotion,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    now = time.time()

    if current_emotion and now - last_log_time >= LOG_INTERVAL_SEC:
        write_log(current_emotion)
        last_log_time = now

        recent_emotions.append(current_emotion)
        if len(recent_emotions) > ROLLING_WINDOW:
            recent_emotions.pop(0)

        print(f"[LOG] {datetime.now().strftime('%H:%M:%S')} → {current_emotion}")

        count = recent_emotions.count(current_emotion)

        if (
            len(recent_emotions) == ROLLING_WINDOW
            and count >= WARNING_LIMIT
            and now - last_warning_time >= WARNING_GAP_SEC
        ):
            active_message = emotion_message(current_emotion)
            message_expiry = now + MESSAGE_VISIBLE_SEC
            last_warning_time = now

            print(
                f"[WARNING] {current_emotion} repeated {count}/{ROLLING_WINDOW}"
            )

            notify_user(current_emotion)

    if active_message and now < message_expiry:
        cv2.putText(
            frame,
            active_message,
            (20, frame.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    cv2.imshow("You're Not Alone", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
print("Session ended.")
