# ===================== IMPORT REQUIRED LIBRARIES =====================
from function import *                     # Custom helper functions (mediapipe, keypoints, actions)
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.layers import LSTM, Dense
import numpy as np
import cv2
import pyttsx3                              # Text-to-speech library

# ===================== TEXT TO SPEECH INITIALIZATION =====================
engine = pyttsx3.init()                    # Initialize text-to-speech engine
engine.setProperty('rate', 150)             # Speech speed
engine.setProperty('volume', 1.0)           # Speech volume (0.0 to 1.0)
last_spoken = ""                            # Stores last spoken word to avoid repetition

# ===================== LOAD TRAINED MODEL =====================
json_file = open('model.json', 'r')         # Load model architecture
model_json = json_file.read()
json_file.close()

model = model_from_json(model_json)         # Create model from JSON
model.load_weights('model.h5')              # Load trained weights

# ===================== DEFINE COLORS FOR VISUALIZATION =====================
colors = []
for i in range(0, 20):
    colors.append((255, 117, 16))

# ===================== FUNCTION TO VISUALIZE PROBABILITIES =====================
def prob_viz(res, actions, input_frame, colors, threshold):
    """
    Draws probability bars for each action
    """
    output_frame = input_frame.copy()

    for num, prob in enumerate(res):
        cv2.rectangle(output_frame,
                      (0, 60 + num * 40),
                      (int(prob * 100), 90 + num * 40),
                      colors[num], -1)
        cv2.putText(output_frame,
                    actions[num],
                    (0, 85 + num * 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

# ===================== VARIABLES FOR PREDICTION =====================
sequence = []          # Stores last 30 frames of keypoints
sentence = []          # Stores detected word
accuracy = []          # Stores confidence score
predictions = []       # Stores prediction history
threshold = 0.8        # Confidence threshold

# ===================== FIXED TRACKING WINDOW =====================
track_window_x = 0
track_window_y = 40
track_window_width = 300
track_window_height = 360

# ===================== OPEN CAMERA =====================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit(1)

cv2.namedWindow('OpenCV Feed')

# ===================== MEDIAPIPE HAND TRACKING =====================
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_height, frame_width = frame.shape[:2]

        # Crop fixed hand tracking area
        cropframe = frame[
            track_window_y:track_window_y + track_window_height,
            track_window_x:track_window_x + track_window_width
        ]

        # Draw tracking rectangle
        cv2.rectangle(frame,
                      (track_window_x, track_window_y),
                      (track_window_x + track_window_width,
                       track_window_y + track_window_height),
                      (245, 117, 16), 2)

        # Mediapipe detection
        image, results = mediapipe_detection(cropframe, hands)

        # Extract keypoints and store sequence
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]  # Keep only last 30 frames

        try:
            # Predict only when we have 30 frames
            if len(sequence) == 30:
                res = model.predict(
                    np.expand_dims(sequence, axis=0),
                    verbose=0
                )[0]

                predictions.append(np.argmax(res))

                # Check prediction stability
                if len(predictions) >= 10:
                    if np.unique(predictions[-10:])[0] == np.argmax(res):
                        if res[np.argmax(res)] > threshold:

                            current_action = actions[np.argmax(res)]

                            # Add only new action
                            if len(sentence) == 0 or current_action != sentence[-1]:
                                sentence.append(current_action)
                                accuracy.append(str(res[np.argmax(res)] * 100))

                                # ===================== TEXT TO SPEECH =====================
                                # Speak only if word is new
                                if current_action != last_spoken:
                                    engine.say(current_action)
                                    engine.runAndWait()
                                    last_spoken = current_action
                                else:
                                    engine.say(current_action)
                                    engine.runAndWait()
                                    last_spoken = current_action
                                # ==========================================================

            # Keep only latest word
            if len(sentence) > 1:
                sentence = sentence[-1:]
                accuracy = accuracy[-1:]

        except Exception as e:
            print(f"Error: {e}")

        # ===================== DISPLAY OUTPUT =====================
        output_y = frame_height - 40
        cv2.rectangle(frame,
                      (0, output_y),
                      (frame_width, frame_height),
                      (245, 117, 16), -1)

        output_text = (
            'Output: ' + ' '.join(sentence) + ' ' + ''.join(accuracy)
            if sentence else 'Output: Waiting...'
        )

        if len(output_text) > 60:
            output_text = output_text[:57] + '...'

        cv2.putText(frame,
                    output_text,
                    (10, frame_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('OpenCV Feed', frame)

        # Press Q to exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()