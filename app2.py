from function import *
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.layers import LSTM, Dense
import numpy as np
import cv2

# load the trained model
json_file = open('model.json', 'r')
model_json = json_file.read()
json_file.close()

model = model_from_json(model_json)
model.load_weights('model.h5')

#set color for different actions
colors = []
for i in range(0,20):
    colors.append((255,117,16))

# function for visualizing the probs
def prob_viz(res, actions, input_frame, colors, threshold):
    output_frame = input_frame.copy()

    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100),90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0,85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

    return output_frame

# detection and display of var
sequence = []
sentence = []
accuracy = []
predictions = []
threshold = 0.8

# Fixed tracking window position and size
track_window_x = 0
track_window_y = 40
track_window_width = 300
track_window_height = 360

cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera. Please check camera permissions.")
    exit(1)

# Create window
cv2.namedWindow('OpenCV Feed')

# initialize mediapipe for hand tracking
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    #loop through every frame
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break
        
        frame_height, frame_width = frame.shape[:2]
        
        # Process the frame in the fixed region
        cropframe = frame[track_window_y:track_window_y+track_window_height, 
                         track_window_x:track_window_x+track_window_width]

        # Draw the fixed tracking window rectangle
        cv2.rectangle(frame, 
                     (track_window_x, track_window_y), 
                     (track_window_x + track_window_width, track_window_y + track_window_height), 
                     (245, 117, 16), 2)
        
        image, results = mediapipe_detection(cropframe, hands)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:] 

        try:
            # Make prediction when we have enough frames (30 frames)
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]  
                predictions.append(np.argmax(res))
                
                # Only process if we have enough predictions
                if len(predictions) >= 10:
                    if np.unique(predictions[-10:])[0] == np.argmax(res): 
                        if res[np.argmax(res)] > threshold: 
                            if len(sentence) > 0: 
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                                    accuracy.append(str(res[np.argmax(res)]*100))

                            else:
                                sentence.append(actions[np.argmax(res)])
                                accuracy.append(str(res[np.argmax(res)]*100))   
            
            # Keep only the last prediction in sentence and accuracy
            if len(sentence) > 1:
                sentence = sentence[-1:]
                accuracy = accuracy[-1:]

        except Exception as e: 
            print(f"Error: {e}")  # Print error for debugging instead of silently passing

        # Draw output text area at the bottom to avoid overlapping with tracking window
        output_y = frame_height - 40
        cv2.rectangle(frame, (0, output_y), (frame_width, frame_height), (245,117,16), -1) 
        output_text = 'Output: ' + ' '.join(sentence) + ' ' + ''.join(accuracy) if sentence else 'Output: Waiting...'
        # Truncate text if too long
        if len(output_text) > 60:
            output_text = output_text[:57] + '...'
        cv2.putText(frame, output_text, (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow('OpenCV Feed', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()