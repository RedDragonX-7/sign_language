import cv2
import numpy as np
import os
import mediapipe as mp
import urllib.request
import ssl

# Hand connections (21 hand landmarks connections)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (0, 17)  # Palm
]

#initialize the mediapipe utilities using the new tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Default hand landmark model URL (MediaPipe's official model)
HAND_LANDMARKER_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_DIR = os.path.join(os.path.expanduser("~"), ".mediapipe_models")
MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")

def ensure_model_exists():
    """Download the hand landmarker model if it doesn't exist"""
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        print(f"Downloading hand landmarker model to {MODEL_PATH}...")
        # Create SSL context that doesn't verify certificates (needed for some systems)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(HAND_LANDMARKER_MODEL_URL, context=ssl_context) as response:
            with open(MODEL_PATH, 'wb') as out_file:
                out_file.write(response.read())
        print("Model downloaded successfully!")
    return MODEL_PATH

# Compatibility class to mimic old Hands API
class Hands:
    """Compatibility class that mimics the old mp.solutions.hands.Hands API"""
    def __init__(self, 
                 static_image_mode=False,
                 max_num_hands=2,
                 model_complexity=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.static_image_mode = static_image_mode
        
        # Ensure model file exists (download if needed)
        model_path = ensure_model_exists()
        
        # Create HandLandmarker with the local model file
        # For compatibility, use IMAGE mode (can process both images and videos)
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path, delegate=BaseOptions.Delegate.CPU),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence)
        self.landmarker = HandLandmarker.create_from_options(options)
    
    def process(self, image):
        """Process an image and return results in old format"""
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        else:
            mp_image = image
        
        # Use detect() for IMAGE mode
        detection_result = self.landmarker.detect(mp_image)
        
        # Convert to old-style results
        hand_landmarks_list = []
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                landmarks = []
                for landmark in hand_landmarks:
                    class Landmark:
                        def __init__(self, x, y, z):
                            self.x = x
                            self.y = y
                            self.z = z
                    landmarks.append(Landmark(landmark.x, landmark.y, landmark.z))
                hand_landmarks_list.append(landmarks)
        
        class Results:
            def __init__(self, hand_landmarks_list):
                self.multi_hand_landmarks = hand_landmarks_list
                self.multi_hand_world_landmarks = []
                self.multi_handedness = []
        
        return Results(hand_landmarks_list)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self.landmarker, 'close'):
            self.landmarker.close()
        return False

# Create mp_hands namespace for compatibility
class HandsNamespace:
    Hands = Hands
    HAND_CONNECTIONS = HAND_CONNECTIONS

mp_hands = HandsNamespace()

#perform mediapipe detection for image (compatibility wrapper)
def mediapipe_detection(image, model):
    """Wrapper function to maintain compatibility with old API"""
    # model is a Hands instance, use its process method
    results = model.process(image)
    return image, results

#draw landmarks and hand connections using OpenCV
def draw_landmarks(image, results):
    if results.multi_hand_landmarks:
        h, w, _ = image.shape
        for hand_landmarks in results.multi_hand_landmarks:
            # Convert normalized coordinates to pixel coordinates
            points = []
            for lm in hand_landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                points.append((x, y))
                # Draw landmark points
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            
            # Draw connections
            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(points) and end_idx < len(points):
                    cv2.line(image, points[start_idx], points[end_idx], (0, 255, 0), 2)

# extract the keypoints from the detected landmarks
def extract_keypoints(results):
    if results.multi_hand_landmarks:
        rh=np.array([[res.x,res.y,res.z] for res in results.multi_hand_landmarks[0]]).flatten()
        return rh
    
    return np.zeros(21*3)

# define paths and parameters for data detection
DATA_PATH = os.path.join('MP_Data')
actions = ['apple', 'ball', 'cat']
no_sequences = 30
sequence_length = 30