import cv2
import time
import winsound
import numpy as np
import threading
from collections import deque
import imutils
from scipy.spatial import distance as dist

# Try to import dlib, but make it optional
try:
    import dlib
    from imutils import face_utils
    DLIB_AVAILABLE = True
    print("dlib is available - Enhanced detection enabled")
except ImportError:
    DLIB_AVAILABLE = False
    print("dlib not available - Using fallback detection")
    print("  For enhanced accuracy, install dlib:")
    print("  pip install https://github.com/z-mahmud22/Dlib_Windows_Python3.x/raw/main/dlib-19.24.0-cp311-cp311-win_amd64.whl")

# Constants for drowsiness detection
EYE_AR_THRESH = 0.25  # Eye aspect ratio threshold
EYE_AR_CONSEC_FRAMES = 20  # Number of consecutive frames for drowsiness
YAWN_THRESH = 20  # Mouth aspect ratio threshold for yawning

# Global variables
alarm_playing = False
drowsy_frames = 0

def calculate_ear(eye):
    """Calculate Eye Aspect Ratio (EAR) - requires dlib landmarks"""
    if not DLIB_AVAILABLE:
        return 0.3  # Default value when dlib is not available
    
    # Compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    
    # Compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def detect_eyes_enhanced_opencv(gray, face):
    """Enhanced eye detection using multiple cascade classifiers"""
    x, y, w, h = face
    roi_gray = gray[y:y+h, x:x+w]
    
    # Try multiple eye detection approaches
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eye_tree_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
    
    # Detect eyes with standard cascade
    eyes1 = eye_cascade.detectMultiScale(roi_gray, 1.1, 5, minSize=(20, 20))
    # Detect eyes with tree cascade (better for glasses)
    eyes2 = eye_tree_cascade.detectMultiScale(roi_gray, 1.1, 5, minSize=(20, 20))
    
    # Combine detections
    all_eyes = []
    for eye_set in [eyes1, eyes2]:
        for (ex, ey, ew, eh) in eye_set:
            # Convert back to full image coordinates
            all_eyes.append((ex + x, ey + y, ew, eh))
    
    # Remove duplicate detections
    if len(all_eyes) > 1:
        # Simple deduplication based on distance
        filtered_eyes = []
        for eye in all_eyes:
            is_duplicate = False
            for existing_eye in filtered_eyes:
                dist = np.sqrt((eye[0] - existing_eye[0])**2 + (eye[1] - existing_eye[1])**2)
                if dist < 30:  # If eyes are too close, consider as duplicate
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered_eyes.append(eye)
        return filtered_eyes
    
    return all_eyes

def calculate_mouth_opening(face_gray):
    """Estimate mouth opening using edge detection (fallback yawn detection)"""
    # Focus on lower part of face for mouth detection
    h, w = face_gray.shape
    mouth_region = face_gray[int(h*0.6):int(h*0.9), int(w*0.3):int(w*0.7)]
    
    # Apply edge detection
    edges = cv2.Canny(mouth_region, 50, 150)
    
    # Count horizontal edges (indicating mouth opening)
    horizontal_edges = 0
    for i in range(1, edges.shape[0]-1):
        for j in range(1, edges.shape[1]-1):
            if edges[i, j] > 0:
                # Check if it's a horizontal edge
                if edges[i-1, j] > 0 or edges[i+1, j] > 0:
                    horizontal_edges += 1
    
    # Normalize by region size
    mouth_opening = horizontal_edges / (mouth_region.shape[0] * mouth_region.shape[1])
    return mouth_opening

def estimate_head_pose_basic(face, frame_shape):
    """Basic head pose estimation using face position"""
    x, y, w, h = face
    frame_h, frame_w = frame_shape[:2]
    
    # Calculate face center
    face_center_x = x + w // 2
    face_center_y = y + h // 2
    
    # Calculate deviation from frame center
    frame_center_x = frame_w // 2
    frame_center_y = frame_h // 2
    
    # Calculate horizontal and vertical deviation as percentage
    horizontal_deviation = (face_center_x - frame_center_x) / frame_center_x * 100
    vertical_deviation = (face_center_y - frame_center_y) / frame_center_y * 100
    
    # Simple thresholds for head pose
    looking_away = abs(horizontal_deviation) > 25 or abs(vertical_deviation) > 20
    
    return looking_away, horizontal_deviation, vertical_deviation
    """Calculate Eye Aspect Ratio (EAR)"""
    # Compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    
    # Compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_mar(mouth):
    """Calculate Mouth Aspect Ratio (MAR) for yawn detection"""
    # Compute the euclidean distances between the vertical mouth landmarks
    A = dist.euclidean(mouth[2], mouth[10])  # 51, 59
    B = dist.euclidean(mouth[4], mouth[8])   # 53, 57
    
    # Compute the euclidean distance between the horizontal mouth landmarks
    C = dist.euclidean(mouth[0], mouth[6])   # 49, 55
    
    # Compute the mouth aspect ratio
    mar = (A + B) / (2.0 * C)
    return mar

def detect_head_pose(landmarks, img_size):
    """Detect head pose to identify if person is looking away"""
    # 3D model points
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])
    
    # 2D image points from facial landmarks
    image_points = np.array([
        landmarks[30],     # Nose tip
        landmarks[8],      # Chin
        landmarks[36],     # Left eye left corner
        landmarks[45],     # Right eye right corner
        landmarks[48],     # Left mouth corner
        landmarks[54]      # Right mouth corner
    ], dtype="double")
    
    # Camera internals
    focal_length = img_size[1]
    center = (img_size[1]/2, img_size[0]/2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double")
    
    dist_coeffs = np.zeros((4,1))  # Assuming no lens distortion
    
    # Solve PnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs)
    
    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    # Calculate Euler angles
    sy = np.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] + 
                 rotation_matrix[1,0] * rotation_matrix[1,0])
    
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2])
        y = np.arctan2(-rotation_matrix[2,0], sy)
        z = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
    else:
        x = np.arctan2(-rotation_matrix[1,2], rotation_matrix[1,1])
        y = np.arctan2(-rotation_matrix[2,0], sy)
        z = 0
    
    # Convert to degrees
    angles = np.array([x, y, z]) * 180.0 / np.pi
    return angles

def calibrate_thresholds(detector, predictor, cap, duration=5):
    """Calibrate personalized thresholds"""
    print("Calibration starting... Please look at the camera normally for 5 seconds")
    start_time = time.time()
    ear_values = []
    
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            continue
            
        if 'imutils' in globals():
            frame = imutils.resize(frame, width=800)
        else:
            frame = cv2.resize(frame, (800, 600))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        rects = detector(gray, 0)
        
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            leftEye = shape[face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][0]:
                           face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][1]]
            rightEye = shape[face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][0]:
                            face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][1]]
            
            leftEAR = calculate_ear(leftEye)
            rightEAR = calculate_ear(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            ear_values.append(ear)
        
        cv2.putText(frame, f"Calibrating... {duration - (time.time() - start_time):.1f}s", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Calibration", frame)
        cv2.waitKey(1)
    
    cv2.destroyWindow("Calibration")
    
    if ear_values:
        avg_ear = np.mean(ear_values)
        threshold = avg_ear * 0.8  # 20% below average
        print(f"Calibration complete. Personal EAR threshold: {threshold:.3f}")
        return threshold
    else:
        print("Calibration failed. Using default threshold.")
        return EYE_AR_THRESH
def play_alarm():
    """Enhanced alarm system with multiple alert types"""
    global alarm_playing
    alert_count = 0
    while alarm_playing and alert_count < 10:  # Limit alarm duration
        winsound.PlaySound("SystemExclamation", winsound.SND_ASYNC)
        time.sleep(0.3)  
        winsound.Beep(1000, 300)  
        time.sleep(0.3)
        winsound.Beep(1500, 300)
        time.sleep(0.4)
        alert_count += 1

def main():
    global alarm_playing, drowsy_frames
    
    # Initialize detection based on available libraries
    use_dlib = DLIB_AVAILABLE
    
    if use_dlib:
        try:
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            print("dlib facial landmark predictor loaded successfully")
        except:
            print("Could not load dlib shape predictor.")
            print("  Download 'shape_predictor_68_face_landmarks.dat' or run setup_models.py")
            print("  Falling back to basic detection...")
            use_dlib = False
    
    if not use_dlib:
        print("Using enhanced OpenCV detection (fallback mode)")
        # Initialize cascade classifiers for fallback
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    alarm_playing = False
    alarm_thread = None
    cap = cv2.VideoCapture(0)
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if use_dlib:
        # Calibrate personalized thresholds
        personal_threshold = calibrate_thresholds(detector, predictor, cap)
    else:
        personal_threshold = EYE_AR_THRESH
        print(f"Using default EAR threshold: {personal_threshold}")

    eyes_closed_time = 0
    last_eye_detection_time = time.time()
    alert_duration = 0
    frame_count = 0
    start_time = time.time()
    
    print("Drowsiness detection started. Press 'q' to quit", end="")
    if use_dlib:
        print(", 'c' to recalibrate.")
    else:
        print(".")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=800) if 'imutils' in globals() else cv2.resize(frame, (800, 600))
        frame_count += 1
        current_time = time.time()
        
        if use_dlib:
            # Enhanced detection using dlib
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            
            eyes_detected = False
            current_ear = 0
            head_pose_alert = False
            yawn_detected = False
            
            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                
                # Extract eye regions
                leftEye = shape[face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][0]:
                               face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][1]]
                rightEye = shape[face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][0]:
                                face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][1]]
                
                # Extract mouth region for yawn detection
                mouth = shape[face_utils.FACIAL_LANDMARKS_IDXS["mouth"][0]:
                             face_utils.FACIAL_LANDMARKS_IDXS["mouth"][1]]
                
                # Calculate EAR for both eyes
                leftEAR = calculate_ear(leftEye)
                rightEAR = calculate_ear(rightEye)
                current_ear = (leftEAR + rightEAR) / 2.0
                
                # Calculate MAR for yawn detection
                mar = calculate_mar(mouth)
                if mar > YAWN_THRESH:
                    yawn_detected = True
                
                # Detect head pose
                try:
                    head_angles = detect_head_pose(shape, frame.shape)
                    if abs(head_angles[1]) > 30 or abs(head_angles[0]) > 20:  # Head turned away
                        head_pose_alert = True
                except:
                    pass
                
                # Draw facial landmarks
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                mouthHull = cv2.convexHull(mouth)
                
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
                
                # Check for drowsiness using EAR
                if current_ear < personal_threshold:
                    drowsy_frames += 1
                    if drowsy_frames >= EYE_AR_CONSEC_FRAMES:
                        eyes_detected = False
                    else:
                        eyes_detected = True
                else:
                    eyes_detected = True
                    drowsy_frames = 0
                
                break  # Process only the first detected face
            
            # Decision logic for alerts (without head pose)
            drowsiness_detected = (not eyes_detected or 
                                 drowsy_frames >= EYE_AR_CONSEC_FRAMES or 
                                 yawn_detected)
        
        else:
            # Enhanced fallback detection using OpenCV
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))
            
            eyes_detected = False
            current_ear = 0
            head_pose_alert = False
            yawn_detected = False
            
            for (x, y, w, h) in faces:
                # Draw face rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Enhanced eye detection
                detected_eyes = detect_eyes_enhanced_opencv(gray, (x, y, w, h))
                
                # Check for head pose using face position
                looking_away, h_dev, v_dev = estimate_head_pose_basic((x, y, w, h), frame.shape)
                head_pose_alert = looking_away
                
                # Mouth opening detection for yawn
                face_gray = gray[y:y+h, x:x+w]
                mouth_opening = calculate_mouth_opening(face_gray)
                if mouth_opening > 0.02:  # Threshold for yawn detection
                    yawn_detected = True
                
                if len(detected_eyes) >= 2:
                    eyes_detected = True
                    drowsy_frames = 0
                    
                    # Calculate simplified EAR for available eyes
                    ear_values = []
                    for (ex, ey, ew, eh) in detected_eyes[:2]:  # Use first two eyes
                        cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                        
                        # Simple EAR based on eye dimensions
                        if ew > 0:
                            ear = eh / ew  # height/width ratio
                            ear_values.append(ear * 0.3)  # Normalize to typical EAR range
                    
                    if ear_values:
                        current_ear = np.mean(ear_values)
                        
                        # Check for drowsiness based on EAR
                        if current_ear < personal_threshold * 0.7:  # Adjusted threshold
                            drowsy_frames += 1
                            if drowsy_frames < EYE_AR_CONSEC_FRAMES:
                                eyes_detected = True
                            else:
                                eyes_detected = False
                else:
                    # No eyes detected
                    drowsy_frames += 1
                    current_ear = 0
                
                # Display detection info on face
                cv2.putText(frame, f"Eyes: {len(detected_eyes)}", (x, y-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                break  # Process only the first detected face
            
            # Decision logic for alerts (without head pose - adjusted for fallback mode)
            drowsiness_detected = (not eyes_detected or 
                                 drowsy_frames >= EYE_AR_CONSEC_FRAMES or 
                                 yawn_detected)
        
        # Handle alerts (same for both modes)
        if drowsiness_detected:
            if not eyes_detected:
                eyes_closed_time = current_time - last_eye_detection_time + 2
            
            if eyes_closed_time > 2 or drowsy_frames >= EYE_AR_CONSEC_FRAMES:
                if not alarm_playing:
                    alarm_playing = True
                    alarm_thread = threading.Thread(target=play_alarm)
                    alarm_thread.start()
                
                # Visual alert
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                
                alert_duration += 1
                if alert_duration % 20 < 10:  # Blinking text
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    cv2.putText(frame, "WAKE UP!", (10, 100),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            eyes_closed_time = 0
            last_eye_detection_time = current_time
            if alarm_playing:
                alarm_playing = False
                if alarm_thread and alarm_thread.is_alive():
                    alarm_thread.join(timeout=1)
                winsound.PlaySound(None, winsound.SND_ASYNC)
                alert_duration = 0
        
        # Display core feature statistics
        if use_dlib and current_ear > 0:
            cv2.putText(frame, f"EAR: {current_ear:.2f}", (frame.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
        cv2.imshow("Drowsiness Detection - Core Features", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and use_dlib:
            # Recalibrate thresholds
            personal_threshold = calibrate_thresholds(detector, predictor, cap)
    
    # Cleanup
    alarm_playing = False
    if alarm_thread and alarm_thread.is_alive():
        alarm_thread.join(timeout=2)
    cap.release()
    cv2.destroyAllWindows()
    print(f"Session ended.")
    print(f"Core features used: Facial Landmarks, Eye Detection (EAR), Yawn Detection (MAR), Head Pose Estimation")

if __name__ == "__main__":
    main() 