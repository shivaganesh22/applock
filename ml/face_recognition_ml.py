"""
Face Recognition ML Module
===========================
Uses OpenCV Haar Cascades for face detection and 
LBPH (Local Binary Pattern Histogram) for robust face recognition.

Integrated from user-provided working LBPH script.
"""

import cv2
import numpy as np
import os
import logging
import time

logger = logging.getLogger(__name__)

class FaceRecognitionModel:
    """Face recognition using OpenCV Haar Cascades and LBPH face recognizer."""

    def __init__(self, models_dir='trained_models', data_dir='user_data'):
        self.models_dir = models_dir
        self.data_dir = data_dir
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        # Load the Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_classifier = cv2.CascadeClassifier(cascade_path)
        
        if self.face_classifier.empty():
            logger.error("Failed to load Haar Cascade XML file!")

    def _face_extractor(self, img):
        """Detect and extract the face from an image. Returns grayscale 200x200 ROI or None."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        if len(faces) == 0:
            return None, None
        
        # Take the first detected face
        (x, y, w, h) = faces[0]
        cropped_face = img[y:y+h, x:x+w]
        
        # Resize to standard 200x200 size and convert to grayscale
        face_resized = cv2.resize(cropped_face, (200, 200))
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        
        return face_gray, (x, y, w, h)

    def capture_and_train(self, user_id, num_samples=20):
        """
        Open a cv2 window to capture face samples and train the LBPH model.
        Returns (success: bool, message: str)
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return False, "Could not open camera."

        user_dir = os.path.join(self.data_dir, f'face_{user_id}')
        os.makedirs(user_dir, exist_ok=True)

        Training_data = []
        Labels = []
        collected = 0
        auto_capture = False
        last_capture_time = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            
            face_gray, box = self._face_extractor(frame)
            
            # Draw UI
            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 50), (20, 20, 30), -1)
            cv2.putText(display_frame, f"FACE ENROLLMENT  |  Samples: {collected}/{num_samples}",
                       (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
            
            if face_gray is not None:
                x, y, w, h = box
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(display_frame, "Face Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Bottom instructions
                if not auto_capture:
                    cv2.putText(display_frame, "Press 'c' to start auto-capture | 'q' to quit", 
                               (30, display_frame.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    cv2.putText(display_frame, "CAPTURING... Slowly move head", 
                               (30, display_frame.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Auto capture logic
                    now = time.time()
                    if now - last_capture_time > 0.3: # every 300ms
                        Training_data.append(np.asarray(face_gray, dtype=np.uint8))
                        Labels.append(1) # We use label 1 for the user
                        collected += 1
                        last_capture_time = now
                        
                        # Save file (optional, just for records/debugging)
                        file_name_path = os.path.join(user_dir, f"{collected}.jpg")
                        cv2.imwrite(file_name_path, face_gray)
                        
                        # Flash screen
                        cv2.rectangle(display_frame, (0,0), (display_frame.shape[1], display_frame.shape[0]), (0,255,0), 5)
                        
                        if collected >= num_samples:
                            break
            else:
                cv2.putText(display_frame, "No Face Detected", 
                           (30, display_frame.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
            cv2.imshow("AppLock - Face Registration", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                auto_capture = True
                last_capture_time = time.time()
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if collected < num_samples:
            return False, f"Enrollment cancelled. Only {collected} samples collected."

        # --- Train LBPH Model ---
        try:
            model = cv2.face.LBPHFaceRecognizer_create()
            model.train(np.asarray(Training_data), np.asarray(Labels, dtype=np.int32))
            
            # Save the trained xml file
            model_path = os.path.join(self.models_dir, f'face_model_{user_id}.xml')
            model.write(model_path)
            
            # Also save info dict to be compatible with Flask routes
            info_dict = {
                'model_path': model_path,
                'n_samples': collected,
                'model_type': 'LBPH Face Recognizer',
                'feature_type': 'Haar Cascades + LBPH'
            }
            import json
            with open(os.path.join(user_dir, 'model_info.json'), 'w') as f:
                json.dump(info_dict, f)
                
            logger.info(f"LBPH model trained for user {user_id} and saved to {model_path}")
            return True, f"Face successfully enrolled! ({collected} samples)"
        except AttributeError:
            logger.error("cv2.face module not found. Please run: pip install opencv-contrib-python")
            return False, "Server missing opencv-contrib-python package for LBPH."

    def verify_with_camera(self, user_id, timeout=60):
        """
        Open cv2 window for face verification using LBPH model.
        Returns (success: bool, confidence_pct: float, message: str)
        """
        model_path = os.path.join(self.models_dir, f'face_model_{user_id}.xml')
        if not os.path.exists(model_path):
            return False, 0.0, "No face model found. Please enroll first."

        try:
            model = cv2.face.LBPHFaceRecognizer_create()
            model.read(model_path)
        except AttributeError:
            return False, 0.0, "Server missing opencv-contrib-python package."
        except Exception as e:
            return False, 0.0, f"Failed to load model: {e}"

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return False, 0.0, "Could not open camera."

        start_time = time.time()
        best_confidence = 0.0
        consecutive_matches = 0
        REQUIRED_MATCHES = 3  # Need 3 good frames in a row
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            display = frame.copy()
            elapsed = time.time() - start_time
            
            face_gray, box = self._face_extractor(frame)
            
            # Top bar
            cv2.rectangle(display, (0, 0), (display.shape[1], 50), (20, 20, 30), -1)
            remaining = max(0, timeout - elapsed)
            cv2.putText(display, f"FACE VERIFICATION  |  Time: {remaining:.0f}s",
                       (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)

            if face_gray is not None:
                x, y, w, h = box
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 255), 2)
                
                # Predict
                label, distance = model.predict(face_gray)
                
                # Convert distance to confidence percentage (user formula)
                if distance < 500:
                    confidence = int(100 * (1 - (distance) / 300))
                else:
                    confidence = 0
                    
                confidence = max(0, min(100, confidence)) # clamp between 0-100
                best_confidence = max(best_confidence, confidence)

                if confidence >= 83:
                    consecutive_matches += 1
                    cv2.putText(display, f"{confidence}% Match", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    consecutive_matches = 0
                    cv2.putText(display, f"Unknown ({confidence}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                if consecutive_matches >= REQUIRED_MATCHES:
                    # Success
                    success_frame = display.copy()
                    cv2.rectangle(success_frame, (0, 0), (success_frame.shape[1], success_frame.shape[0]), (0, 255, 0), 8)
                    cv2.putText(success_frame, "FACE VERIFIED!",
                               (success_frame.shape[1]//2 - 140, success_frame.shape[0]//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 4)
                    cv2.imshow('AppLock - Verification', success_frame)
                    cv2.waitKey(1500)
                    cap.release()
                    cv2.destroyAllWindows()
                    # Return confidence as 0-1 float to match Flask route expectation
                    return True, best_confidence / 100.0, "Face verified successfully!"
            else:
                consecutive_matches = 0
                cv2.putText(display, "No face found", (30, display.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow('AppLock - Verification', display)
            
            if cv2.waitKey(1) & 0xFF == ord('q') or elapsed > timeout:
                break

        cap.release()
        cv2.destroyAllWindows()
        return False, best_confidence / 100.0, f"Verification failed (best match: {best_confidence}%)"

    def get_model_info(self, user_id):
        """Get information about a trained face model."""
        user_dir = os.path.join(self.data_dir, f'face_{user_id}')
        info_path = os.path.join(user_dir, 'model_info.json')
        
        if os.path.exists(info_path):
            import json
            with open(info_path, 'r') as f:
                return json.load(f)
        return None
