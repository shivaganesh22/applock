"""
Hand Gesture Recognition ML Module
====================================
Uses MediaPipe Hands to detect hand landmarks and classify gestures
using a rule-based approach (finger state detection).

Users create a GESTURE SEQUENCE pattern (like a password) by showing
gestures one at a time. The sequence is then used for verification.

Capture happens via OpenCV cv2.imshow window.
"""

import cv2
import numpy as np
import mediapipe as mp
import json
import os
import logging
import time
import hashlib

logger = logging.getLogger(__name__)

# MediaPipe solutions
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def recognize_gesture(hand_landmarks):
    """
    Rule-based gesture recognition from MediaPipe hand landmarks.
    Detects which fingers are extended and classifies into gestures.
    Returns gesture name string or None.
    """
    landmarks = hand_landmarks.landmark

    # Get finger tip and pip positions
    tips = [4, 8, 12, 16, 20]      # Thumb, Index, Middle, Ring, Pinky tips
    pips = [3, 6, 10, 14, 18]      # Corresponding PIP joints
    mcps = [2, 5, 9, 13, 17]       # MCP joints

    # Determine handedness by comparing thumb tip x with wrist x
    wrist_x = landmarks[0].x
    thumb_tip_x = landmarks[4].x
    is_right_hand = thumb_tip_x < wrist_x  # In mirrored video

    fingers = []

    # Thumb: compare tip x with IP joint x
    if is_right_hand:
        fingers.append(1 if landmarks[4].x < landmarks[3].x else 0)
    else:
        fingers.append(1 if landmarks[4].x > landmarks[3].x else 0)

    # Other 4 fingers: tip y < pip y means extended (y-axis is inverted in image)
    for tip, pip in zip(tips[1:], pips[1:]):
        fingers.append(1 if landmarks[tip].y < landmarks[pip].y else 0)

    total_up = sum(fingers)
    thumb, index, middle, ring, pinky = fingers

    # Classify gesture based on finger states
    if total_up == 0:
        return "Fist"
    elif total_up == 5:
        return "Open Palm"
    elif fingers == [0, 1, 1, 0, 0]:
        return "Peace"
    elif thumb == 1 and total_up == 1:
        return "Thumbs Up"
    elif index == 1 and total_up == 1:
        return "Pointing Up"
    elif fingers == [0, 1, 0, 0, 1] or fingers == [1, 1, 0, 0, 1]:
        return "Rock"
    elif fingers == [1, 0, 0, 0, 1]:
        return "Call Me"
    elif fingers == [0, 1, 1, 1, 0]:
        return "Three"
    elif fingers == [0, 1, 1, 1, 1]:
        return "Four"
    elif index == 1 and middle == 1 and ring == 0 and pinky == 0 and thumb == 1:
        return "Three Alt"
    elif thumb == 1 and index == 1 and total_up == 2:
        return "L Shape"
    elif middle == 1 and total_up == 1:
        return "Middle Finger"
    elif ring == 1 and total_up == 1:
        return "Ring Finger"
    elif pinky == 1 and total_up == 1:
        return "Pinky"
    elif total_up == 2 and index == 1 and pinky == 1:
        return "Spider"
    elif total_up == 2 and index == 1 and middle == 1:
        return "Peace"
    elif total_up == 3 and thumb == 0 and pinky == 0:
        return "Three Middle"
    else:
        return f"{total_up} Fingers"


# Emoji/symbol for each gesture
GESTURE_ICONS = {
    "Fist": "✊",
    "Open Palm": "🖐",
    "Peace": "✌",
    "Thumbs Up": "👍",
    "Pointing Up": "☝",
    "Rock": "🤘",
    "Call Me": "🤙",
    "Three": "3️⃣",
    "Four": "4️⃣",
    "Three Alt": "🔱",
    "L Shape": "👆",
    "Middle Finger": "🖕",
    "Ring Finger": "💍",
    "Pinky": "🤏",
    "Spider": "🕷",
    "Three Middle": "🤟",
}

# Colors for different gestures (BGR)
GESTURE_COLORS = {
    "Fist": (60, 60, 220),
    "Open Palm": (60, 220, 60),
    "Peace": (220, 180, 60),
    "Thumbs Up": (60, 180, 220),
    "Pointing Up": (220, 60, 180),
    "Rock": (180, 60, 220),
    "Call Me": (60, 220, 180),
}


class GestureRecognitionModel:
    """Gesture-based authentication using gesture sequence patterns."""

    def __init__(self, models_dir='trained_models', data_dir='user_data'):
        self.models_dir = models_dir
        self.data_dir = data_dir
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

    def capture_and_train(self, user_id):
        """
        Open a cv2 window where user creates a gesture sequence pattern.
        - User shows a gesture → it's recognized and displayed
        - Press 'C' to capture current gesture into the sequence
        - Repeat to build a sequence (minimum 1 gesture)
        - Press 'S' or 'Q' to stop and save

        Returns (success: bool, message: str)
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return False, "Could not open camera."

        user_dir = os.path.join(self.data_dir, f'gesture_{user_id}')
        os.makedirs(user_dir, exist_ok=True)

        gesture_sequence = []
        current_gesture = None
        gesture_stable_time = 0
        last_gesture = None
        STABLE_THRESHOLD = 0.5  # Gesture must be stable for 0.5s before capture

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        ) as hands:

            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                display = frame.copy()
                h, w = display.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                hand_detected = False
                detected_gesture = None

                if results.multi_hand_landmarks:
                    hand_detected = True
                    for hand_lm in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            display, hand_lm,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                        detected_gesture = recognize_gesture(hand_lm)

                    # Track gesture stability
                    if detected_gesture == last_gesture:
                        gesture_stable_time += 1/30.0  # ~30fps
                    else:
                        gesture_stable_time = 0
                    last_gesture = detected_gesture

                    if gesture_stable_time >= STABLE_THRESHOLD:
                        current_gesture = detected_gesture

                # ====== DRAW UI ======

                # Top bar: Title + sequence display
                bar_h = 70 if gesture_sequence else 50
                cv2.rectangle(display, (0, 0), (w, bar_h), (20, 20, 35), -1)

                cv2.putText(display, f"GESTURE ENROLLMENT  |  Sequence: {len(gesture_sequence)} gesture(s)",
                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 1)

                # Show current sequence at top
                if gesture_sequence:
                    seq_text = " -> ".join(gesture_sequence)
                    icon_text = " ".join([GESTURE_ICONS.get(g, "?") for g in gesture_sequence])
                    cv2.putText(display, f"Pattern: {seq_text}",
                               (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 255, 200), 1)

                # Current detected gesture (large, center)
                if current_gesture and hand_detected:
                    icon = GESTURE_ICONS.get(current_gesture, "?")
                    color = GESTURE_COLORS.get(current_gesture, (200, 200, 200))

                    # Gesture label
                    label = f"Detected: {current_gesture} {icon}"
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                    text_x = (w - text_size[0]) // 2
                    cv2.putText(display, label,
                               (text_x, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                    # Stability indicator
                    if gesture_stable_time >= STABLE_THRESHOLD:
                        cv2.putText(display, "[READY - Press 'C' to capture]",
                                   (text_x - 20, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                elif not hand_detected:
                    cv2.putText(display, "Show your hand to camera",
                               (w // 2 - 140, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Bottom bar: Instructions
                cv2.rectangle(display, (0, h - 40), (w, h), (20, 20, 35), -1)

                if len(gesture_sequence) >= 1:
                    cv2.putText(display, "'C' = Capture  |  'Z' = Undo last  |  'S' = Stop & Save  |  'Q' = Quit",
                               (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 200, 150), 1)
                else:
                    cv2.putText(display, "Show a gesture, then press 'C' to add to sequence  |  'Q' = Quit",
                               (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 200, 150), 1)

                # Progress dots (right side)
                for i, g in enumerate(gesture_sequence):
                    dot_color = GESTURE_COLORS.get(g, (200, 200, 200))
                    cx = w - 25
                    cy = bar_h + 20 + i * 30
                    cv2.circle(display, (cx, cy), 10, dot_color, -1)
                    cv2.putText(display, f"{i+1}:{g[:6]}", (cx - 80, cy + 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, dot_color, 1)

                cv2.imshow("AppLock - Gesture Enrollment", display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('c') and current_gesture and hand_detected:
                    # Capture current gesture into sequence
                    gesture_sequence.append(current_gesture)
                    # Flash green border
                    flash_frame = display.copy()
                    cv2.rectangle(flash_frame, (0, 0), (w-1, h-1), (0, 255, 0), 8)
                    text = f"Captured: {current_gesture}! ({len(gesture_sequence)} in sequence)"
                    ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.putText(flash_frame, text, ((w - ts[0])//2, h//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("AppLock - Gesture Enrollment", flash_frame)
                    cv2.waitKey(500)
                    current_gesture = None
                    gesture_stable_time = 0

                elif key == ord('z') and gesture_sequence:
                    # Undo last gesture
                    removed = gesture_sequence.pop()
                    current_gesture = None
                    gesture_stable_time = 0

                elif key == ord('s'):
                    # Stop and save
                    if len(gesture_sequence) >= 1:
                        break
                    else:
                        # Show warning
                        cv2.putText(display, "Need at least 1 gesture!",
                                   (w//2 - 120, h//2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        cv2.imshow("AppLock - Gesture Enrollment", display)
                        cv2.waitKey(1000)

                elif key == ord('q'):
                    if len(gesture_sequence) >= 1:
                        break
                    else:
                        cap.release()
                        cv2.destroyAllWindows()
                        return False, "Enrollment cancelled. No gestures captured."

        cap.release()
        cv2.destroyAllWindows()

        if len(gesture_sequence) < 1:
            return False, "Need at least 1 gesture in the sequence."

        # Save the gesture sequence
        return self._save_sequence(user_id, gesture_sequence)

    def _save_sequence(self, user_id, gesture_sequence):
        """Save the gesture sequence as the user's gesture password."""
        user_dir = os.path.join(self.data_dir, f'gesture_{user_id}')
        os.makedirs(user_dir, exist_ok=True)

        # Hash the sequence for secure storage
        seq_str = '|'.join(gesture_sequence)
        seq_hash = hashlib.sha256(seq_str.encode()).hexdigest()

        data = {
            'sequence': gesture_sequence,
            'sequence_hash': seq_hash,
            'sequence_length': len(gesture_sequence),
            'sequence_display': ' → '.join(gesture_sequence),
        }

        meta_path = os.path.join(user_dir, 'metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Gesture sequence enrolled for user {user_id}: {len(gesture_sequence)} gestures")
        return True, f"Gesture sequence enrolled! ({len(gesture_sequence)} gestures: {' → '.join(gesture_sequence)})"

    def verify_with_camera(self, user_id, timeout=30):
        """
        Open cv2 window for gesture sequence verification.
        User shows gestures from memory (press C to submit each).
        System checks internally without revealing the expected pattern.
        Returns (success: bool, confidence: float, message: str)
        """
        user_dir = os.path.join(self.data_dir, f'gesture_{user_id}')
        meta_path = os.path.join(user_dir, 'metadata.json')

        if not os.path.exists(meta_path):
            return False, 0.0, "No gesture sequence enrolled. Please enroll first."

        with open(meta_path) as f:
            data = json.load(f)

        target_sequence = data['sequence']
        total_gestures = len(target_sequence)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return False, 0.0, "Could not open camera."

        current_step = 0
        current_gesture = None
        gesture_stable_time = 0
        last_gesture = None
        STABLE_THRESHOLD = 0.5
        start_time = time.time()
        submitted_gestures = []
        flash_msg = None
        flash_time = 0
        mistakes = 0
        MAX_MISTAKES = 3

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        ) as hands:

            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                display = frame.copy()
                h, w = display.shape[:2]
                elapsed = time.time() - start_time

                results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                hand_detected = False
                detected_gesture = None

                if results.multi_hand_landmarks:
                    hand_detected = True
                    for hand_lm in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            display, hand_lm,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                        detected_gesture = recognize_gesture(hand_lm)

                    if detected_gesture == last_gesture:
                        gesture_stable_time += 1/30.0
                    else:
                        gesture_stable_time = 0
                    last_gesture = detected_gesture

                    if gesture_stable_time >= STABLE_THRESHOLD:
                        current_gesture = detected_gesture

                # ====== DRAW UI ======

                # Top bar
                bar_h = 55
                cv2.rectangle(display, (0, 0), (w, bar_h), (20, 20, 35), -1)

                remaining = max(0, timeout - elapsed)
                cv2.putText(display, f"GESTURE VERIFICATION  |  Gesture {current_step + 1} of {total_gestures}  |  Time: {remaining:.0f}s",
                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 255), 1)

                # Show progress dots (without revealing gesture names)
                dot_y = 45
                for i in range(total_gestures):
                    dot_x = 10 + i * 25
                    if i < current_step:
                        # Matched — green filled circle
                        cv2.circle(display, (dot_x + 8, dot_y), 8, (0, 255, 0), -1)
                        cv2.putText(display, "?", (dot_x + 4, dot_y + 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (20, 20, 35), 1)
                    elif i == current_step:
                        # Current — yellow outline
                        cv2.circle(display, (dot_x + 8, dot_y), 8, (0, 220, 255), 2)
                    else:
                        # Upcoming — gray outline
                        cv2.circle(display, (dot_x + 8, dot_y), 8, (80, 80, 100), 1)

                # Current detected gesture (large, center)
                if current_gesture and hand_detected:
                    icon = GESTURE_ICONS.get(current_gesture, "?")
                    color = GESTURE_COLORS.get(current_gesture, (200, 200, 200))

                    label = f"Detected: {current_gesture} {icon}"
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                    text_x = (w - text_size[0]) // 2
                    cv2.putText(display, label,
                               (text_x, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                    if gesture_stable_time >= STABLE_THRESHOLD:
                        cv2.putText(display, "[READY - Press 'C' to submit]",
                                   (text_x - 20, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                elif not hand_detected:
                    cv2.putText(display, "Show your hand to the camera",
                               (w // 2 - 160, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Flash message (correct/incorrect feedback)
                if flash_msg and (time.time() - flash_time < 1.0):
                    msg_text, msg_color = flash_msg
                    ts = cv2.getTextSize(msg_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    cv2.putText(display, msg_text, ((w - ts[0]) // 2, h // 2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, msg_color, 2)
                else:
                    flash_msg = None

                # Bottom bar
                cv2.rectangle(display, (0, h - 40), (w, h), (20, 20, 35), -1)
                cv2.putText(display, "Show your gesture and press 'C' to submit  |  'Q' = Quit",
                           (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 200, 150), 1)

                cv2.imshow("AppLock - Gesture Verification", display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('c') and current_gesture and hand_detected:
                    # User submits current gesture — check against target
                    target = target_sequence[current_step]
                    submitted_gestures.append(current_gesture)

                    if current_gesture == target:
                        current_step += 1
                        flash_msg = (f"Gesture {current_step}/{total_gestures} correct!", (0, 255, 0))
                        flash_time = time.time()

                        # Flash green border
                        cv2.rectangle(display, (0, 0), (w-1, h-1), (0, 255, 0), 8)
                        cv2.imshow("AppLock - Gesture Verification", display)
                        cv2.waitKey(400)

                        if current_step >= total_gestures:
                            # All gestures matched!
                            success_frame = display.copy()
                            cv2.rectangle(success_frame, (0, 0), (w-1, h-1), (0, 255, 0), 8)
                            cv2.putText(success_frame, "ALL GESTURES VERIFIED!",
                                       (w//2 - 180, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                            cv2.imshow("AppLock - Gesture Verification", success_frame)
                            cv2.waitKey(1500)
                            break
                    else:
                        # Wrong gesture — add mistake and let them retry
                        mistakes += 1
                        attempts_left = MAX_MISTAKES - mistakes
                        
                        flash_frame = display.copy()
                        cv2.rectangle(flash_frame, (0, 0), (w-1, h-1), (0, 0, 255), 8)
                        cv2.putText(flash_frame, f"WRONG GESTURE! ({attempts_left} tries left)",
                                   (w//2 - 180, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                        cv2.imshow("AppLock - Gesture Verification", flash_frame)
                        cv2.waitKey(1500)
                        
                        if mistakes >= MAX_MISTAKES:
                            break

                    current_gesture = None
                    gesture_stable_time = 0

                elif key == ord('q') or elapsed > timeout:
                    break

        cap.release()
        cv2.destroyAllWindows()

        confidence = current_step / total_gestures if total_gestures > 0 else 0

        if current_step >= total_gestures:
            return True, confidence, f"Gesture sequence verified! ({total_gestures} gestures matched)"
        else:
            return False, confidence, f"Verification failed at gesture {current_step + 1}/{total_gestures}"

    def get_model_info(self, user_id):
        """Get information about enrolled gesture sequence."""
        user_dir = os.path.join(self.data_dir, f'gesture_{user_id}')
        meta_path = os.path.join(user_dir, 'metadata.json')

        if not os.path.exists(meta_path):
            return None

        with open(meta_path) as f:
            data = json.load(f)

        return {
            'model_type': 'Gesture Sequence (Rule-based Recognition)',
            'feature_type': 'MediaPipe Hand Landmarks (21 points)',
            'gesture_name': data.get('sequence_display', ''),
            'gesture_display': data.get('sequence_display', ''),
            'sequence_length': data.get('sequence_length', 0),
            'sequence': data.get('sequence', [])
        }
