"""
Unlock / Verification Routes
==============================
Handles verification for all 5 authentication methods during login.

Face and Gesture verification now open OpenCV cv2 windows.
"""

from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, session
from flask_login import login_user, current_user
from models import db, User, AuthMethod
from werkzeug.security import check_password_hash
from config import Config
import json
import hashlib
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

unlock_bp = Blueprint('unlock', __name__, url_prefix='/unlock')


@unlock_bp.route('/')
def choose():
    """Show available unlock methods for the pending user."""
    user_id = session.get('pending_user_id')
    if not user_id:
        flash('Please log in first.', 'error')
        return redirect(url_for('auth.login'))

    user = User.query.get(user_id)
    if not user:
        flash('User not found.', 'error')
        return redirect(url_for('auth.login'))

    enrolled = user.get_enrolled_methods()
    methods = []
    for m in enrolled:
        method_info = {'type': m.method_type, 'enrolled_at': m.enrolled_at}
        if m.method_type == 'pin':
            method_info.update({'name': 'PIN Code', 'icon': 'fa-key', 'color': '#6c5ce7'})
        elif m.method_type == 'pattern':
            method_info.update({'name': 'Pattern Lock', 'icon': 'fa-th', 'color': '#00cec9'})
        elif m.method_type == 'face':
            method_info.update({'name': 'Face ID', 'icon': 'fa-smile', 'color': '#e17055', 'ml': True})
        elif m.method_type == 'gesture':
            method_info.update({'name': 'Hand Gesture', 'icon': 'fa-hand-paper', 'color': '#fdcb6e', 'ml': True})
        elif m.method_type == 'fingerprint':
            method_info.update({'name': 'Fingerprint', 'icon': 'fa-fingerprint', 'color': '#55efc4'})
        methods.append(method_info)

    return render_template('unlock/choose.html', methods=methods, user=user)


# ===================== PIN UNLOCK =====================

@unlock_bp.route('/pin', methods=['GET', 'POST'])
def pin():
    user_id = session.get('pending_user_id')
    if not user_id:
        return redirect(url_for('auth.login'))

    if request.method == 'POST':
        data = request.get_json() if request.is_json else None
        pin_code = data.get('pin', '') if data else request.form.get('pin', '')

        method = AuthMethod.query.filter_by(user_id=user_id, method_type='pin').first()
        if not method:
            return _fail('PIN not enrolled.', is_json=request.is_json)

        method_data = json.loads(method.method_data)
        if check_password_hash(method_data['pin_hash'], pin_code):
            user = User.query.get(user_id)
            return _login_success(user, 'PIN', is_json=request.is_json)
        else:
            return _fail('Incorrect PIN.', is_json=request.is_json)

    return render_template('unlock/pin.html')


# ===================== PATTERN UNLOCK =====================

@unlock_bp.route('/pattern', methods=['GET', 'POST'])
def pattern():
    user_id = session.get('pending_user_id')
    if not user_id:
        return redirect(url_for('auth.login'))

    if request.method == 'POST':
        data = request.get_json()
        pattern_seq = data.get('pattern', [])

        method = AuthMethod.query.filter_by(user_id=user_id, method_type='pattern').first()
        if not method:
            return jsonify({'success': False, 'message': 'Pattern not enrolled.'})

        method_data = json.loads(method.method_data)
        pattern_str = '-'.join(map(str, pattern_seq))
        pattern_hash = hashlib.sha256(pattern_str.encode()).hexdigest()

        if pattern_hash == method_data['pattern_hash']:
            user = User.query.get(user_id)
            return _login_success(user, 'Pattern', is_json=True)
        else:
            return jsonify({'success': False, 'message': 'Incorrect pattern.'})

    return render_template('unlock/pattern.html')


# ===================== FACE UNLOCK =====================

@unlock_bp.route('/face', methods=['GET', 'POST'])
def face():
    user_id = session.get('pending_user_id')
    if not user_id:
        return redirect(url_for('auth.login'))

    if request.method == 'POST':
        from ml.face_recognition_ml import FaceRecognitionModel

        face_model = FaceRecognitionModel(
            models_dir=Config.TRAINED_MODELS_DIR,
            data_dir=Config.USER_DATA_DIR
        )

        # Opens cv2 window for face verification
        success, confidence, message = face_model.verify_with_camera(user_id, timeout=60)

        if success:
            user = User.query.get(user_id)
            return _login_success(user, 'Face ID', is_json=True, extra={
                'confidence': round(confidence * 100, 1)
            })
        else:
            return jsonify({
                'success': False,
                'message': message,
                'confidence': round(confidence * 100, 1)
            })

    return render_template('unlock/face.html')


# ===================== GESTURE UNLOCK =====================

@unlock_bp.route('/gesture', methods=['GET', 'POST'])
def gesture():
    user_id = session.get('pending_user_id')
    if not user_id:
        return redirect(url_for('auth.login'))

    if request.method == 'POST':
        from ml.gesture_recognition import GestureRecognitionModel

        gesture_model = GestureRecognitionModel(
            models_dir=Config.TRAINED_MODELS_DIR,
            data_dir=Config.USER_DATA_DIR
        )

        # Opens cv2 window for gesture verification
        success, confidence, message = gesture_model.verify_with_camera(user_id, timeout=60)

        if success:
            user = User.query.get(user_id)
            return _login_success(user, 'Hand Gesture', is_json=True, extra={
                'confidence': round(confidence * 100, 1)
            })
        else:
            return jsonify({
                'success': False,
                'message': message,
                'confidence': round(confidence * 100, 1)
            })

    return render_template('unlock/gesture.html')


# ===================== FINGERPRINT UNLOCK =====================

@unlock_bp.route('/fingerprint', methods=['GET'])
def fingerprint():
    user_id = session.get('pending_user_id')
    if not user_id:
        return redirect(url_for('auth.login'))
    return render_template('unlock/fingerprint.html')


@unlock_bp.route('/fingerprint/auth-options', methods=['POST'])
def fingerprint_auth_options():
    """Generate WebAuthn auth options — uses dynamic RP_ID."""
    import secrets

    user_id = session.get('pending_user_id')
    if not user_id:
        return jsonify({'success': False, 'message': 'Not authenticated.'})

    method = AuthMethod.query.filter_by(user_id=user_id, method_type='fingerprint').first()
    if not method or not method.credential_id:
        return jsonify({'success': False, 'message': 'Fingerprint not enrolled.'})

    challenge = secrets.token_bytes(32)
    session['webauthn_auth_challenge'] = challenge.hex()

    # Dynamic RP_ID from request
    rp_id = request.host.split(':')[0]

    options = {
        'challenge': list(challenge),
        'rpId': rp_id,
        'allowCredentials': [{
            'id': list(method.credential_id),
            'type': 'public-key',
            'transports': ['internal']
        }],
        'userVerification': 'required',
        'timeout': 60000
    }

    return jsonify(options)


@unlock_bp.route('/fingerprint/auth-complete', methods=['POST'])
def fingerprint_auth_complete():
    user_id = session.get('pending_user_id')
    if not user_id:
        return jsonify({'success': False, 'message': 'Not authenticated.'})

    data = request.get_json()

    try:
        method = AuthMethod.query.filter_by(user_id=user_id, method_type='fingerprint').first()
        if not method or not method.credential_id:
            return jsonify({'success': False, 'message': 'Fingerprint not enrolled.'})

        credential_id = bytes(data.get('credentialId', []))
        if credential_id == method.credential_id:
            method.sign_count += 1
            db.session.commit()
            user = User.query.get(user_id)
            return _login_success(user, 'Fingerprint', is_json=True)
        else:
            return jsonify({'success': False, 'message': 'Fingerprint verification failed.'})

    except Exception as e:
        logger.error(f"WebAuthn auth error: {e}")
        return jsonify({'success': False, 'message': f'Authentication failed: {str(e)}'})


# ===================== HELPERS =====================

def _login_success(user, method_name, is_json=False, extra=None):
    login_user(user)
    user.last_login = datetime.utcnow() + timedelta(hours=5, minutes=30)
    db.session.commit()
    session.pop('pending_user_id', None)

    if is_json:
        response = {
            'success': True,
            'message': f'Unlocked with {method_name}! Welcome back! 🎉',
            'redirect': url_for('auth.dashboard')
        }
        if extra:
            response.update(extra)
        return jsonify(response)
    else:
        flash(f'Unlocked with {method_name}! Welcome back! 🎉', 'success')
        return redirect(url_for('auth.dashboard'))


def _fail(message, is_json=False):
    if is_json:
        return jsonify({'success': False, 'message': message})
    else:
        flash(message, 'error')
        return redirect(request.url)
