"""
Enrollment Routes
==================
Handles enrollment for all 5 authentication methods:
PIN, Pattern, Face ID, Hand Gesture, Fingerprint.

Face and Gesture enrollment now open OpenCV cv2 windows for capture.
"""

from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, session
from flask_login import login_required, current_user
from models import db, AuthMethod
from werkzeug.security import generate_password_hash
from config import Config
import json
import hashlib
import os
import logging
import threading

logger = logging.getLogger(__name__)

enroll_bp = Blueprint('enroll', __name__, url_prefix='/enroll')


@enroll_bp.route('/')
@login_required
def hub():
    """Enrollment hub."""
    methods = [
        {
            'type': 'pin', 'name': 'PIN Code', 'icon': 'fa-key',
            'description': 'Set a 4-6 digit numeric PIN for quick access',
            'color': '#6c5ce7', 'gradient': 'linear-gradient(135deg, #6c5ce7, #a29bfe)',
            'enrolled': current_user.has_method('pin')
        },
        {
            'type': 'pattern', 'name': 'Pattern Lock', 'icon': 'fa-th',
            'description': 'Draw a pattern on a 3×3 grid to unlock',
            'color': '#00cec9', 'gradient': 'linear-gradient(135deg, #00cec9, #81ecec)',
            'enrolled': current_user.has_method('pattern')
        },
        {
            'type': 'face', 'name': 'Face ID', 'icon': 'fa-smile',
            'description': 'Train SVM model to recognize your face via camera',
            'color': '#e17055', 'gradient': 'linear-gradient(135deg, #e17055, #fab1a0)',
            'enrolled': current_user.has_method('face'),
            'ml_badge': True
        },
        {
            'type': 'gesture', 'name': 'Gesture Sequence', 'icon': 'fa-hand-paper',
            'description': 'Create a gesture password sequence using hand signs',
            'color': '#fdcb6e', 'gradient': 'linear-gradient(135deg, #fdcb6e, #ffeaa7)',
            'enrolled': current_user.has_method('gesture'),
            'ml_badge': True
        },
        {
            'type': 'fingerprint', 'name': 'Fingerprint', 'icon': 'fa-fingerprint',
            'description': 'Use your device fingerprint scanner for biometric auth',
            'color': '#55efc4', 'gradient': 'linear-gradient(135deg, #55efc4, #00b894)',
            'enrolled': current_user.has_method('fingerprint')
        },
    ]
    return render_template('enroll/hub.html', methods=methods)


# ===================== PIN ENROLLMENT =====================

@enroll_bp.route('/pin', methods=['GET', 'POST'])
@login_required
def pin():
    """Enroll a 4-6 digit PIN."""
    if request.method == 'POST':
        pin_code = request.form.get('pin', '')
        confirm_pin = request.form.get('confirm_pin', '')

        if not pin_code.isdigit() or len(pin_code) < 4 or len(pin_code) > 6:
            flash('PIN must be 4-6 digits.', 'error')
            return render_template('enroll/pin.html')

        if pin_code != confirm_pin:
            flash('PINs do not match.', 'error')
            return render_template('enroll/pin.html')

        pin_hash = generate_password_hash(pin_code)
        _save_method(current_user.id, 'pin', {'pin_hash': pin_hash})

        flash('PIN enrolled successfully! ✅', 'success')
        return redirect(url_for('enroll.hub'))

    return render_template('enroll/pin.html')


# ===================== PATTERN ENROLLMENT =====================

@enroll_bp.route('/pattern', methods=['GET', 'POST'])
@login_required
def pattern():
    """Enroll a pattern lock."""
    if request.method == 'POST':
        data = request.get_json()
        pattern_seq = data.get('pattern', [])
        confirm_seq = data.get('confirm_pattern', [])

        if len(pattern_seq) < 4:
            return jsonify({'success': False, 'message': 'Pattern must connect at least 4 dots.'})

        if pattern_seq != confirm_seq:
            return jsonify({'success': False, 'message': 'Patterns do not match.'})

        pattern_str = '-'.join(map(str, pattern_seq))
        pattern_hash = hashlib.sha256(pattern_str.encode()).hexdigest()
        _save_method(current_user.id, 'pattern', {'pattern_hash': pattern_hash})

        return jsonify({'success': True, 'message': 'Pattern enrolled successfully! ✅'})

    return render_template('enroll/pattern.html')


# ===================== FACE ID ENROLLMENT =====================

@enroll_bp.route('/face', methods=['GET'])
@login_required
def face():
    """Face ID enrollment page."""
    return render_template('enroll/face.html')


@enroll_bp.route('/face/start', methods=['POST'])
@login_required
def face_start():
    """Start face enrollment — opens a cv2 window for capture & training."""
    from ml.face_recognition_ml import FaceRecognitionModel

    face_model = FaceRecognitionModel(
        models_dir=Config.TRAINED_MODELS_DIR,
        data_dir=Config.USER_DATA_DIR
    )

    # This opens a cv2 window, captures faces, and trains the SVM
    success, message = face_model.capture_and_train(current_user.id, num_samples=20)

    if success:
        model_info = face_model.get_model_info(current_user.id)
        _save_method(current_user.id, 'face', {
            'model_path': model_info['model_path'],
            'n_samples': model_info['n_samples'],
            'model_type': model_info['model_type']
        })
        return jsonify({
            'success': True,
            'message': message,
            'model_info': model_info
        })
    else:
        return jsonify({'success': False, 'message': message})


# ===================== GESTURE ENROLLMENT =====================

@enroll_bp.route('/gesture', methods=['GET'])
@login_required
def gesture():
    """Hand gesture sequence enrollment page."""
    return render_template('enroll/gesture.html')


@enroll_bp.route('/gesture/start', methods=['POST'])
@login_required
def gesture_start():
    """Start gesture enrollment — opens a cv2 window for sequence creation."""
    from ml.gesture_recognition import GestureRecognitionModel

    gesture_model = GestureRecognitionModel(
        models_dir=Config.TRAINED_MODELS_DIR,
        data_dir=Config.USER_DATA_DIR
    )

    # Opens cv2 window where user creates a gesture sequence
    success, message = gesture_model.capture_and_train(current_user.id)

    if success:
        model_info = gesture_model.get_model_info(current_user.id)
        _save_method(current_user.id, 'gesture', {
            'gesture_display': model_info['gesture_display'],
            'sequence_length': model_info['sequence_length'],
            'model_type': model_info['model_type']
        })
        return jsonify({
            'success': True,
            'message': message,
            'model_info': model_info
        })
    else:
        return jsonify({'success': False, 'message': message})


# ===================== FINGERPRINT ENROLLMENT =====================

@enroll_bp.route('/fingerprint', methods=['GET'])
@login_required
def fingerprint():
    """Fingerprint enrollment page using WebAuthn."""
    return render_template('enroll/fingerprint.html')


@enroll_bp.route('/fingerprint/register-options', methods=['POST'])
@login_required
def fingerprint_register_options():
    """Generate WebAuthn registration options — uses dynamic RP_ID from request."""
    import secrets
    challenge = secrets.token_bytes(32)
    session['webauthn_challenge'] = challenge.hex()

    # Dynamically get RP_ID from the actual request host
    rp_id = request.host.split(':')[0]  # 'localhost' or '127.0.0.1'

    options = {
        'challenge': list(challenge),
        'rp': {
            'name': Config.RP_NAME,
            'id': rp_id
        },
        'user': {
            'id': list(current_user.id.to_bytes(8, 'big')),
            'name': current_user.email,
            'displayName': current_user.email.split('@')[0]
        },
        'pubKeyCredParams': [
            {'alg': -7, 'type': 'public-key'},
            {'alg': -257, 'type': 'public-key'}
        ],
        'authenticatorSelection': {
            'authenticatorAttachment': 'platform',
            'userVerification': 'required',
            'residentKey': 'preferred'
        },
        'timeout': 60000,
        'attestation': 'none'
    }

    return jsonify(options)


@enroll_bp.route('/fingerprint/register-complete', methods=['POST'])
@login_required
def fingerprint_register_complete():
    """Complete WebAuthn registration."""
    data = request.get_json()

    try:
        credential_id = bytes(data['credentialId'])
        public_key = json.dumps(data['response'])

        _save_method(current_user.id, 'fingerprint', {
            'credential_type': 'webauthn',
            'webauthn_registered': True
        })

        method = AuthMethod.query.filter_by(
            user_id=current_user.id,
            method_type='fingerprint'
        ).first()
        if method:
            method.credential_id = credential_id
            method.public_key = public_key.encode()
            db.session.commit()

        return jsonify({'success': True, 'message': 'Fingerprint enrolled successfully! 🔐'})

    except Exception as e:
        logger.error(f"WebAuthn registration error: {e}")
        return jsonify({'success': False, 'message': f'Registration failed: {str(e)}'})


# ===================== REMOVE METHOD =====================

@enroll_bp.route('/remove/<method_type>', methods=['POST'])
@login_required
def remove_method(method_type):
    """Remove an enrolled authentication method."""
    method = AuthMethod.query.filter_by(
        user_id=current_user.id,
        method_type=method_type
    ).first()

    if method:
        db.session.delete(method)
        db.session.commit()
        if method_type in ('face', 'gesture'):
            _cleanup_ml_files(current_user.id, method_type)
        flash(f'{method_type.title()} method removed.', 'info')

    return redirect(url_for('enroll.hub'))


# ===================== HELPERS =====================

def _save_method(user_id, method_type, data):
    """Save or update an authentication method."""
    method = AuthMethod.query.filter_by(user_id=user_id, method_type=method_type).first()
    if not method:
        method = AuthMethod(user_id=user_id, method_type=method_type)
        db.session.add(method)

    method.method_data = json.dumps(data)
    method.is_enrolled = True
    method.enrolled_at = __import__('datetime').datetime.utcnow()
    if 'model_path' in data:
        method.model_path = data['model_path']
    db.session.commit()


def _cleanup_ml_files(user_id, method_type):
    """Clean up ML model files."""
    import shutil
    if method_type == 'face':
        model_file = os.path.join(Config.TRAINED_MODELS_DIR, f'face_model_{user_id}.pkl')
        data_dir = os.path.join(Config.USER_DATA_DIR, f'face_{user_id}')
    elif method_type == 'gesture':
        model_file = os.path.join(Config.TRAINED_MODELS_DIR, f'gesture_model_{user_id}.pkl')
        data_dir = os.path.join(Config.USER_DATA_DIR, f'gesture_{user_id}')
    else:
        return
    if os.path.exists(model_file):
        os.remove(model_file)
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
