"""
Authentication Routes
======================
Handles user signup, basic login (email), and logout.
"""

from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from flask_login import login_user, logout_user, login_required, current_user
from models import db, User, AuthMethod
from datetime import datetime

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/')
def index():
    """Landing page."""
    if current_user.is_authenticated:
        return redirect(url_for('auth.dashboard'))
    return render_template('index.html')


@auth_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    """User registration."""
    if current_user.is_authenticated:
        return redirect(url_for('auth.dashboard'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        # Validation
        if not email or not password:
            flash('Email and password are required.', 'error')
            return render_template('signup.html')

        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
            return render_template('signup.html')

        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('signup.html')

        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'error')
            return render_template('signup.html')

        # Create user
        user = User(email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        login_user(user)
        flash('Account created successfully! Set up your security methods.', 'success')
        return redirect(url_for('enroll.hub'))

    return render_template('signup.html')


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Login - enter email to see available unlock methods."""
    if current_user.is_authenticated:
        return redirect(url_for('auth.dashboard'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        user = User.query.filter_by(email=email).first()

        if not user or not user.check_password(password):
            flash('Invalid email or password.', 'error')
            return render_template('login.html')

        # Check if user has enrolled methods
        enrolled = user.get_enrolled_methods()
        if not enrolled:
            # No security methods enrolled, log in directly
            login_user(user)
            user.last_login = datetime.utcnow()
            db.session.commit()
            flash('Welcome back! Consider setting up security methods.', 'info')
            return redirect(url_for('enroll.hub'))

        # Store user_id in session for unlock flow
        session['pending_user_id'] = user.id
        return redirect(url_for('unlock.choose'))

    return render_template('login.html')


@auth_bp.route('/dashboard')
@login_required
def dashboard():
    """User dashboard showing enrolled methods and security status."""
    enrolled_methods = current_user.get_enrolled_methods()
    all_methods = [
        {'type': 'pin', 'name': 'PIN Code', 'icon': 'fa-key', 'color': '#6c5ce7'},
        {'type': 'pattern', 'name': 'Pattern Lock', 'icon': 'fa-th', 'color': '#00cec9'},
        {'type': 'face', 'name': 'Face ID', 'icon': 'fa-smile', 'color': '#e17055'},
        {'type': 'gesture', 'name': 'Hand Gesture', 'icon': 'fa-hand-paper', 'color': '#fdcb6e'},
        {'type': 'fingerprint', 'name': 'Fingerprint', 'icon': 'fa-fingerprint', 'color': '#55efc4'},
    ]

    for method in all_methods:
        method['enrolled'] = current_user.has_method(method['type'])

    return render_template('dashboard.html',
                           methods=all_methods,
                           enrolled_count=len(enrolled_methods),
                           user=current_user)


@auth_bp.route('/logout')
@login_required
def logout():
    """Logout user."""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('auth.index'))
