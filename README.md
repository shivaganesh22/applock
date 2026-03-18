# 🔐 AppLock — ML-Based Multi-Auth Security Application

A professional Flask web application that lets users sign up and enroll in multiple authentication methods powered by **Machine Learning / Deep Learning** models. Users can log in using any enrolled method.

## 🧠 ML Models

| Method | Algorithm | Library | Features |
|--------|-----------|---------|----------|
| **Face ID** | SVM (RBF Kernel) | MediaPipe + scikit-learn | 468 face landmarks + key-point distances |
| **Hand Gesture** | Random Forest | MediaPipe Hands + scikit-learn | 21 hand landmarks + joint angles + curl features |
| **Fingerprint** | WebAuthn | Device hardware | Cryptographic biometric auth |
| **PIN Code** | Hash comparison | Werkzeug | SHA-256 hash |
| **Pattern Lock** | Hash comparison | hashlib | SHA-256 hash |

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Webcam (for Face ID and Hand Gesture)
- Device with fingerprint sensor (for Fingerprint)

### Installation

```bash
cd app_lock
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000** in your browser.

> ⚠️ **Important:** For fingerprint (WebAuthn) to work, you must access via `http://localhost:5000`, NOT `http://127.0.0.1:5000`.

## 📱 User Flow

1. **Sign Up** → Create account with email + password
2. **Enroll Methods** → Choose 1 to 5 methods to enroll:
   - **PIN** — Enter a 4-6 digit numeric PIN
   - **Pattern** — Draw a pattern on a 3×3 grid
   - **Face ID** — Webcam captures 20 face images → SVM model is trained
   - **Hand Gesture** — Select a gesture → Webcam captures 30 frames → Random Forest is trained
   - **Fingerprint** — Device prompts fingerprint scan via WebAuthn
3. **Login** → Enter email + password → Choose any enrolled method → Verify → Dashboard

## 🏗️ Project Structure

```
app_lock/
├── app.py                          # Flask app factory
├── config.py                       # Configuration
├── models.py                       # SQLAlchemy models
├── ml/                             # ML modules
│   ├── face_recognition_ml.py      # Face (MediaPipe + SVM)
│   └── gesture_recognition.py      # Gesture (MediaPipe Hands + RF)
├── routes/                         # Flask blueprints
│   ├── auth.py                     # Signup, Login, Dashboard
│   ├── enroll.py                   # Enrollment (all 5 methods)
│   └── unlock.py                   # Verification (all 5 methods)
├── static/css/style.css            # Custom component styles
├── templates/                      # Jinja2 templates (Tailwind CSS CDN)
│   ├── base.html, index.html, signup.html, login.html, dashboard.html
│   ├── enroll/ (hub, pin, pattern, face, gesture, fingerprint)
│   └── unlock/ (choose, pin, pattern, face, gesture, fingerprint)
├── trained_models/                 # Per-user trained ML models (.pkl)
├── user_data/                      # Per-user face/gesture training data
└── requirements.txt
```

## 🎨 Tech Stack

- **Backend:** Flask, Flask-SQLAlchemy, Flask-Login
- **ML/AI:** MediaPipe, scikit-learn (SVM, Random Forest), OpenCV, NumPy
- **Frontend:** Tailwind CSS CDN, Font Awesome, Vanilla JS
- **Auth:** WebAuthn API (fingerprint), Werkzeug password hashing
- **Database:** SQLite
