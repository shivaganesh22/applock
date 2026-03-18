import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'app-lock-super-secret-key-2024')
    SQLALCHEMY_DATABASE_URI = f'sqlite:///{os.path.join(BASE_DIR, "app_lock.db")}'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Directories for user data and trained models
    USER_DATA_DIR = os.path.join(BASE_DIR, 'user_data')
    TRAINED_MODELS_DIR = os.path.join(BASE_DIR, 'trained_models')
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')

    # WebAuthn config
    RP_ID = 'localhost'
    RP_NAME = 'AppLock Security'
    ORIGIN = 'http://localhost:5000'

    # NOTE: If accessing via 127.0.0.1, WebAuthn won't work.
    # You MUST access via http://localhost:5000 for fingerprint to work.

    # ML Config
    FACE_SAMPLES_REQUIRED = 20
    GESTURE_SAMPLES_REQUIRED = 30

    @staticmethod
    def init_dirs():
        for d in [Config.USER_DATA_DIR, Config.TRAINED_MODELS_DIR, Config.UPLOAD_FOLDER]:
            os.makedirs(d, exist_ok=True)
