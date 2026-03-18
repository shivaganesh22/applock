"""
App Lock - ML-Based Multi-Auth Security Application
=====================================================
Main Flask application. Initializes the app, database, login manager,
and registers all blueprints.
"""

from flask import Flask
from flask_login import LoginManager
from models import db, User
from config import Config
import logging
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app_lock.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_app():
    """Application factory."""
    app = Flask(__name__)
    app.config.from_object(Config)

    # Initialize directories
    Config.init_dirs()

    # Initialize database
    db.init_app(app)

    # Initialize Flask-Login
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message_category = 'info'

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # Register blueprints
    from routes.auth import auth_bp
    from routes.enroll import enroll_bp
    from routes.unlock import unlock_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(enroll_bp)
    app.register_blueprint(unlock_bp)

    # Create database tables
    with app.app_context():
        db.create_all()
        logger.info("Database initialized successfully.")

    return app


if __name__ == '__main__':
    app = create_app()
    logger.info("Starting App Lock server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
