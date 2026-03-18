from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()


class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)

    # Relationships
    auth_methods = db.relationship('AuthMethod', backref='user', lazy=True, cascade='all, delete-orphan')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def get_enrolled_methods(self):
        return [m for m in self.auth_methods if m.is_enrolled]

    def has_method(self, method_type):
        return any(m.method_type == method_type and m.is_enrolled for m in self.auth_methods)

    def get_method(self, method_type):
        return AuthMethod.query.filter_by(user_id=self.id, method_type=method_type).first()

    def __repr__(self):
        return f'<User {self.email}>'


class AuthMethod(db.Model):
    __tablename__ = 'auth_methods'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    method_type = db.Column(db.String(20), nullable=False)  # pin, pattern, face, gesture, fingerprint
    method_data = db.Column(db.Text, nullable=True)  # JSON string for storing hashes, model paths, etc.
    is_enrolled = db.Column(db.Boolean, default=False)
    enrolled_at = db.Column(db.DateTime, nullable=True)
    model_path = db.Column(db.String(500), nullable=True)  # Path to trained ML model

    # WebAuthn fields for fingerprint
    credential_id = db.Column(db.LargeBinary, nullable=True)
    public_key = db.Column(db.LargeBinary, nullable=True)
    sign_count = db.Column(db.Integer, default=0)

    __table_args__ = (
        db.UniqueConstraint('user_id', 'method_type', name='unique_user_method'),
    )

    def __repr__(self):
        return f'<AuthMethod {self.method_type} for User {self.user_id}>'
