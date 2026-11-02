from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """User model for authentication"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    uploads = db.relationship('Upload', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check if password matches hash"""
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'


class Upload(db.Model):
    """Model for uploaded images and detection results"""
    __tablename__ = 'uploads'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(500), nullable=False)
    detection_result = db.Column(db.String(50), nullable=False)  # 'harassment' or 'normal'
    confidence = db.Column(db.Float, nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    reviewed = db.Column(db.Boolean, default=False)
    
    # Relationship to alerts
    alert = db.relationship('Alert', backref='upload', uselist=False, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Upload {self.filename}>'


class Alert(db.Model):
    """Model for harassment alerts to admins"""
    __tablename__ = 'alerts'
    
    id = db.Column(db.Integer, primary_key=True)
    upload_id = db.Column(db.Integer, db.ForeignKey('uploads.id'), nullable=False)
    status = db.Column(db.String(20), default='pending')  # pending, reviewed, dismissed
    admin_notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    reviewed_at = db.Column(db.DateTime)
    reviewed_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    
    def __repr__(self):
        return f'<Alert {self.id} - {self.status}>'

