from datetime import datetime
from .base import db

VALID_CATEGORIES = [
    'Fuel', 'Electricity', 'Gas', 'Public Transport',
    'Home Improvement', 'Automotive', 'Utilities', 'Other'
]


class TransactionClassification(db.Model):
    __tablename__ = 'transaction_classifications'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    transaction_id = db.Column(db.String(20), db.ForeignKey('transactions.transaction_id'), nullable=False, unique=True, index=True)
    category = db.Column(db.String(30), nullable=False, default='Other')
    confidence = db.Column(db.Float, default=0.5)
    status = db.Column(db.String(20), default='auto')  # auto, manual, review
    is_manual = db.Column(db.Boolean, default=False)
    model_version = db.Column(db.String(30))
    classified_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'transaction_id': self.transaction_id,
            'category': self.category,
            'confidence': self.confidence,
            'status': self.status,
            'is_manual': self.is_manual,
            'model_version': self.model_version,
            'classified_at': self.classified_at.isoformat() if self.classified_at else None,
        }
