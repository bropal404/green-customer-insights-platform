from datetime import datetime
from .base import db


class Recommendation(db.Model):
    __tablename__ = 'recommendations'
    __table_args__ = (
        db.UniqueConstraint('customer_id', 'product_code', name='uq_rec_customer_product'),
    )

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    customer_id = db.Column(db.String(20), db.ForeignKey('customers.customer_id'), nullable=False, index=True)
    product_code = db.Column(db.String(10), nullable=False)
    status = db.Column(db.String(20), default='active')  # active, dismissed, converted
    dismissed_at = db.Column(db.DateTime)
    dismiss_reason = db.Column(db.String(200))
    estimated_annual_impact_kg = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    explanation = db.relationship('LLMExplanation', backref='recommendation', uselist=False)

    def to_dict(self):
        return {
            'id': self.id,
            'customer_id': self.customer_id,
            'product_code': self.product_code,
            'status': self.status,
            'dismissed_at': self.dismissed_at.isoformat() if self.dismissed_at else None,
            'dismiss_reason': self.dismiss_reason,
            'estimated_annual_impact_kg': self.estimated_annual_impact_kg,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
