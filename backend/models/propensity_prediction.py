from datetime import datetime
from .base import db


class PropensityPrediction(db.Model):
    __tablename__ = 'propensity_predictions'
    __table_args__ = (
        db.UniqueConstraint('customer_id', 'product_code', name='uq_customer_product'),
    )

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    customer_id = db.Column(db.String(20), db.ForeignKey('customers.customer_id'), nullable=False, index=True)
    product_code = db.Column(db.String(10), nullable=False)
    propensity_score = db.Column(db.Float, nullable=False)
    confidence_score = db.Column(db.Float, nullable=False)
    reason_codes = db.Column(db.Text)  # JSON list
    eligibility_status = db.Column(db.String(20), default='eligible')
    eligibility_notes = db.Column(db.Text)
    model_version = db.Column(db.String(30))
    predicted_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        import json
        return {
            'id': self.id,
            'customer_id': self.customer_id,
            'product_code': self.product_code,
            'propensity_score': self.propensity_score,
            'confidence_score': self.confidence_score,
            'reason_codes': json.loads(self.reason_codes) if self.reason_codes else [],
            'eligibility_status': self.eligibility_status,
            'eligibility_notes': self.eligibility_notes,
            'model_version': self.model_version,
            'predicted_at': self.predicted_at.isoformat() if self.predicted_at else None,
        }
