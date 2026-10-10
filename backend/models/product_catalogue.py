from datetime import datetime
from .base import db


class ProductCatalogue(db.Model):
    __tablename__ = 'product_catalogue'

    product_code = db.Column(db.String(10), primary_key=True)
    product_name = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(50))
    description = db.Column(db.Text)
    min_propensity_threshold = db.Column(db.Float, default=0.3)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'product_code': self.product_code,
            'product_name': self.product_name,
            'category': self.category,
            'description': self.description,
            'min_propensity_threshold': self.min_propensity_threshold,
            'is_active': self.is_active,
        }
