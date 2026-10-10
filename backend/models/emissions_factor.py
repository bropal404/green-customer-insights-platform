from datetime import datetime
from .base import db


class EmissionsFactor(db.Model):
    __tablename__ = 'emissions_factors'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    activity_type = db.Column(db.String(50), nullable=False)  # fuel_car, electricity, gas, public_transport
    factor_value = db.Column(db.Float, nullable=False)  # kg CO2e per unit
    unit = db.Column(db.String(30), nullable=False)  # per_litre, per_kwh, per_km
    source = db.Column(db.String(100))
    year = db.Column(db.Integer)
    version = db.Column(db.String(30))
    is_illustrative = db.Column(db.Boolean, default=True)
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'activity_type': self.activity_type,
            'factor_value': self.factor_value,
            'unit': self.unit,
            'source': self.source,
            'year': self.year,
            'version': self.version,
            'is_illustrative': self.is_illustrative,
            'notes': self.notes,
        }
