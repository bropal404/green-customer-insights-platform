from datetime import datetime
from .base import db


class ImpactEstimate(db.Model):
    __tablename__ = 'impact_estimates'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    customer_id = db.Column(db.String(20), db.ForeignKey('customers.customer_id'), nullable=False, index=True)
    baseline_kg_co2e = db.Column(db.Float)
    projected_kg_co2e = db.Column(db.Float)
    reduction_kg_co2e = db.Column(db.Float)
    fuel_baseline_kg = db.Column(db.Float, default=0.0)
    electricity_baseline_kg = db.Column(db.Float, default=0.0)
    gas_baseline_kg = db.Column(db.Float, default=0.0)
    transport_baseline_kg = db.Column(db.Float, default=0.0)
    fuel_projected_kg = db.Column(db.Float, default=0.0)
    electricity_projected_kg = db.Column(db.Float, default=0.0)
    gas_projected_kg = db.Column(db.Float, default=0.0)
    transport_projected_kg = db.Column(db.Float, default=0.0)
    assumptions = db.Column(db.Text)  # JSON
    factor_version = db.Column(db.String(30))
    computed_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        import json
        return {
            'id': self.id,
            'customer_id': self.customer_id,
            'baseline_kg_co2e': self.baseline_kg_co2e,
            'projected_kg_co2e': self.projected_kg_co2e,
            'reduction_kg_co2e': self.reduction_kg_co2e,
            'fuel_baseline_kg': self.fuel_baseline_kg,
            'electricity_baseline_kg': self.electricity_baseline_kg,
            'gas_baseline_kg': self.gas_baseline_kg,
            'transport_baseline_kg': self.transport_baseline_kg,
            'fuel_projected_kg': self.fuel_projected_kg,
            'electricity_projected_kg': self.electricity_projected_kg,
            'gas_projected_kg': self.gas_projected_kg,
            'transport_projected_kg': self.transport_projected_kg,
            'assumptions': json.loads(self.assumptions) if self.assumptions else {},
            'factor_version': self.factor_version,
            'computed_at': self.computed_at.isoformat() if self.computed_at else None,
        }
