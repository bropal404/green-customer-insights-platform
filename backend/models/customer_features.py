from datetime import datetime
from .base import db


class CustomerFeatures(db.Model):
    __tablename__ = 'customer_features'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    customer_id = db.Column(db.String(20), db.ForeignKey('customers.customer_id'), nullable=False, unique=True, index=True)
    # Spending features
    monthly_fuel_spend_gbp = db.Column(db.Float, default=0.0)
    monthly_electricity_spend_gbp = db.Column(db.Float, default=0.0)
    monthly_gas_spend_gbp = db.Column(db.Float, default=0.0)
    monthly_transport_spend_gbp = db.Column(db.Float, default=0.0)
    monthly_home_improvement_spend_gbp = db.Column(db.Float, default=0.0)
    monthly_utilities_spend_gbp = db.Column(db.Float, default=0.0)
    monthly_total_spend_gbp = db.Column(db.Float, default=0.0)
    # Activity features
    annual_fuel_transactions = db.Column(db.Integer, default=0)
    annual_transport_transactions = db.Column(db.Integer, default=0)
    # Derived features
    estimated_annual_mileage = db.Column(db.Float)
    estimated_annual_kwh_electricity = db.Column(db.Float)
    estimated_annual_kwh_gas = db.Column(db.Float)
    # Behavioural
    avg_transactions_per_month = db.Column(db.Float, default=0.0)
    digital_logins_per_month = db.Column(db.Float, default=0.0)
    # Labels (synthetic/simulated)
    label_ev_taker = db.Column(db.Boolean)
    label_green_home_taker = db.Column(db.Boolean)
    label_solar_taker = db.Column(db.Boolean)
    label_paperless = db.Column(db.Boolean)
    computed_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'customer_id': self.customer_id,
            'monthly_fuel_spend_gbp': self.monthly_fuel_spend_gbp,
            'monthly_electricity_spend_gbp': self.monthly_electricity_spend_gbp,
            'monthly_gas_spend_gbp': self.monthly_gas_spend_gbp,
            'monthly_transport_spend_gbp': self.monthly_transport_spend_gbp,
            'monthly_home_improvement_spend_gbp': self.monthly_home_improvement_spend_gbp,
            'monthly_utilities_spend_gbp': self.monthly_utilities_spend_gbp,
            'monthly_total_spend_gbp': self.monthly_total_spend_gbp,
            'annual_fuel_transactions': self.annual_fuel_transactions,
            'annual_transport_transactions': self.annual_transport_transactions,
            'estimated_annual_mileage': self.estimated_annual_mileage,
            'estimated_annual_kwh_electricity': self.estimated_annual_kwh_electricity,
            'estimated_annual_kwh_gas': self.estimated_annual_kwh_gas,
            'avg_transactions_per_month': self.avg_transactions_per_month,
            'digital_logins_per_month': self.digital_logins_per_month,
            'label_ev_taker': self.label_ev_taker,
            'label_green_home_taker': self.label_green_home_taker,
            'label_solar_taker': self.label_solar_taker,
            'label_paperless': self.label_paperless,
            'computed_at': self.computed_at.isoformat() if self.computed_at else None,
        }
