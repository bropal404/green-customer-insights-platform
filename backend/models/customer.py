from datetime import datetime
from .base import db


class Customer(db.Model):
    __tablename__ = 'customers'

    customer_id = db.Column(db.String(20), primary_key=True)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    age_band = db.Column(db.String(20), nullable=False)  # 18-24, 25-34, etc.
    gender = db.Column(db.String(10))
    region = db.Column(db.String(50), nullable=False)
    postcode_district = db.Column(db.String(10))
    income_band = db.Column(db.String(20), nullable=False)  # Low, Medium, High
    annual_income_gbp = db.Column(db.Float)
    household_size = db.Column(db.Integer)
    household_type = db.Column(db.String(30))  # Single, Couple, Family, etc.
    is_homeowner = db.Column(db.Boolean, default=False)
    housing_type = db.Column(db.String(30))  # Detached, Semi, Terraced, Flat
    has_mortgage = db.Column(db.Boolean, default=False)
    property_value_band = db.Column(db.String(20))
    has_vehicle = db.Column(db.Boolean, default=False)
    vehicle_type = db.Column(db.String(30))  # Petrol, Diesel, Electric, Hybrid, None
    estimated_annual_mileage = db.Column(db.Float)
    digital_activity_score = db.Column(db.Float)  # 0-1
    paper_statements = db.Column(db.Boolean, default=False)
    mobile_banking_active = db.Column(db.Boolean, default=True)
    customer_since = db.Column(db.Date)
    customer_segment = db.Column(db.String(30))  # Mass, Affluent, HNW
    split_set = db.Column(db.String(10), default='train')  # train/val/test
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    accounts = db.relationship('Account', backref='customer', lazy='dynamic')
    features = db.relationship('CustomerFeatures', backref='customer', uselist=False)
    predictions = db.relationship('PropensityPrediction', backref='customer', lazy='dynamic')

    def to_dict(self):
        return {
            'customer_id': self.customer_id,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'full_name': f'{self.first_name} {self.last_name}',
            'age': self.age,
            'age_band': self.age_band,
            'gender': self.gender,
            'region': self.region,
            'postcode_district': self.postcode_district,
            'income_band': self.income_band,
            'annual_income_gbp': self.annual_income_gbp,
            'household_size': self.household_size,
            'household_type': self.household_type,
            'is_homeowner': self.is_homeowner,
            'housing_type': self.housing_type,
            'has_mortgage': self.has_mortgage,
            'property_value_band': self.property_value_band,
            'has_vehicle': self.has_vehicle,
            'vehicle_type': self.vehicle_type,
            'estimated_annual_mileage': self.estimated_annual_mileage,
            'digital_activity_score': self.digital_activity_score,
            'paper_statements': self.paper_statements,
            'mobile_banking_active': self.mobile_banking_active,
            'customer_since': self.customer_since.isoformat() if self.customer_since else None,
            'customer_segment': self.customer_segment,
            'split_set': self.split_set,
        }
