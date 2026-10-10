from datetime import datetime
from .base import db


class FinancialProduct(db.Model):
    __tablename__ = 'financial_products'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    customer_id = db.Column(db.String(20), db.ForeignKey('customers.customer_id'), nullable=False, index=True)
    product_code = db.Column(db.String(20), nullable=False)  # MORTGAGE, LOAN, CC, etc.
    product_name = db.Column(db.String(100))
    balance_gbp = db.Column(db.Float)
    monthly_payment_gbp = db.Column(db.Float)
    interest_rate = db.Column(db.Float)
    start_date = db.Column(db.Date)
    end_date = db.Column(db.Date)
    status = db.Column(db.String(20), default='Active')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'customer_id': self.customer_id,
            'product_code': self.product_code,
            'product_name': self.product_name,
            'balance_gbp': self.balance_gbp,
            'monthly_payment_gbp': self.monthly_payment_gbp,
            'interest_rate': self.interest_rate,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'status': self.status,
        }
