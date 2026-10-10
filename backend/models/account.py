from datetime import datetime
from .base import db


class Account(db.Model):
    __tablename__ = 'accounts'

    account_id = db.Column(db.String(20), primary_key=True)
    customer_id = db.Column(db.String(20), db.ForeignKey('customers.customer_id'), nullable=False, index=True)
    account_type = db.Column(db.String(30), nullable=False)  # Current, Savings, ISA
    balance_gbp = db.Column(db.Float, default=0.0)
    overdraft_limit_gbp = db.Column(db.Float, default=0.0)
    is_primary = db.Column(db.Boolean, default=False)
    opened_date = db.Column(db.Date)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    transactions = db.relationship('Transaction', backref='account', lazy='dynamic')

    def to_dict(self):
        return {
            'account_id': self.account_id,
            'customer_id': self.customer_id,
            'account_type': self.account_type,
            'balance_gbp': self.balance_gbp,
            'overdraft_limit_gbp': self.overdraft_limit_gbp,
            'is_primary': self.is_primary,
            'opened_date': self.opened_date.isoformat() if self.opened_date else None,
        }
