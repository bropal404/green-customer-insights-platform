from datetime import datetime
from .base import db


class Transaction(db.Model):
    __tablename__ = 'transactions'
    __table_args__ = (
        db.Index('ix_transactions_customer_date', 'customer_id', 'transaction_date'),
        db.Index('ix_transactions_date', 'transaction_date'),
    )

    transaction_id = db.Column(db.String(20), primary_key=True)
    customer_id = db.Column(db.String(20), db.ForeignKey('customers.customer_id'), nullable=False, index=True)
    account_id = db.Column(db.String(20), db.ForeignKey('accounts.account_id'), nullable=False)
    transaction_date = db.Column(db.Date, nullable=False)
    merchant_description = db.Column(db.String(200), nullable=False)
    amount_gbp = db.Column(db.Float, nullable=False)
    transaction_type = db.Column(db.String(20))  # Debit, Credit
    mcc_code = db.Column(db.String(10))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    classification = db.relationship('TransactionClassification', backref='transaction', uselist=False)

    def to_dict(self):
        cls = self.classification
        return {
            'transaction_id': self.transaction_id,
            'customer_id': self.customer_id,
            'account_id': self.account_id,
            'transaction_date': self.transaction_date.isoformat() if self.transaction_date else None,
            'merchant_description': self.merchant_description,
            'amount_gbp': self.amount_gbp,
            'transaction_type': self.transaction_type,
            'mcc_code': self.mcc_code,
            'category': cls.category if cls else 'Other',
            'confidence': cls.confidence if cls else 0.5,
            'classification_status': cls.status if cls else 'unclassified',
            'is_manual': cls.is_manual if cls else False,
        }
