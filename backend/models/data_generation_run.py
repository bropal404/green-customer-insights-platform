from datetime import datetime
from .base import db


class DataGenerationRun(db.Model):
    __tablename__ = 'data_generation_runs'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    run_at = db.Column(db.DateTime, default=datetime.utcnow)
    random_seed = db.Column(db.Integer)
    num_customers = db.Column(db.Integer)
    num_transactions = db.Column(db.Integer)
    num_accounts = db.Column(db.Integer)
    status = db.Column(db.String(20), default='complete')
    notes = db.Column(db.Text)

    def to_dict(self):
        return {
            'id': self.id,
            'run_at': self.run_at.isoformat() if self.run_at else None,
            'random_seed': self.random_seed,
            'num_customers': self.num_customers,
            'num_transactions': self.num_transactions,
            'num_accounts': self.num_accounts,
            'status': self.status,
            'notes': self.notes,
        }
