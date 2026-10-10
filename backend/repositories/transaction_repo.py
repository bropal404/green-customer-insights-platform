from datetime import date
from sqlalchemy import func, or_
from backend.models import db, Transaction, TransactionClassification


class TransactionRepository:
    def list_transactions(self, page=1, per_page=50, customer_id=None, category=None,
                          region=None, start_date=None, end_date=None,
                          min_amount=None, max_amount=None, search=None,
                          low_confidence=False, manual_only=False):
        q = db.session.query(Transaction).join(
            TransactionClassification,
            Transaction.transaction_id == TransactionClassification.transaction_id,
            isouter=True
        )
        if customer_id:
            q = q.filter(Transaction.customer_id == customer_id)
        if category:
            q = q.filter(TransactionClassification.category == category)
        if start_date:
            q = q.filter(Transaction.transaction_date >= start_date)
        if end_date:
            q = q.filter(Transaction.transaction_date <= end_date)
        if min_amount is not None:
            q = q.filter(Transaction.amount_gbp >= min_amount)
        if max_amount is not None:
            q = q.filter(Transaction.amount_gbp <= max_amount)
        if search:
            q = q.filter(or_(
                Transaction.merchant_description.ilike(f'%{search}%'),
                Transaction.customer_id.ilike(f'%{search}%'),
            ))
        if low_confidence:
            q = q.filter(TransactionClassification.confidence < 0.7)
        if manual_only:
            q = q.filter(TransactionClassification.is_manual == True)
        total = q.count()
        items = q.order_by(Transaction.transaction_date.desc()).offset((page - 1) * per_page).limit(per_page).all()
        return items, total

    def get_transaction(self, transaction_id: str):
        return Transaction.query.get(transaction_id)

    def get_classification(self, transaction_id: str):
        return TransactionClassification.query.filter_by(transaction_id=transaction_id).first()

    def update_classification(self, transaction_id: str, category: str):
        cls = TransactionClassification.query.filter_by(transaction_id=transaction_id).first()
        if not cls:
            cls = TransactionClassification(transaction_id=transaction_id)
            db.session.add(cls)
        cls.category = category
        cls.is_manual = True
        cls.status = 'manual'
        cls.confidence = 1.0
        db.session.commit()
        return cls

    def category_summary(self, customer_id=None):
        q = db.session.query(
            TransactionClassification.category,
            func.count(Transaction.transaction_id).label('count'),
            func.sum(Transaction.amount_gbp).label('total_gbp')
        ).join(Transaction, Transaction.transaction_id == TransactionClassification.transaction_id)
        if customer_id:
            q = q.filter(Transaction.customer_id == customer_id)
        return q.group_by(TransactionClassification.category).all()

    def monthly_summary(self, customer_id=None, year=2023):
        q = db.session.query(
            func.strftime('%Y-%m', Transaction.transaction_date).label('month'),
            func.count(Transaction.transaction_id).label('count'),
            func.sum(Transaction.amount_gbp).label('total_gbp')
        ).filter(func.strftime('%Y', Transaction.transaction_date) == str(year))
        if customer_id:
            q = q.filter(Transaction.customer_id == customer_id)
        return q.group_by('month').order_by('month').all()
