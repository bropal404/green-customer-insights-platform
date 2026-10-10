"""
Data quality routes: dataset health and classification confidence stats.
"""
from flask import Blueprint, jsonify
from sqlalchemy import func

from backend.models import (
    db, Customer, Transaction, TransactionClassification,
    CustomerFeatures, DataGenerationRun,
)

data_quality_bp = Blueprint('data_quality', __name__)


@data_quality_bp.route('/data-quality', methods=['GET'])
def data_quality():
    try:
        # Basic counts
        customer_count = db.session.query(func.count(Customer.customer_id)).scalar() or 0
        transaction_count = db.session.query(func.count(Transaction.transaction_id)).scalar() or 0
        classification_count = db.session.query(func.count(TransactionClassification.id)).scalar() or 0

        # Missing field counts for customers
        missing_region = db.session.query(func.count(Customer.customer_id)).filter(
            (Customer.region == None) | (Customer.region == '')
        ).scalar() or 0
        missing_income_band = db.session.query(func.count(Customer.customer_id)).filter(
            (Customer.income_band == None) | (Customer.income_band == '')
        ).scalar() or 0
        missing_features = db.session.query(func.count(Customer.customer_id)).outerjoin(
            CustomerFeatures, CustomerFeatures.customer_id == Customer.customer_id
        ).filter(CustomerFeatures.id == None).scalar() or 0

        # Classification confidence distribution
        confidence_bins = {
            'high_confidence_0_9_plus': db.session.query(func.count(TransactionClassification.id)).filter(
                TransactionClassification.confidence >= 0.9
            ).scalar() or 0,
            'medium_confidence_0_7_to_0_9': db.session.query(func.count(TransactionClassification.id)).filter(
                TransactionClassification.confidence >= 0.7,
                TransactionClassification.confidence < 0.9,
            ).scalar() or 0,
            'low_confidence_below_0_7': db.session.query(func.count(TransactionClassification.id)).filter(
                TransactionClassification.confidence < 0.7
            ).scalar() or 0,
            'manual_overrides': db.session.query(func.count(TransactionClassification.id)).filter(
                TransactionClassification.is_manual == True
            ).scalar() or 0,
        }

        # Unclassified transactions (in transactions table but no classification row)
        unclassified_count = transaction_count - classification_count

        # Category distribution
        cat_rows = db.session.query(
            TransactionClassification.category,
            func.count(TransactionClassification.id).label('count'),
        ).group_by(TransactionClassification.category).order_by(
            func.count(TransactionClassification.id).desc()
        ).all()
        category_distribution = [{'category': row.category, 'count': row.count} for row in cat_rows]

        # Last data generation run
        last_run = DataGenerationRun.query.order_by(DataGenerationRun.run_at.desc()).first()

        return jsonify({
            'dataset_info': {
                'customer_count': customer_count,
                'transaction_count': transaction_count,
                'classification_count': classification_count,
                'unclassified_count': max(0, unclassified_count),
                'feature_coverage': customer_count - missing_features,
            },
            'missing_fields': {
                'customers_missing_region': missing_region,
                'customers_missing_income_band': missing_income_band,
                'customers_missing_features': missing_features,
            },
            'classification_confidence': confidence_bins,
            'category_distribution': category_distribution,
            'last_data_generation_run': last_run.to_dict() if last_run else None,
        })

    except Exception as exc:
        return jsonify({'error': f'Failed to fetch data quality metrics: {exc}'}), 500
