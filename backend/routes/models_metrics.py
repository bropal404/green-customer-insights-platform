"""
Model metrics routes: versions, prediction stats, emissions config.
"""
from flask import Blueprint, jsonify
from sqlalchemy import func

from backend.models import (
    db, PropensityPrediction, EmissionsFactor, DataGenerationRun,
    ProductCatalogue, TransactionClassification,
)
from backend.config import Config

models_bp = Blueprint('models', __name__)


@models_bp.route('/models/metrics', methods=['GET'])
def model_metrics():
    try:
        # Model versions from config
        propensity_version = Config.PROPENSITY_MODEL_VERSION
        classification_version = Config.CLASSIFICATION_MODEL_VERSION
        emissions_version = Config.EMISSIONS_FACTOR_VERSION

        # Last prediction timestamp
        last_pred = db.session.query(func.max(PropensityPrediction.predicted_at)).scalar()

        # Total prediction count
        total_predictions = db.session.query(func.count(PropensityPrediction.id)).scalar() or 0

        # Prediction count per product
        per_product_rows = db.session.query(
            PropensityPrediction.product_code,
            func.count(PropensityPrediction.id).label('count'),
            func.avg(PropensityPrediction.propensity_score).label('avg_score'),
            func.sum(
                db.case((PropensityPrediction.propensity_score >= 0.6, 1), else_=0)
            ).label('high_propensity_count'),
        ).group_by(PropensityPrediction.product_code).all()

        cat_map = {row.product_code: row.product_name for row in ProductCatalogue.query.all()}

        prediction_counts_by_product = [
            {
                'product_code': row.product_code,
                'product_name': cat_map.get(row.product_code, row.product_code),
                'total_predictions': row.count,
                'avg_propensity_score': round(float(row.avg_score or 0), 4),
                'high_propensity_count': int(row.high_propensity_count or 0),
            }
            for row in per_product_rows
        ]

        # Emissions factor config from DB
        ef_rows = EmissionsFactor.query.all()
        emissions_factors = [ef.to_dict() for ef in ef_rows]

        # Classification stats
        total_classifications = db.session.query(
            func.count(TransactionClassification.id)
        ).scalar() or 0
        manual_classifications = db.session.query(
            func.count(TransactionClassification.id)
        ).filter(TransactionClassification.is_manual == True).scalar() or 0
        avg_confidence = db.session.query(
            func.avg(TransactionClassification.confidence)
        ).scalar()

        # Last data generation run
        last_run = DataGenerationRun.query.order_by(DataGenerationRun.run_at.desc()).first()

        return jsonify({
            'model_versions': {
                'propensity': propensity_version,
                'classification': classification_version,
                'emissions_factors': emissions_version,
            },
            'propensity_model': {
                'last_prediction_at': last_pred.isoformat() if last_pred else None,
                'total_predictions': total_predictions,
                'prediction_counts_by_product': prediction_counts_by_product,
            },
            'classification_model': {
                'total_classifications': total_classifications,
                'manual_overrides': manual_classifications,
                'avg_confidence': round(float(avg_confidence or 0), 4),
            },
            'emissions_factor_config': {
                'version': emissions_version,
                'factors': emissions_factors,
                'note': 'All factors are illustrative estimates for demonstration purposes only.',
            },
            'last_data_generation_run': last_run.to_dict() if last_run else None,
        })

    except Exception as exc:
        return jsonify({'error': f'Failed to fetch model metrics: {exc}'}), 500
