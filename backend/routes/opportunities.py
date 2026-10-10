"""
Opportunities routes: list predictions, per-customer, dismiss recommendation.
"""
from flask import Blueprint, jsonify, request
from sqlalchemy import func
from datetime import datetime

from backend.models import (
    db, Customer, PropensityPrediction, Recommendation,
    LLMExplanation, ProductCatalogue,
)
from backend.repositories.opportunity_repo import OpportunityRepository

opportunities_bp = Blueprint('opportunities', __name__)

_opp_repo = OpportunityRepository()


def _build_opportunity_item(pred, customer, rec, explanation, cat_map):
    """Assemble an opportunity response dict from ORM objects."""
    import json

    reason_codes = []
    if pred.reason_codes:
        try:
            reason_codes = json.loads(pred.reason_codes)
        except (ValueError, TypeError):
            reason_codes = []

    product_name = cat_map.get(pred.product_code, pred.product_code)

    return {
        'customer_id': pred.customer_id,
        'customer_name': f'{customer.first_name} {customer.last_name}' if customer else None,
        'region': customer.region if customer else None,
        'product_code': pred.product_code,
        'product_name': product_name,
        'propensity_score': pred.propensity_score,
        'confidence_score': pred.confidence_score,
        'reason_codes': reason_codes,
        'explanation': explanation.explanation_text if explanation else None,
        'recommendation_status': rec.status if rec else None,
        'recommendation_id': rec.id if rec else None,
        'eligibility_status': pred.eligibility_status,
        'eligibility_notes': pred.eligibility_notes,
        'estimated_annual_impact_kg': rec.estimated_annual_impact_kg if rec else None,
        'predicted_at': pred.predicted_at.isoformat() if pred.predicted_at else None,
    }


@opportunities_bp.route('/opportunities', methods=['GET'])
def list_opportunities():
    try:
        product_code = request.args.get('product_code')
        region = request.args.get('region')
        status = request.args.get('status')

        min_score = None
        max_score = None
        try:
            if request.args.get('min_score') is not None:
                min_score = float(request.args.get('min_score'))
            if request.args.get('max_score') is not None:
                max_score = float(request.args.get('max_score'))
            page = max(1, int(request.args.get('page', 1)))
            per_page = min(100, max(1, int(request.args.get('per_page', 25))))
        except (ValueError, TypeError) as exc:
            return jsonify({'error': f'Invalid query parameter: {exc}'}), 400

        preds, total = _opp_repo.get_predictions(
            product_code=product_code,
            min_score=min_score,
            max_score=max_score,
            region=region,
            status=status,
            page=page,
            per_page=per_page,
        )

        # Bulk-load required data
        cat_map = {row.product_code: row.product_name for row in ProductCatalogue.query.all()}
        customer_ids = list({p.customer_id for p in preds})
        customers = {c.customer_id: c for c in Customer.query.filter(Customer.customer_id.in_(customer_ids)).all()}

        items = []
        for pred in preds:
            customer = customers.get(pred.customer_id)
            rec = _opp_repo.get_recommendation(pred.customer_id, pred.product_code)
            explanation = LLMExplanation.query.filter_by(
                recommendation_id=rec.id
            ).first() if rec else None
            items.append(_build_opportunity_item(pred, customer, rec, explanation, cat_map))

        return jsonify({
            'data': items,
            'total': total,
            'page': page,
            'per_page': per_page,
            'pages': max(1, (total + per_page - 1) // per_page),
        })
    except Exception as exc:
        return jsonify({'error': f'Failed to list opportunities: {exc}'}), 500


@opportunities_bp.route('/customers/<customer_id>/opportunities', methods=['GET'])
def get_customer_opportunities(customer_id):
    try:
        customer = Customer.query.get(customer_id)
        if not customer:
            return jsonify({'error': 'Customer not found'}), 404

        preds = PropensityPrediction.query.filter_by(customer_id=customer_id).all()
        cat_map = {row.product_code: row.product_name for row in ProductCatalogue.query.all()}

        items = []
        for pred in preds:
            rec = _opp_repo.get_recommendation(customer_id, pred.product_code)
            explanation = LLMExplanation.query.filter_by(
                recommendation_id=rec.id
            ).first() if rec else None
            items.append(_build_opportunity_item(pred, customer, rec, explanation, cat_map))

        # Sort by propensity score descending
        items.sort(key=lambda x: x['propensity_score'] or 0, reverse=True)

        return jsonify({'data': items, 'total': len(items)})
    except Exception as exc:
        return jsonify({'error': f'Failed to fetch customer opportunities: {exc}'}), 500


@opportunities_bp.route('/recommendations/<int:recommendation_id>/dismiss', methods=['POST'])
def dismiss_recommendation(recommendation_id):
    try:
        rec = Recommendation.query.get(recommendation_id)
        if not rec:
            return jsonify({'error': 'Recommendation not found'}), 404

        body = request.get_json(silent=True) or {}
        reason = body.get('reason')

        updated = _opp_repo.dismiss_recommendation(recommendation_id, reason=reason)
        return jsonify({'data': updated.to_dict()})
    except Exception as exc:
        return jsonify({'error': f'Failed to dismiss recommendation: {exc}'}), 500
