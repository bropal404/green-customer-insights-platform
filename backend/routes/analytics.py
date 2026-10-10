"""
Analytics routes: customer segments and regional breakdowns.
"""
from flask import Blueprint, jsonify, request
from sqlalchemy import func

from backend.models import db, Customer, PropensityPrediction, Transaction

analytics_bp = Blueprint('analytics', __name__)


@analytics_bp.route('/analytics/segments', methods=['GET'])
def segments_breakdown():
    """
    Returns per-segment counts and propensity stats.
    """
    try:
        # Total customers per segment
        total_q = db.session.query(
            Customer.customer_segment.label('segment'),
            func.count(Customer.customer_id).label('count'),
        ).group_by(Customer.customer_segment).all()

        total_map = {row.segment: row.count for row in total_q}

        # Average propensity and high-propensity count per segment
        stats_q = db.session.query(
            Customer.customer_segment.label('segment'),
            func.avg(PropensityPrediction.propensity_score).label('avg_propensity'),
            func.sum(
                db.case((PropensityPrediction.propensity_score >= 0.6, 1), else_=0)
            ).label('high_propensity_count'),
        ).join(
            PropensityPrediction,
            PropensityPrediction.customer_id == Customer.customer_id,
        ).group_by(Customer.customer_segment).all()

        stats_map = {
            row.segment: {
                'avg_propensity': round(float(row.avg_propensity or 0), 4),
                'high_propensity_count': int(row.high_propensity_count or 0),
            }
            for row in stats_q
        }

        result = []
        for segment, count in total_map.items():
            stats = stats_map.get(segment, {'avg_propensity': 0.0, 'high_propensity_count': 0})
            result.append({
                'segment': segment,
                'count': count,
                'avg_propensity': stats['avg_propensity'],
                'high_propensity_count': stats['high_propensity_count'],
            })

        result.sort(key=lambda x: x['count'], reverse=True)
        return jsonify({'data': result})

    except Exception as exc:
        return jsonify({'error': f'Failed to compute segment analytics: {exc}'}), 500


@analytics_bp.route('/analytics/regions', methods=['GET'])
def regions_breakdown():
    """
    Returns per-region counts with optional product/score filtering.
    """
    try:
        product_code = request.args.get('product_code')
        min_score = None
        if request.args.get('min_score') is not None:
            try:
                min_score = float(request.args.get('min_score'))
            except ValueError:
                return jsonify({'error': 'min_score must be a number'}), 400

        # Total customers + total spend per region
        base_q = db.session.query(
            Customer.region,
            func.count(Customer.customer_id.distinct()).label('count'),
        ).group_by(Customer.region)

        spend_q = db.session.query(
            Customer.region,
            func.sum(Transaction.amount_gbp).label('total_spend_gbp'),
        ).join(
            Transaction,
            Transaction.customer_id == Customer.customer_id,
        ).group_by(Customer.region)

        base_rows = base_q.all()
        spend_rows = spend_q.all()

        spend_map = {row.region: round(float(row.total_spend_gbp or 0), 2) for row in spend_rows}

        # Average propensity per region with optional filters
        prop_q = db.session.query(
            Customer.region,
            func.avg(PropensityPrediction.propensity_score).label('avg_propensity'),
        ).join(
            PropensityPrediction,
            PropensityPrediction.customer_id == Customer.customer_id,
        )
        if product_code:
            prop_q = prop_q.filter(PropensityPrediction.product_code == product_code)
        if min_score is not None:
            prop_q = prop_q.filter(PropensityPrediction.propensity_score >= min_score)
        prop_q = prop_q.group_by(Customer.region)
        prop_rows = prop_q.all()
        prop_map = {row.region: round(float(row.avg_propensity or 0), 4) for row in prop_rows}

        result = [
            {
                'region': row.region,
                'count': row.count,
                'avg_propensity': prop_map.get(row.region, 0.0),
                'total_spend_gbp': spend_map.get(row.region, 0.0),
            }
            for row in base_rows
        ]
        result.sort(key=lambda x: x['count'], reverse=True)
        return jsonify({'data': result})

    except Exception as exc:
        return jsonify({'error': f'Failed to compute region analytics: {exc}'}), 500
