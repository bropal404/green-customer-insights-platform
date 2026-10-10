"""
Health and dashboard routes.
"""
from datetime import datetime, date

from flask import Blueprint, jsonify, request
from sqlalchemy import func, text

from backend.models import (
    db, Customer, Transaction, TransactionClassification,
    PropensityPrediction, ImpactEstimate, Recommendation, ProductCatalogue,
)

health_bp = Blueprint('health', __name__)


@health_bp.route('/health', methods=['GET'])
def health_check():
    db_status = 'ok'
    try:
        db.session.execute(text('SELECT 1'))
    except Exception as exc:
        db_status = f'error: {exc}'

    return jsonify({
        'status': 'ok' if db_status == 'ok' else 'degraded',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'db_status': db_status,
        'version': '1.0.0',
    })


@health_bp.route('/dashboard', methods=['GET'])
def dashboard():
    try:
        region = request.args.get('region')
        product = request.args.get('product')
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')

        start_date = None
        end_date = None
        if start_date_str:
            try:
                start_date = date.fromisoformat(start_date_str)
            except ValueError:
                return jsonify({'error': 'Invalid start_date format, use YYYY-MM-DD'}), 400
        if end_date_str:
            try:
                end_date = date.fromisoformat(end_date_str)
            except ValueError:
                return jsonify({'error': 'Invalid end_date format, use YYYY-MM-DD'}), 400

        # --- Total customers ---
        cust_q = db.session.query(func.count(Customer.customer_id))
        if region:
            cust_q = cust_q.filter(Customer.region == region)
        total_customers = cust_q.scalar() or 0

        # --- Total transactions ---
        txn_q = db.session.query(func.count(Transaction.transaction_id))
        if region:
            txn_q = txn_q.join(Customer, Customer.customer_id == Transaction.customer_id)
            txn_q = txn_q.filter(Customer.region == region)
        if start_date:
            txn_q = txn_q.filter(Transaction.transaction_date >= start_date)
        if end_date:
            txn_q = txn_q.filter(Transaction.transaction_date <= end_date)
        total_transactions = txn_q.scalar() or 0

        # --- High propensity count (score >= 0.6) ---
        hp_q = db.session.query(func.count(PropensityPrediction.id.distinct())).filter(
            PropensityPrediction.propensity_score >= 0.6
        )
        if product:
            hp_q = hp_q.filter(PropensityPrediction.product_code == product)
        if region:
            hp_q = hp_q.join(Customer, Customer.customer_id == PropensityPrediction.customer_id)
            hp_q = hp_q.filter(Customer.region == region)
        high_propensity_count = hp_q.scalar() or 0

        # --- Estimated CO2e reduction ---
        impact_q = db.session.query(func.sum(ImpactEstimate.reduction_kg_co2e))
        if region:
            impact_q = impact_q.join(Customer, Customer.customer_id == ImpactEstimate.customer_id)
            impact_q = impact_q.filter(Customer.region == region)
        estimated_co2e_reduction_kg = impact_q.scalar() or 0.0

        # --- Paperless opportunity count (GP04, score >= 0.6) ---
        paperless_q = db.session.query(func.count(PropensityPrediction.id)).filter(
            PropensityPrediction.product_code == 'GP04',
            PropensityPrediction.propensity_score >= 0.6,
        )
        if region:
            paperless_q = paperless_q.join(Customer, Customer.customer_id == PropensityPrediction.customer_id)
            paperless_q = paperless_q.filter(Customer.region == region)
        paperless_opportunity_count = paperless_q.scalar() or 0

        # --- Opportunities by product ---
        cat_map = {row.product_code: row.product_name for row in ProductCatalogue.query.all()}
        opp_q = db.session.query(
            PropensityPrediction.product_code,
            func.count(PropensityPrediction.id).label('count'),
        ).filter(PropensityPrediction.propensity_score >= 0.6)
        if product:
            opp_q = opp_q.filter(PropensityPrediction.product_code == product)
        if region:
            opp_q = opp_q.join(Customer, Customer.customer_id == PropensityPrediction.customer_id)
            opp_q = opp_q.filter(Customer.region == region)
        opp_rows = opp_q.group_by(PropensityPrediction.product_code).all()
        opportunities_by_product = [
            {
                'product_code': row.product_code,
                'product_name': cat_map.get(row.product_code, row.product_code),
                'count': row.count,
            }
            for row in opp_rows
        ]

        # --- Monthly trends (transactions) ---
        mt_q = db.session.query(
            func.strftime('%Y-%m', Transaction.transaction_date).label('month'),
            func.count(Transaction.transaction_id).label('count'),
            func.sum(Transaction.amount_gbp).label('total_gbp'),
        )
        if region:
            mt_q = mt_q.join(Customer, Customer.customer_id == Transaction.customer_id)
            mt_q = mt_q.filter(Customer.region == region)
        if start_date:
            mt_q = mt_q.filter(Transaction.transaction_date >= start_date)
        if end_date:
            mt_q = mt_q.filter(Transaction.transaction_date <= end_date)
        mt_rows = mt_q.group_by('month').order_by('month').all()
        monthly_trends = [
            {
                'month': row.month,
                'count': row.count,
                'total_gbp': round(row.total_gbp or 0.0, 2),
            }
            for row in mt_rows
        ]

        # --- Region distribution ---
        rd_q = db.session.query(
            Customer.region,
            func.count(Customer.customer_id).label('count'),
        )
        if region:
            rd_q = rd_q.filter(Customer.region == region)
        rd_rows = rd_q.group_by(Customer.region).order_by(func.count(Customer.customer_id).desc()).all()
        region_distribution = [{'region': row.region, 'count': row.count} for row in rd_rows]

        # --- High propensity by segment ---
        seg_q = db.session.query(
            Customer.customer_segment.label('segment'),
            func.count(PropensityPrediction.id.distinct()).label('count'),
        ).join(PropensityPrediction, PropensityPrediction.customer_id == Customer.customer_id).filter(
            PropensityPrediction.propensity_score >= 0.6
        )
        if region:
            seg_q = seg_q.filter(Customer.region == region)
        if product:
            seg_q = seg_q.filter(PropensityPrediction.product_code == product)
        seg_rows = seg_q.group_by(Customer.customer_segment).all()
        high_propensity_segments = [{'segment': row.segment, 'count': row.count} for row in seg_rows]

        # --- Recent recommendations (last 10) ---
        rec_rows = db.session.query(Recommendation).order_by(Recommendation.created_at.desc()).limit(10).all()
        recent_recommendations = [
            {
                'customer_id': r.customer_id,
                'product_code': r.product_code,
                'status': r.status,
            }
            for r in rec_rows
        ]

        return jsonify({
            'total_customers': total_customers,
            'total_transactions': total_transactions,
            'high_propensity_count': high_propensity_count,
            'estimated_co2e_reduction_kg': round(float(estimated_co2e_reduction_kg), 2),
            'paperless_opportunity_count': paperless_opportunity_count,
            'opportunities_by_product': opportunities_by_product,
            'monthly_trends': monthly_trends,
            'region_distribution': region_distribution,
            'high_propensity_segments': high_propensity_segments,
            'recent_recommendations': recent_recommendations,
        })

    except Exception as exc:
        return jsonify({'error': f'Dashboard query failed: {exc}'}), 500
