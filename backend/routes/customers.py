"""
Customer routes: list, profile, transactions, refresh.
"""
import json
from datetime import date

from flask import Blueprint, jsonify, request

from backend.models import (
    db, Customer, Account, FinancialProduct, CustomerFeatures,
    PropensityPrediction, Recommendation, LLMExplanation, ProductCatalogue,
)
from backend.repositories.customer_repo import CustomerRepository
from backend.repositories.transaction_repo import TransactionRepository
from backend.repositories.opportunity_repo import OpportunityRepository
from backend.services.propensity_service import PropensityService
from backend.services.eligibility_service import EligibilityService
from backend.services.explanation_service import ExplanationService
from backend.config import Config

customers_bp = Blueprint('customers', __name__)

_customer_repo = CustomerRepository()
_txn_repo = TransactionRepository()
_opp_repo = OpportunityRepository()
_propensity_svc = PropensityService(model_version=Config.PROPENSITY_MODEL_VERSION)
_eligibility_svc = EligibilityService()
_explanation_svc = ExplanationService()

PRODUCTS = ['GP01', 'GP02', 'GP03', 'GP04']


@customers_bp.route('/customers', methods=['GET'])
def list_customers():
    try:
        region = request.args.get('region')
        age_band = request.args.get('age_band')
        income_band = request.args.get('income_band')
        segment = request.args.get('segment')
        search = request.args.get('search')
        try:
            page = max(1, int(request.args.get('page', 1)))
            per_page = min(100, max(1, int(request.args.get('per_page', 25))))
        except (ValueError, TypeError):
            return jsonify({'error': 'page and per_page must be integers'}), 400

        items, total = _customer_repo.list_customers(
            region=region,
            age_band=age_band,
            income_band=income_band,
            segment=segment,
            search=search,
            page=page,
            per_page=per_page,
        )

        return jsonify({
            'data': [c.to_dict() for c in items],
            'total': total,
            'page': page,
            'per_page': per_page,
            'pages': max(1, (total + per_page - 1) // per_page),
        })
    except Exception as exc:
        return jsonify({'error': f'Failed to list customers: {exc}'}), 500


@customers_bp.route('/customers/<customer_id>', methods=['GET'])
def get_customer(customer_id):
    try:
        customer = Customer.query.get(customer_id)
        if not customer:
            return jsonify({'error': 'Customer not found'}), 404

        accounts = [a.to_dict() for a in customer.accounts.all()]
        features = customer.features.to_dict() if customer.features else {}
        financial_products = [
            fp.to_dict()
            for fp in FinancialProduct.query.filter_by(customer_id=customer_id).all()
        ]
        predictions = [
            p.to_dict()
            for p in PropensityPrediction.query.filter_by(customer_id=customer_id).all()
        ]

        data = customer.to_dict()
        data['accounts'] = accounts
        data['features'] = features
        data['financial_products'] = financial_products
        data['predictions'] = predictions

        return jsonify({'data': data})
    except Exception as exc:
        return jsonify({'error': f'Failed to fetch customer: {exc}'}), 500


@customers_bp.route('/customers/<customer_id>/transactions', methods=['GET'])
def get_customer_transactions(customer_id):
    try:
        customer = Customer.query.get(customer_id)
        if not customer:
            return jsonify({'error': 'Customer not found'}), 404

        category = request.args.get('category')
        search = request.args.get('search')
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        min_amount_str = request.args.get('min_amount')
        max_amount_str = request.args.get('max_amount')

        try:
            page = max(1, int(request.args.get('page', 1)))
            per_page = min(100, max(1, int(request.args.get('per_page', 25))))
        except (ValueError, TypeError):
            return jsonify({'error': 'page and per_page must be integers'}), 400

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

        min_amount = None
        max_amount = None
        if min_amount_str is not None:
            try:
                min_amount = float(min_amount_str)
            except ValueError:
                return jsonify({'error': 'min_amount must be a number'}), 400
        if max_amount_str is not None:
            try:
                max_amount = float(max_amount_str)
            except ValueError:
                return jsonify({'error': 'max_amount must be a number'}), 400

        items, total = _txn_repo.list_transactions(
            customer_id=customer_id,
            category=category,
            start_date=start_date,
            end_date=end_date,
            min_amount=min_amount,
            max_amount=max_amount,
            search=search,
            page=page,
            per_page=per_page,
        )

        return jsonify({
            'data': [t.to_dict() for t in items],
            'total': total,
            'page': page,
            'per_page': per_page,
            'pages': max(1, (total + per_page - 1) // per_page),
        })
    except Exception as exc:
        return jsonify({'error': f'Failed to fetch transactions: {exc}'}), 500


@customers_bp.route('/customers/<customer_id>/refresh', methods=['POST'])
def refresh_customer(customer_id):
    try:
        customer = Customer.query.get(customer_id)
        if not customer:
            return jsonify({'error': 'Customer not found'}), 404

        features_obj = CustomerFeatures.query.filter_by(customer_id=customer_id).first()
        if not features_obj:
            return jsonify({'error': 'Customer features not found — run data seed first'}), 404

        customer_dict = customer.to_dict()
        features_dict = features_obj.to_dict()

        results = []
        for product_code in PRODUCTS:
            # Propensity score
            prop_result = _propensity_svc.score_customer(customer_dict, features_dict, product_code)

            # Eligibility
            elig_result = _eligibility_svc.check(customer_dict, product_code)

            # Upsert prediction
            pred = _opp_repo.upsert_prediction({
                'customer_id': customer_id,
                'product_code': product_code,
                'propensity_score': prop_result.propensity_score,
                'confidence_score': prop_result.confidence_score,
                'reason_codes': json.dumps(prop_result.reason_codes),
                'eligibility_status': elig_result.status,
                'eligibility_notes': elig_result.notes,
                'model_version': prop_result.model_version,
            })

            # Upsert recommendation
            rec = _opp_repo.upsert_recommendation(
                customer_id=customer_id,
                product_code=product_code,
                status='active' if elig_result.is_eligible else 'excluded',
            )

            # Generate explanation
            explanation_text = _explanation_svc.generate(
                product_code, prop_result.reason_codes, customer_dict
            )
            _opp_repo.upsert_explanation(
                recommendation_id=rec.id,
                text=explanation_text,
                template=product_code,
            )

            results.append(pred.to_dict())

        return jsonify({
            'customer_id': customer_id,
            'refreshed': True,
            'predictions': results,
        })
    except Exception as exc:
        return jsonify({'error': f'Refresh failed: {exc}'}), 500
