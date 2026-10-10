"""
Impact routes: get/compute carbon impact and scenario modeling.
"""
import json

from flask import Blueprint, jsonify, request

from backend.models import db, Customer, CustomerFeatures, ImpactEstimate, EmissionsFactor
from backend.repositories.impact_repo import ImpactRepository
from backend.services.carbon_service import CarbonImpactService
from backend.config import Config

impact_bp = Blueprint('impact', __name__)

_impact_repo = ImpactRepository()
_carbon_svc = CarbonImpactService(factor_version=Config.EMISSIONS_FACTOR_VERSION)


def _impact_result_to_dict(result, factor_info: dict) -> dict:
    """Serialise an ImpactResult dataclass to a dict including factor metadata."""
    return {
        'customer_id': result.customer_id,
        'baseline_kg_co2e': result.baseline_kg_co2e,
        'projected_kg_co2e': result.projected_kg_co2e,
        'reduction_kg_co2e': result.reduction_kg_co2e,
        'fuel_baseline_kg': result.fuel_baseline_kg,
        'electricity_baseline_kg': result.electricity_baseline_kg,
        'gas_baseline_kg': result.gas_baseline_kg,
        'transport_baseline_kg': result.transport_baseline_kg,
        'fuel_projected_kg': result.fuel_projected_kg,
        'electricity_projected_kg': result.electricity_projected_kg,
        'gas_projected_kg': result.gas_projected_kg,
        'transport_projected_kg': result.transport_projected_kg,
        'assumptions': result.assumptions,
        'factor_version': result.factor_version,
        'factor_info': factor_info,
    }


@impact_bp.route('/customers/<customer_id>/impact', methods=['GET'])
def get_customer_impact(customer_id):
    try:
        customer = Customer.query.get(customer_id)
        if not customer:
            return jsonify({'error': 'Customer not found'}), 404

        # Return cached estimate if available
        existing = _impact_repo.get_impact(customer_id)
        if existing:
            data = existing.to_dict()
            data['factor_info'] = _carbon_svc.get_factor_info()
            return jsonify({'data': data})

        # Compute fresh
        features_obj = CustomerFeatures.query.filter_by(customer_id=customer_id).first()
        if not features_obj:
            return jsonify({'error': 'Customer features not found — run data seed first'}), 404

        customer_dict = customer.to_dict()
        features_dict = features_obj.to_dict()

        result = _carbon_svc.compute_baseline(features_dict, customer_dict)

        # Persist
        _impact_repo.save_impact({
            'customer_id': customer_id,
            'baseline_kg_co2e': result.baseline_kg_co2e,
            'projected_kg_co2e': result.projected_kg_co2e,
            'reduction_kg_co2e': result.reduction_kg_co2e,
            'fuel_baseline_kg': result.fuel_baseline_kg,
            'electricity_baseline_kg': result.electricity_baseline_kg,
            'gas_baseline_kg': result.gas_baseline_kg,
            'transport_baseline_kg': result.transport_baseline_kg,
            'fuel_projected_kg': result.fuel_projected_kg,
            'electricity_projected_kg': result.electricity_projected_kg,
            'gas_projected_kg': result.gas_projected_kg,
            'transport_projected_kg': result.transport_projected_kg,
            'assumptions': result.assumptions,
            'factor_version': result.factor_version,
        })

        return jsonify({'data': _impact_result_to_dict(result, _carbon_svc.get_factor_info())})

    except Exception as exc:
        return jsonify({'error': f'Failed to fetch impact: {exc}'}), 500


@impact_bp.route('/customers/<customer_id>/impact/scenario', methods=['POST'])
def compute_impact_scenario(customer_id):
    try:
        customer = Customer.query.get(customer_id)
        if not customer:
            return jsonify({'error': 'Customer not found'}), 404

        features_obj = CustomerFeatures.query.filter_by(customer_id=customer_id).first()
        if not features_obj:
            return jsonify({'error': 'Customer features not found — run data seed first'}), 404

        body = request.get_json(silent=True)
        if not body:
            return jsonify({'error': 'Request body must be JSON'}), 400

        # Validate inputs
        scenario = {}
        if 'annual_mileage' in body:
            try:
                val = float(body['annual_mileage'])
                if val < 0:
                    raise ValueError
                scenario['annual_mileage'] = val
            except (ValueError, TypeError):
                return jsonify({'error': 'annual_mileage must be a non-negative number'}), 400

        if 'annual_kwh_electricity' in body:
            try:
                val = float(body['annual_kwh_electricity'])
                if val < 0:
                    raise ValueError
                scenario['annual_kwh_electricity'] = val
            except (ValueError, TypeError):
                return jsonify({'error': 'annual_kwh_electricity must be a non-negative number'}), 400

        if 'ev_transport_shift' in body:
            try:
                val = float(body['ev_transport_shift'])
                if not (0.0 <= val <= 1.0):
                    raise ValueError
                scenario['ev_transport_shift'] = val
            except (ValueError, TypeError):
                return jsonify({'error': 'ev_transport_shift must be a float between 0.0 and 1.0'}), 400

        customer_dict = customer.to_dict()
        features_dict = features_obj.to_dict()

        result = _carbon_svc.compute_scenario(features_dict, customer_dict, scenario)

        return jsonify({
            'data': _impact_result_to_dict(result, _carbon_svc.get_factor_info()),
            'scenario': scenario,
            'note': 'Scenario result is ephemeral and not persisted.',
        })

    except Exception as exc:
        return jsonify({'error': f'Failed to compute scenario: {exc}'}), 500
