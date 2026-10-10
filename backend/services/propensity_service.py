"""
PropensityService: deterministic mock propensity scoring.
Scores are computed from customer features using fixed rules and a hash-based
perturbation so they remain consistent across API calls unless refreshed.
"""
import hashlib
import json
from dataclasses import dataclass
from typing import List

MODEL_VERSION = 'MOCK-v1.0'

PRODUCT_RULES = {
    'GP01': {  # EV Finance
        'label': 'EV Finance',
        'base_features': [
            ('has_vehicle', 0.3, True),
            ('monthly_fuel_spend_gbp', 0.25, 150),
            ('income_band_high', 0.2, True),
            ('age_band_prime', 0.15, True),
            ('digital_activity_score', 0.1, 0.6),
        ],
        'reason_map': {
            'monthly_fuel_spend_gbp': ('HIGH_FUEL_SPEND', 150),
            'estimated_annual_mileage': ('HIGH_ANNUAL_MILEAGE', 8000),
            'has_vehicle': ('EXISTING_VEHICLE', True),
            'income_band_high': ('HIGH_INCOME', True),
        },
    },
    'GP02': {  # Green Home Loan
        'label': 'Green Home Loan',
        'base_features': [
            ('is_homeowner', 0.4, True),
            ('monthly_electricity_spend_gbp', 0.2, 80),
            ('monthly_home_improvement_spend_gbp', 0.2, 50),
            ('income_band_high', 0.1, True),
            ('has_mortgage', 0.1, True),
        ],
        'reason_map': {
            'is_homeowner': ('HOMEOWNER', True),
            'monthly_electricity_spend_gbp': ('HIGH_ELECTRICITY_SPEND', 80),
            'monthly_home_improvement_spend_gbp': ('HIGH_UTILITY_SPEND', 50),
        },
    },
    'GP03': {  # Solar Finance
        'label': 'Solar Finance',
        'base_features': [
            ('is_homeowner', 0.35, True),
            ('monthly_electricity_spend_gbp', 0.3, 100),
            ('income_band_high', 0.2, True),
            ('household_size_large', 0.15, True),
        ],
        'reason_map': {
            'is_homeowner': ('HOMEOWNER', True),
            'monthly_electricity_spend_gbp': ('HIGH_ELECTRICITY_SPEND', 100),
            'monthly_utilities_spend_gbp': ('HIGH_UTILITY_SPEND', 60),
        },
    },
    'GP04': {  # Paperless Banking
        'label': 'Paperless Banking',
        'base_features': [
            ('paper_statements', 0.5, True),
            ('digital_activity_score', 0.3, 0.4),
            ('mobile_banking_active', 0.2, False),
        ],
        'reason_map': {
            'paper_statements': ('PAPER_STATEMENT_USER', True),
            'digital_activity_score': ('HIGH_DIGITAL_ACTIVITY', 0.4),
        },
    },
}


@dataclass
class PropensityResult:
    customer_id: str
    product_code: str
    propensity_score: float
    confidence_score: float
    reason_codes: List[str]
    model_version: str


class PropensityService:
    def __init__(self, model_version: str = MODEL_VERSION):
        self.model_version = model_version

    def _hash_perturbation(self, customer_id: str, product_code: str) -> float:
        """Deterministic small noise per (customer, product) pair."""
        h = hashlib.md5(f'{customer_id}:{product_code}:{self.model_version}'.encode()).hexdigest()
        return (int(h[:4], 16) / 65535) * 0.1 - 0.05  # [-0.05, 0.05]

    def _compute_score(self, customer: dict, features: dict, product_code: str) -> tuple:
        """Compute deterministic propensity score from features."""
        rules = PRODUCT_RULES.get(product_code)
        if not rules:
            return 0.0, [], 0.5

        score = 0.0
        reason_codes = []
        reason_map = rules['reason_map']

        # Derived feature flags
        income_band = customer.get('income_band', 'Medium')
        age = customer.get('age', 35)
        household_size = customer.get('household_size', 2)

        derived = {
            'has_vehicle': customer.get('has_vehicle', False),
            'is_homeowner': customer.get('is_homeowner', False),
            'has_mortgage': customer.get('has_mortgage', False),
            'paper_statements': customer.get('paper_statements', False),
            'mobile_banking_active': customer.get('mobile_banking_active', True),
            'income_band_high': income_band in ('High', 'Medium'),
            'age_band_prime': 25 <= age <= 55,
            'household_size_large': household_size >= 3,
            'monthly_fuel_spend_gbp': features.get('monthly_fuel_spend_gbp', 0),
            'monthly_electricity_spend_gbp': features.get('monthly_electricity_spend_gbp', 0),
            'monthly_home_improvement_spend_gbp': features.get('monthly_home_improvement_spend_gbp', 0),
            'monthly_utilities_spend_gbp': features.get('monthly_utilities_spend_gbp', 0),
            'digital_activity_score': customer.get('digital_activity_score', 0.5),
            'estimated_annual_mileage': customer.get('estimated_annual_mileage', 0),
        }

        for feat_name, weight, threshold in rules['base_features']:
            val = derived.get(feat_name, 0)
            if isinstance(threshold, bool):
                contrib = weight if bool(val) == threshold else 0
            elif isinstance(threshold, (int, float)):
                if val >= threshold:
                    contrib = weight * min(1.0, val / (threshold * 2))
                else:
                    contrib = 0
            else:
                contrib = 0
            score += contrib

        # Reason codes
        for feat, (code, threshold) in reason_map.items():
            val = derived.get(feat, 0)
            if isinstance(threshold, bool):
                if bool(val) == threshold:
                    reason_codes.append(code)
            elif val >= threshold:
                reason_codes.append(code)

        # Add noise
        noise = self._hash_perturbation(customer.get('customer_id', ''), product_code)
        score = max(0.01, min(0.99, score + noise))

        # Confidence: higher when more reason codes fire
        confidence = 0.55 + len(reason_codes) * 0.08
        confidence = max(0.5, min(0.95, confidence))

        return round(score, 4), reason_codes, round(confidence, 4)

    def score_customer(self, customer: dict, features: dict, product_code: str) -> PropensityResult:
        score, codes, conf = self._compute_score(customer, features, product_code)
        return PropensityResult(
            customer_id=customer['customer_id'],
            product_code=product_code,
            propensity_score=score,
            confidence_score=conf,
            reason_codes=codes,
            model_version=self.model_version,
        )

    def score_all_products(self, customer: dict, features: dict) -> List[PropensityResult]:
        return [self.score_customer(customer, features, pc) for pc in PRODUCT_RULES]
