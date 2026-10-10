"""
EligibilityService: configurable mock product eligibility rules.
All results are DEMO ELIGIBILITY only, not actual lending decisions.
"""
from dataclasses import dataclass
from typing import Optional

DEMO_DISCLAIMER = 'DEMO ONLY: This is not an actual eligibility or lending decision.'


@dataclass
class EligibilityResult:
    customer_id: str
    product_code: str
    is_eligible: bool
    status: str  # eligible, excluded, borderline
    notes: str
    disclaimer: str = DEMO_DISCLAIMER


# Explicit exclusion rules per product
EXCLUSION_RULES = {
    'GP01': [  # EV Finance
        ('income_band', lambda v: v == 'Low', 'Income band too low for EV finance demo threshold'),
        ('age', lambda v: v < 21, 'Age below minimum demo threshold'),
    ],
    'GP02': [  # Green Home Loan
        ('is_homeowner', lambda v: not v, 'Homeownership required for Green Home Loan'),
        ('income_band', lambda v: v == 'Low', 'Income band too low for Green Home Loan demo threshold'),
    ],
    'GP03': [  # Solar Finance
        ('is_homeowner', lambda v: not v, 'Homeownership required for Solar Finance'),
        ('housing_type', lambda v: v == 'Flat', 'Solar Finance not applicable for flats in this demo'),
    ],
    'GP04': [  # Paperless Banking
        ('age', lambda v: v > 80, 'Age-based exclusion for paperless demo (accessibility consideration)'),
    ],
}


class EligibilityService:
    def check(self, customer: dict, product_code: str) -> EligibilityResult:
        rules = EXCLUSION_RULES.get(product_code, [])
        for field, check_fn, reason in rules:
            val = customer.get(field)
            if val is not None and check_fn(val):
                return EligibilityResult(
                    customer_id=customer['customer_id'],
                    product_code=product_code,
                    is_eligible=False,
                    status='excluded',
                    notes=reason,
                )
        return EligibilityResult(
            customer_id=customer['customer_id'],
            product_code=product_code,
            is_eligible=True,
            status='eligible',
            notes='Meets all demo eligibility criteria.',
        )
