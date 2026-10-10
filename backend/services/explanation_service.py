"""
ExplanationService: template-based recommendation explanations.
No external LLM used. Outputs refer only to supplied facts.
All outputs are SIMULATED for demonstration.
"""
from typing import List

SIMULATED_DISCLAIMER = '[SIMULATED EXPLANATION — FOR DEMONSTRATION ONLY]'

TEMPLATES = {
    'GP01': (
        'Based on this customer\'s spending patterns, {reasons_sentence} '
        'This suggests they may be a good candidate for an EV Finance product. '
        'Customers with similar profiles in this dataset have shown interest in electric vehicle financing. '
        f'{SIMULATED_DISCLAIMER}'
    ),
    'GP02': (
        'This customer {reasons_sentence} '
        'These factors indicate they may benefit from a Green Home Loan to finance energy-efficiency improvements. '
        f'{SIMULATED_DISCLAIMER}'
    ),
    'GP03': (
        'Analysis of this customer\'s profile indicates {reasons_sentence} '
        'A Solar Finance product may help them reduce long-term energy costs. '
        f'{SIMULATED_DISCLAIMER}'
    ),
    'GP04': (
        'This customer currently {reasons_sentence} '
        'Switching to Paperless Banking would reduce paper usage and may align with their digital preferences. '
        f'{SIMULATED_DISCLAIMER}'
    ),
}

REASON_SENTENCES = {
    'HIGH_FUEL_SPEND': 'their fuel spending is above the dataset average',
    'HIGH_ANNUAL_MILEAGE': 'they have a high estimated annual mileage',
    'EXISTING_VEHICLE': 'they already own a vehicle',
    'HOMEOWNER': 'they are a homeowner',
    'HIGH_ELECTRICITY_SPEND': 'their electricity spending is elevated',
    'HIGH_UTILITY_SPEND': 'they have significant utility and home spending',
    'HIGH_DIGITAL_ACTIVITY': 'they are active digital banking users',
    'PAPER_STATEMENT_USER': 'they currently receive paper statements',
    'HIGH_INCOME': 'their income band suggests financial capacity',
}


class ExplanationService:
    def generate(self, product_code: str, reason_codes: List[str], customer: dict) -> str:
        template = TEMPLATES.get(product_code, f'This customer may benefit from product {product_code}. {SIMULATED_DISCLAIMER}')
        readable = [REASON_SENTENCES.get(r, r.lower().replace('_', ' ')) for r in reason_codes]
        if readable:
            if len(readable) == 1:
                reasons_sentence = readable[0] + '.'
            else:
                reasons_sentence = ', '.join(readable[:-1]) + ', and ' + readable[-1] + '.'
        else:
            reasons_sentence = 'their profile matches the target segment for this product.'
        return template.format(reasons_sentence=reasons_sentence)
