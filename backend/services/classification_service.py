"""
TransactionClassificationService: merchant-rule-based classification.
Uses deterministic rules first, then a hash-based fallback.
"""
import hashlib
import re

MODEL_VERSION = 'RULES-v1.0'

VALID_CATEGORIES = [
    'Fuel', 'Electricity', 'Gas', 'Public Transport',
    'Home Improvement', 'Automotive', 'Utilities', 'Other'
]

# Rule patterns: (pattern, category, confidence)
RULES = [
    # Fuel
    (r'\b(bp|shell|esso|texaco|jet|moto|total|sainsburys fuel|tesco fuel|costco fuel|petrol|fuel|forecourt|gulf)\b', 'Fuel', 0.95),
    # Electricity
    (r'\b(british gas electric|eon electric|npower|scottishpower|edf|octopus energy|bulb|ovo energy|electricity|e\.on|e\.on electric|utility warehouse electric)\b', 'Electricity', 0.93),
    # Gas
    (r'\b(british gas|centrica|npower gas|scottishpower gas|gas supply|gas direct|utilita gas|calor gas|flogas|bord gas)\b', 'Gas', 0.92),
    # Public Transport
    (r'\b(tfl|tube|london underground|oyster|national rail|southern rail|gwr|avanti|lner|crosscountry|stagecoach|first bus|arriva|go ahead|transdev|megabus|national express|trainline|rail fare|bus fare|tram|metro|cork bus|dublin bus)\b', 'Public Transport', 0.94),
    # Home Improvement
    (r'\b(b&q|homebase|screwfix|toolstation|wickes|travis perkins|jewson|selco|tile giant|topps tiles|victorian plumbing|plumbworld|bathroom fitters|kitchen fitters|loft insulation|double glazing|boiler|heat pump|cavity wall)\b', 'Home Improvement', 0.91),
    # Automotive
    (r'\b(halfords|kwik fit|national tyres|arnold clark|autotrader|dvla|mot test|car service|car parts|eurocarparts|gsf car parts|motoring|breakdown|rac|aa membership|ford service|vauxhall service|bmw service|audi service|toyota service|volkswagen)\b', 'Automotive', 0.90),
    # Utilities
    (r'\b(thames water|southern water|severn trent|united utilities|welsh water|anglian water|water bill|virgin media|sky broadband|bt broadband|sky tv|netflix|broadband|internet|phone bill|council tax|tv licence)\b', 'Utilities', 0.88),
]

COMPILED_RULES = [(re.compile(p, re.IGNORECASE), cat, conf) for p, cat, conf in RULES]


class TransactionClassificationService:
    def __init__(self, model_version: str = MODEL_VERSION):
        self.model_version = model_version

    def _fallback_category(self, description: str) -> tuple:
        """Deterministic hash-based fallback for unmatched descriptions."""
        h = int(hashlib.md5(description.lower().strip().encode()).hexdigest(), 16)
        idx = h % 3  # 0,1,2 -> Utilities, Other, Other (mostly Other)
        if idx == 0:
            return 'Utilities', 0.52
        return 'Other', 0.55

    def classify(self, description: str) -> dict:
        """Classify a merchant description. Returns category and confidence."""
        for pattern, category, confidence in COMPILED_RULES:
            if pattern.search(description):
                return {
                    'category': category,
                    'confidence': confidence,
                    'model_version': self.model_version,
                    'status': 'auto',
                }
        cat, conf = self._fallback_category(description)
        return {
            'category': cat,
            'confidence': conf,
            'model_version': self.model_version,
            'status': 'auto',
        }

    def batch_classify(self, descriptions: list) -> list:
        return [self.classify(d) for d in descriptions]
