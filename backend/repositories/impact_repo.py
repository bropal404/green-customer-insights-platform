from datetime import datetime
from backend.models import db, ImpactEstimate


class ImpactRepository:
    def get_impact(self, customer_id: str):
        return ImpactEstimate.query.filter_by(customer_id=customer_id).order_by(ImpactEstimate.computed_at.desc()).first()

    def save_impact(self, data: dict):
        existing = ImpactEstimate.query.filter_by(customer_id=data['customer_id']).first()
        if existing:
            for k, v in data.items():
                setattr(existing, k, v)
            existing.computed_at = datetime.utcnow()
        else:
            import json
            if 'assumptions' in data and isinstance(data['assumptions'], dict):
                data = data.copy()
                data['assumptions'] = json.dumps(data['assumptions'])
            existing = ImpactEstimate(**data)
            db.session.add(existing)
        db.session.commit()
        return existing
