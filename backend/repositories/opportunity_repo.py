from datetime import datetime
from sqlalchemy import func
from backend.models import db, PropensityPrediction, Recommendation, LLMExplanation, Customer, CustomerFeatures


class OpportunityRepository:
    def get_predictions(self, customer_id=None, product_code=None, min_score=None,
                        max_score=None, page=1, per_page=50, region=None, status=None):
        q = db.session.query(PropensityPrediction)
        if customer_id:
            q = q.filter(PropensityPrediction.customer_id == customer_id)
        if product_code:
            q = q.filter(PropensityPrediction.product_code == product_code)
        if min_score is not None:
            q = q.filter(PropensityPrediction.propensity_score >= min_score)
        if max_score is not None:
            q = q.filter(PropensityPrediction.propensity_score <= max_score)
        if region or status:
            q = q.join(Customer, Customer.customer_id == PropensityPrediction.customer_id)
            if region:
                q = q.filter(Customer.region == region)
        if status:
            q = q.join(Recommendation,
                (Recommendation.customer_id == PropensityPrediction.customer_id) &
                (Recommendation.product_code == PropensityPrediction.product_code),
                isouter=True)
            q = q.filter(Recommendation.status == status)
        total = q.count()
        items = q.order_by(PropensityPrediction.propensity_score.desc()).offset((page - 1) * per_page).limit(per_page).all()
        return items, total

    def upsert_prediction(self, data: dict):
        existing = PropensityPrediction.query.filter_by(
            customer_id=data['customer_id'],
            product_code=data['product_code']
        ).first()
        if existing:
            for k, v in data.items():
                setattr(existing, k, v)
            existing.predicted_at = datetime.utcnow()
        else:
            existing = PropensityPrediction(**data)
            db.session.add(existing)
        db.session.commit()
        return existing

    def get_recommendation(self, customer_id: str, product_code: str):
        return Recommendation.query.filter_by(
            customer_id=customer_id, product_code=product_code
        ).first()

    def upsert_recommendation(self, customer_id: str, product_code: str, **kwargs):
        rec = self.get_recommendation(customer_id, product_code)
        if not rec:
            rec = Recommendation(customer_id=customer_id, product_code=product_code)
            db.session.add(rec)
        for k, v in kwargs.items():
            setattr(rec, k, v)
        db.session.commit()
        return rec

    def dismiss_recommendation(self, recommendation_id: int, reason: str = None):
        rec = Recommendation.query.get(recommendation_id)
        if rec:
            rec.status = 'dismissed'
            rec.dismissed_at = datetime.utcnow()
            rec.dismiss_reason = reason
            db.session.commit()
        return rec

    def upsert_explanation(self, recommendation_id: int, text: str, template: str = None):
        exp = LLMExplanation.query.filter_by(recommendation_id=recommendation_id).first()
        if not exp:
            exp = LLMExplanation(recommendation_id=recommendation_id)
            db.session.add(exp)
        exp.explanation_text = text
        exp.explanation_template = template
        exp.is_simulated = True
        db.session.commit()
        return exp

    def count_by_product(self, min_score=0.5):
        return db.session.query(
            PropensityPrediction.product_code,
            func.count(PropensityPrediction.customer_id)
        ).filter(PropensityPrediction.propensity_score >= min_score).group_by(PropensityPrediction.product_code).all()
