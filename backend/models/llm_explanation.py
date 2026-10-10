from datetime import datetime
from .base import db


class LLMExplanation(db.Model):
    __tablename__ = 'llm_explanations'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    recommendation_id = db.Column(db.Integer, db.ForeignKey('recommendations.id'), nullable=False, unique=True)
    explanation_text = db.Column(db.Text, nullable=False)
    explanation_template = db.Column(db.String(50))
    generated_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_simulated = db.Column(db.Boolean, default=True)

    def to_dict(self):
        return {
            'id': self.id,
            'recommendation_id': self.recommendation_id,
            'explanation_text': self.explanation_text,
            'explanation_template': self.explanation_template,
            'generated_at': self.generated_at.isoformat() if self.generated_at else None,
            'is_simulated': self.is_simulated,
        }
