import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DATABASE_URL = os.getenv('DATABASE_URL', f'sqlite:///{BASE_DIR}/data/green_insights.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:8501').split(',')
    DATA_DIR = BASE_DIR / 'data' / 'synthetic'
    RANDOM_SEED = int(os.getenv('RANDOM_SEED', '42'))
    EMISSIONS_FACTOR_VERSION = os.getenv('EMISSIONS_FACTOR_VERSION', 'ILLUSTRATIVE-v1.0')
    PROPENSITY_MODEL_VERSION = os.getenv('PROPENSITY_MODEL_VERSION', 'MOCK-v1.0')
    CLASSIFICATION_MODEL_VERSION = os.getenv('CLASSIFICATION_MODEL_VERSION', 'RULES-v1.0')
    DEBUG = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
