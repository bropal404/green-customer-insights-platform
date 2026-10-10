from flask import Flask, jsonify
from flask_cors import CORS
from backend.models import db
from backend.config import Config
from backend.routes.health import health_bp
from backend.routes.customers import customers_bp
from backend.routes.transactions import transactions_bp
from backend.routes.opportunities import opportunities_bp
from backend.routes.impact import impact_bp
from backend.routes.analytics import analytics_bp
from backend.routes.data_quality import data_quality_bp
from backend.routes.models_metrics import models_bp


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    app.config['SQLALCHEMY_DATABASE_URI'] = config_class.DATABASE_URL
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)
    CORS(app, origins=config_class.CORS_ORIGINS)

    app.register_blueprint(health_bp, url_prefix='/api/v1')
    app.register_blueprint(customers_bp, url_prefix='/api/v1')
    app.register_blueprint(transactions_bp, url_prefix='/api/v1')
    app.register_blueprint(opportunities_bp, url_prefix='/api/v1')
    app.register_blueprint(impact_bp, url_prefix='/api/v1')
    app.register_blueprint(analytics_bp, url_prefix='/api/v1')
    app.register_blueprint(data_quality_bp, url_prefix='/api/v1')
    app.register_blueprint(models_bp, url_prefix='/api/v1')

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({'error': 'Not found', 'status': 404}), 404

    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({'error': str(e), 'status': 400}), 400

    @app.errorhandler(500)
    def server_error(e):
        return jsonify({'error': 'Internal server error', 'status': 500}), 500

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5050, debug=Config.DEBUG)
