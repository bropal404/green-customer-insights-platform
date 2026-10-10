"""
Transaction routes: update category.
"""
from flask import Blueprint, jsonify, request

from backend.models import db, Transaction
from backend.repositories.transaction_repo import TransactionRepository

transactions_bp = Blueprint('transactions', __name__)

_txn_repo = TransactionRepository()

ALLOWED_CATEGORIES = [
    'Fuel', 'Electricity', 'Gas', 'Public Transport',
    'Home Improvement', 'Automotive', 'Utilities', 'Other',
]


@transactions_bp.route('/transactions/<transaction_id>', methods=['PATCH'])
def update_transaction_category(transaction_id):
    try:
        txn = Transaction.query.get(transaction_id)
        if not txn:
            return jsonify({'error': 'Transaction not found'}), 404

        body = request.get_json(silent=True)
        if not body:
            return jsonify({'error': 'Request body must be JSON'}), 400

        category = body.get('category')
        if category is None:
            return jsonify({'error': 'Missing required field: category'}), 400
        if category not in ALLOWED_CATEGORIES:
            return jsonify({
                'error': f'Invalid category. Allowed values: {ALLOWED_CATEGORIES}',
            }), 400

        _txn_repo.update_classification(transaction_id, category)

        # Re-query to get the updated object
        txn = Transaction.query.get(transaction_id)
        return jsonify({'data': txn.to_dict()})

    except Exception as exc:
        return jsonify({'error': f'Failed to update transaction: {exc}'}), 500
