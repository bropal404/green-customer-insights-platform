from sqlalchemy import or_, func
from backend.models import db, Customer, CustomerFeatures, Account, FinancialProduct


class CustomerRepository:
    def list_customers(self, page=1, per_page=20, region=None, segment=None,
                       income_band=None, search=None):
        q = Customer.query
        if region:
            q = q.filter(Customer.region == region)
        if segment:
            q = q.filter(Customer.customer_segment == segment)
        if income_band:
            q = q.filter(Customer.income_band == income_band)
        if search:
            q = q.filter(or_(
                Customer.customer_id.ilike(f'%{search}%'),
                Customer.first_name.ilike(f'%{search}%'),
                Customer.last_name.ilike(f'%{search}%'),
            ))
        total = q.count()
        items = q.order_by(Customer.customer_id).offset((page - 1) * per_page).limit(per_page).all()
        return items, total

    def get_customer(self, customer_id: str):
        return Customer.query.get(customer_id)

    def get_features(self, customer_id: str):
        return CustomerFeatures.query.filter_by(customer_id=customer_id).first()

    def get_accounts(self, customer_id: str):
        return Account.query.filter_by(customer_id=customer_id).all()

    def get_financial_products(self, customer_id: str):
        return FinancialProduct.query.filter_by(customer_id=customer_id).all()

    def count_by_region(self):
        return db.session.query(Customer.region, func.count(Customer.customer_id)).group_by(Customer.region).all()

    def count_by_segment(self):
        return db.session.query(Customer.customer_segment, func.count(Customer.customer_id)).group_by(Customer.customer_segment).all()
