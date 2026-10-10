"""
CarbonImpactService: configurable mock carbon engine.
All factors are ILLUSTRATIVE. Do not present as verified official values.
"""
from dataclasses import dataclass, asdict
from typing import Optional

FACTOR_VERSION = 'ILLUSTRATIVE-v1.0'

# Illustrative factors (kg CO2e per unit)
DEFAULT_FACTORS = {
    'petrol_per_litre': 2.31,        # kg CO2e / litre
    'diesel_per_litre': 2.68,        # kg CO2e / litre
    'electricity_per_kwh': 0.233,    # kg CO2e / kWh (UK grid 2023 illustrative)
    'gas_per_kwh': 0.183,            # kg CO2e / kWh
    'bus_per_km': 0.089,             # kg CO2e / km
    'rail_per_km': 0.035,            # kg CO2e / km
    'ev_per_km': 0.053,              # kg CO2e / km (well-to-wheel illustrative)
    'petrol_per_km': 0.192,          # kg CO2e / km (avg UK car)
    'diesel_per_km': 0.171,          # kg CO2e / km
}

FACTOR_NOTES = {
    'source': 'Illustrative factors for demo purposes only. Not based on official DESNZ/DEFRA values.',
    'year': 2023,
    'disclaimer': 'These are NOT official UK government conversion factors. Results are for demonstration only.',
}

# Fuel price assumption (GBP/litre) for spend -> litres conversion
FUEL_PRICE_PER_LITRE = 1.55
ELECTRICITY_PRICE_PER_KWH = 0.28
GAS_PRICE_PER_KWH = 0.065
BUS_FARE_PER_KM = 0.18
RAIL_FARE_PER_KM = 0.22


@dataclass
class ImpactResult:
    customer_id: str
    baseline_kg_co2e: float
    projected_kg_co2e: float
    reduction_kg_co2e: float
    fuel_baseline_kg: float
    electricity_baseline_kg: float
    gas_baseline_kg: float
    transport_baseline_kg: float
    fuel_projected_kg: float
    electricity_projected_kg: float
    gas_projected_kg: float
    transport_projected_kg: float
    assumptions: dict
    factor_version: str

    def to_dict(self):
        return asdict(self)


class CarbonImpactService:
    def __init__(self, factors: Optional[dict] = None, factor_version: str = FACTOR_VERSION):
        self.factors = factors or DEFAULT_FACTORS.copy()
        self.factor_version = factor_version

    def compute_baseline(self, features: dict, customer: dict) -> ImpactResult:
        """Compute baseline emissions from customer spending features."""
        # Fuel
        monthly_fuel_spend = features.get('monthly_fuel_spend_gbp', 0)
        annual_fuel_spend = monthly_fuel_spend * 12
        litres = annual_fuel_spend / FUEL_PRICE_PER_LITRE
        vehicle_type = customer.get('vehicle_type', 'Petrol')
        if vehicle_type == 'Diesel':
            fuel_kg = litres * self.factors['diesel_per_litre']
        elif vehicle_type in ('Electric', 'EV'):
            fuel_kg = 0.0
        elif vehicle_type == 'Hybrid':
            fuel_kg = litres * 0.5 * self.factors['petrol_per_litre']
        else:
            fuel_kg = litres * self.factors['petrol_per_litre']

        # Electricity
        monthly_elec_spend = features.get('monthly_electricity_spend_gbp', 0)
        annual_kwh_elec = (monthly_elec_spend * 12) / ELECTRICITY_PRICE_PER_KWH
        elec_kg = annual_kwh_elec * self.factors['electricity_per_kwh']

        # Gas
        monthly_gas_spend = features.get('monthly_gas_spend_gbp', 0)
        annual_kwh_gas = (monthly_gas_spend * 12) / GAS_PRICE_PER_KWH
        gas_kg = annual_kwh_gas * self.factors['gas_per_kwh']

        # Transport
        monthly_transport_spend = features.get('monthly_transport_spend_gbp', 0)
        # Estimate mix: 60% rail, 40% bus
        annual_transport_spend = monthly_transport_spend * 12
        rail_km = (annual_transport_spend * 0.6) / RAIL_FARE_PER_KM
        bus_km = (annual_transport_spend * 0.4) / BUS_FARE_PER_KM
        transport_kg = rail_km * self.factors['rail_per_km'] + bus_km * self.factors['bus_per_km']

        # Projections (conservative: EV reduces fuel by 80%, solar reduces elec by 30%)
        fuel_projected = fuel_kg * 0.2 if customer.get('is_homeowner') else fuel_kg
        elec_projected = elec_kg * 0.7 if customer.get('is_homeowner') else elec_kg
        gas_projected = gas_kg * 0.85  # insulation improvements
        transport_projected = transport_kg  # no change assumed by default

        baseline = fuel_kg + elec_kg + gas_kg + transport_kg
        projected = fuel_projected + elec_projected + gas_projected + transport_projected
        reduction = max(0.0, baseline - projected)

        assumptions = {
            'fuel_price_per_litre_gbp': FUEL_PRICE_PER_LITRE,
            'electricity_price_per_kwh_gbp': ELECTRICITY_PRICE_PER_KWH,
            'gas_price_per_kwh_gbp': GAS_PRICE_PER_KWH,
            'transport_rail_share': 0.6,
            'factors': self.factors,
            'factor_notes': FACTOR_NOTES,
            'disclaimer': 'All values are illustrative estimates for demonstration. Not official calculations.',
        }

        return ImpactResult(
            customer_id=customer.get('customer_id', ''),
            baseline_kg_co2e=round(baseline, 2),
            projected_kg_co2e=round(projected, 2),
            reduction_kg_co2e=round(reduction, 2),
            fuel_baseline_kg=round(fuel_kg, 2),
            electricity_baseline_kg=round(elec_kg, 2),
            gas_baseline_kg=round(gas_kg, 2),
            transport_baseline_kg=round(transport_kg, 2),
            fuel_projected_kg=round(fuel_projected, 2),
            electricity_projected_kg=round(elec_projected, 2),
            gas_projected_kg=round(gas_projected, 2),
            transport_projected_kg=round(transport_projected, 2),
            assumptions=assumptions,
            factor_version=self.factor_version,
        )

    def compute_scenario(self, features: dict, customer: dict, scenario: dict) -> ImpactResult:
        """Recompute with user-specified scenario parameters."""
        mod_features = features.copy()
        mod_customer = customer.copy()

        annual_mileage = scenario.get('annual_mileage')
        annual_kwh = scenario.get('annual_kwh_electricity')
        ev_shift = scenario.get('ev_transport_shift', 0.0)  # 0-1

        if annual_mileage is not None:
            # Override fuel spend from mileage
            if mod_customer.get('vehicle_type') in ('Petrol', None, 'Unknown'):
                litres = annual_mileage * 0.08  # 8L/100km ~= UK avg
            else:
                litres = annual_mileage * 0.07
            mod_features['monthly_fuel_spend_gbp'] = (litres * FUEL_PRICE_PER_LITRE) / 12

        if annual_kwh is not None:
            mod_features['monthly_electricity_spend_gbp'] = (annual_kwh * ELECTRICITY_PRICE_PER_KWH) / 12

        result = self.compute_baseline(mod_features, mod_customer)

        if ev_shift > 0:
            # Shift fraction of fuel to EV equivalent
            fuel_shifted_kg = result.fuel_baseline_kg * ev_shift
            ev_kg = (annual_mileage or 10000) * ev_shift * self.factors['ev_per_km']
            fuel_new = result.fuel_baseline_kg - fuel_shifted_kg + ev_kg
            result.fuel_projected_kg = round(max(0, fuel_new), 2)
            result.projected_kg_co2e = round(
                result.fuel_projected_kg + result.electricity_projected_kg +
                result.gas_projected_kg + result.transport_projected_kg, 2
            )
            result.reduction_kg_co2e = round(
                max(0, result.baseline_kg_co2e - result.projected_kg_co2e), 2
            )

        return result

    def get_factor_info(self) -> dict:
        return {
            'version': self.factor_version,
            'factors': self.factors,
            **FACTOR_NOTES,
        }
