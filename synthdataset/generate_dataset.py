#!/usr/bin/env python3
"""
Synthetic data generator - Megathon X PS1 (Green Customer Insights Platform)
=============================================================================
Produces three CSVs that follow ./schemas/*.schema.json:

  1. customers.csv       - generated FIRST (demographics, housing, vehicle, finances, digital behaviour)
  2. transactions.csv    - generated SECOND, conditioned on each customer (income, rent/mortgage, loans,
                           energy, fuel, transport, DIY, car upkeep, bills, discretionary spend)
  3. product_labels.csv  - generated LAST (eligibility rules + latent propensity + simulated take-up)

After step 2 the six avg_monthly_*_spend columns in customers.csv are RECOMPUTED from the generated
transactions, so the two tables reconcile exactly.

ALL DATA IS SYNTHETIC. No real personal or banking data is used.

Run:   python generate_dataset.py --n 1500 --seed 42 --out ./data --schema-dir ./schemas
Check: python generate_dataset.py --validate-only --out ./data --schema-dir ./schemas

------------------------------------------------------------------------------
KEY ASSUMPTIONS (document these in your submission)
------------------------------------------------------------------------------
Window / as-of date : 12 months 2025-10-01 .. 2026-09-30, data_as_of = 2026-09-30.
                      Customers with tenure < 12 months only have transactions for their tenure.
Prices (indicative) : electricity 27p/kWh + 52p/day standing; gas 7.1p/kWh + 29p/day standing;
                      petrol 138p/L, diesel 146p/L; public EV charging 62p/kWh; EV 0.30 kWh/mile.
Energy demand       : scaled by floor area, household size, EPC factor (A=0.55 ... G=1.8), heating type;
                      solar customers import ~35% less electricity. Gas only if heating_type = GAS_BOILER.
Tax (net pay)       : approx. rUK 2026/27 income tax + employee NI; Scottish bands ignored.
Category boundaries : Fuel = petrol/diesel stations (MCC 5541/5542) AND public EV charging (5552).
                      Heating oil, water, broadband, mobile, council tax, TV licence, heat network = Utilities.
                      Garages, MOT, tyres, parts, car wash = Automotive. Insurance, parking, taxis, groceries,
                      dining, retail, furniture, subscriptions, travel = Other. DIY / building materials /
                      trades / garden centres = Home Improvement. Rent + mortgage = Housing.
                      Auto loan, credit card, BNPL repayments = Debt. Payroll/pension/client payments = Income.
Eligibility rules   : see eligibility() below (documented product catalogue rules).
Labels              : propensity = sigmoid(feature-based score + noise), intercept calibrated per product to a
                      target mean among eligible customers; take-up ~ Bernoulli(propensity) if eligible else 0.
Split               : customer-level 70/15/15 random split; transactions/labels inherit via customer_id.
"""
import argparse
import calendar
import json
import math
import os
import re
import sys
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:                      # tqdm is optional: fall back to a plain iterator
    def tqdm(it, **kw):
        return it

# ----------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------
AS_OF = date(2026, 9, 30)
WINDOW = [(2025, 10), (2025, 11), (2025, 12)] + [(2026, m) for m in range(1, 10)]  # 12 months

REGIONS = ["North East", "North West", "Yorkshire and The Humber", "East Midlands", "West Midlands",
           "East of England", "London", "South East", "South West", "Wales", "Scotland", "Northern Ireland"]
REG_P = np.array([4.0, 11.0, 8.2, 7.3, 8.9, 9.4, 13.4, 13.7, 8.4, 4.7, 8.2, 2.8])
REG_P = REG_P / REG_P.sum()
REG_INC = dict(zip(REGIONS, [.88, .93, .92, .95, .95, 1.05, 1.28, 1.12, .98, .90, .97, .90]))
REG_MORT = dict(zip(REGIONS, [.70, .80, .80, .85, .88, 1.05, 1.35, 1.15, 1.00, .80, .85, .70]))
REG_RENT = dict(zip(REGIONS, [700, 850, 780, 800, 850, 1100, 1650, 1200, 1050, 780, 850, 700]))
REG_CTAX = dict(zip(REGIONS, [170, 170, 165, 170, 175, 190, 165, 195, 200, 155, 150, 100]))
OIL_REGIONS = {"Scotland", "Wales", "South West", "East of England", "Yorkshire and The Humber", "North East"}

WATER = {"London": "THAMES WATER", "South East": "SOUTHERN WATER", "East of England": "ANGLIAN WATER",
         "South West": "SOUTH WEST WATER", "West Midlands": "SEVERN TRENT WATER",
         "East Midlands": "SEVERN TRENT WATER", "North West": "UNITED UTILITIES",
         "Yorkshire and The Humber": "YORKSHIRE WATER", "North East": "NORTHUMBRIAN WATER",
         "Wales": "DWR CYMRU WELSH WATER", "Scotland": "SCOTTISH WATER", "Northern Ireland": "NI WATER"}
COUNCILS = {"London": ["LB LAMBETH", "LB BARNET", "LB HACKNEY", "LB CROYDON", "LB EALING"],
            "South East": ["KENT CC", "SURREY HEATH BC", "BRIGHTON & HOVE CC", "OXFORD CC"],
            "East of England": ["NORWICH CC", "CAMBRIDGE CC", "COLCHESTER BC", "LUTON BC"],
            "South West": ["BRISTOL CC", "PLYMOUTH CC", "EXETER CC", "BATH & NE SOMERSET"],
            "East Midlands": ["NOTTINGHAM CC", "LEICESTER CC", "DERBY CC"],
            "West Midlands": ["BIRMINGHAM CC", "COVENTRY CC", "WOLVERHAMPTON CC"],
            "North West": ["MANCHESTER CC", "LIVERPOOL CC", "SALFORD CC", "WARRINGTON BC"],
            "Yorkshire and The Humber": ["LEEDS CC", "SHEFFIELD CC", "YORK CC", "BRADFORD MDC"],
            "North East": ["NEWCASTLE CC", "SUNDERLAND CC", "DURHAM CC"],
            "Wales": ["CARDIFF", "SWANSEA", "NEWPORT CC"],
            "Scotland": ["GLASGOW CC", "CITY OF EDINBURGH", "ABERDEEN CC"],
            "Northern Ireland": ["LAND & PROPERTY SERVICES RATES"]}

EPC_F = {"A": .55, "B": .70, "C": .90, "D": 1.05, "E": 1.25, "F": 1.50, "G": 1.80}
GAS_S = np.array([1.9, 1.7, 1.4, 1.0, 0.6, 0.4, 0.35, 0.35, 0.5, 0.9, 1.4, 1.8]); GAS_S = GAS_S / GAS_S.mean()
ELEC_S = np.array([1.15, 1.1, 1.0, 0.9, 0.85, 0.8, 0.8, 0.8, 0.9, 1.0, 1.1, 1.2]); ELEC_S = ELEC_S / ELEC_S.mean()

P_ELEC, P_ELEC_SC = 0.27, 0.52 * 365
P_GAS, P_GAS_SC = 0.071, 0.29 * 365
P_PETROL, P_DIESEL, P_EVPUB, EV_KWH_MILE = 1.38, 1.46, 0.62, 0.30

SUPPLIERS = ["OCTOPUS ENERGY", "BRITISH GAS", "EDF ENERGY", "E.ON NEXT", "OVO ENERGY", "SCOTTISH POWER", "SHELL ENERGY"]
BROADBAND = ["BT BROADBAND", "VIRGIN MEDIA", "SKY BROADBAND", "PLUSNET", "TALKTALK", "VODAFONE BROADBAND", "HYPEROPTIC"]
MOBILE = ["EE MOBILE", "O2 MOBILE", "VODAFONE MOBILE", "THREE UK", "GIFFGAFF", "TESCO MOBILE", "SMARTY"]
MORT_LENDERS = ["LLOYDS BANK MORTGAGE", "HALIFAX MORTGAGE", "NATIONWIDE BS MORTGAGE", "BARCLAYS MORTGAGE",
                "SANTANDER MORTGAGE", "NATWEST MORTGAGE", "HSBC MORTGAGE", "YORKSHIRE BS MORTGAGE",
                "COVENTRY BS MORTGAGE", "VIRGIN MONEY MORTGAGE"]
MORT_W = [.18, .17, .12, .09, .09, .09, .08, .06, .06, .06]
LOAN_LENDERS = ["BLACK HORSE VEHICLE FINANCE", "MOTONOVO FINANCE", "SANTANDER CONSUMER FINANCE",
                "CLOSE BROTHERS MOTOR FINANCE", "VOLKSWAGEN FINANCIAL SERVICES", "BMW FINANCIAL SERVICES"]
CARD_ISSUERS = ["BARCLAYCARD", "MBNA", "CAPITAL ONE", "LLOYDS BANK CREDIT CARD", "NATWEST CREDIT CARD", "AMERICAN EXPRESS"]
LETTINGS = ["FOXTONS LETTINGS RENT", "BELVOIR LETTINGS RENT", "LEXINGTON LETTINGS RENT", "LANDLORD RENT PAYMENT",
            "OPENRENT RENT", "HAMPTONS LETTINGS RENT", "CONNELLS LETTINGS RENT"]
EMPLOYERS = ["NORTHGATE LOGISTICS LTD", "BRIGHTWAY RETAIL GROUP", "CALDER ENGINEERING LTD", "NHS TRUST SALARY",
             "PENNINE HEALTHCARE LTD", "ASHFORD CONSULTING LLP", "HARBOUR TECH SOLUTIONS", "METRO COUNCIL PAYROLL",
             "UNIVERSITY PAYROLL", "REDWOOD FINANCIAL SERVICES", "SUMMIT CONSTRUCTION LTD", "OAKLEY EDUCATION TRUST",
             "BLUEPEAK DIGITAL LTD", "STONEBRIDGE FOODS LTD", "KINGSLEY & PARTNERS LLP", "VERTEX MANUFACTURING",
             "GREENFIELD HOSPITALITY", "ATLAS TRANSPORT SERVICES", "HIGHLAND ENERGY SERVICES", "CITY BOROUGH COUNCIL",
             "MERIDIAN INSURANCE SERVICES", "SOUTHERN RAIL PAYROLL", "LANTERN CARE GROUP", "THAMES DATA SYSTEMS",
             "WESSEX PHARMA LTD", "ORCHARD SUPERMARKETS PLC", "DALTON & HAYES SOLICITORS", "CROWN POSTAL SERVICES"]
PENSION_PROV = ["STANDARD LIFE PENSION", "AVIVA PENSION", "LEGAL & GENERAL PENSION", "SCOTTISH WIDOWS PENSION",
                "NEST PENSION", "NHS PENSIONS", "TEACHERS PENSION", "CIVIL SERVICE PENSION"]
CLIENTS = ["CLIENT PAYMENT", "INVOICE PAYMENT", "FASTER PAYMENT FROM CLIENT"]

FUEL_ST = [("SHELL", 5541, 12), ("BP", 5541, 12), ("ESSO", 5541, 10), ("TEXACO", 5541, 5), ("JET", 5541, 4),
           ("GULF", 5541, 2), ("APPLEGREEN", 5541, 2), ("MFG", 5541, 3), ("TESCO PFS", 5541, 16),
           ("SAINSBURYS PETROL", 5541, 9), ("ASDA PETROL", 5541, 8), ("MORRISONS PETROL", 5541, 8),
           ("SHELL SELF SERVE", 5542, 5), ("BP EXPRESS PAY", 5542, 4)]
EV_CH = [("POD POINT", 5552, 3), ("BP PULSE", 5552, 4), ("SHELL RECHARGE", 5552, 3), ("INSTAVOLT", 5552, 2),
         ("GRIDSERVE", 5552, 2), ("OSPREY CHARGING", 5552, 1)]
GROCERY = [("TESCO STORES", 5411, 22), ("SAINSBURYS", 5411, 16), ("ASDA STORES", 5411, 12), ("MORRISONS", 5411, 10),
           ("LIDL GB", 5411, 9), ("ALDI", 5411, 9), ("CO-OP FOOD", 5411, 7), ("WAITROSE", 5411, 5),
           ("M&S SIMPLY FOOD", 5411, 5), ("ICELAND", 5411, 2), ("OCADO", 5411, 3)]
DINING = [("PRET A MANGER", 5814, 8), ("COSTA COFFEE", 5814, 8), ("GREGGS", 5814, 8), ("MCDONALDS", 5814, 7),
          ("STARBUCKS", 5814, 6), ("NANDOS", 5812, 6), ("WAGAMAMA", 5812, 4), ("PIZZA EXPRESS", 5812, 4),
          ("LOCAL RESTAURANT", 5812, 8), ("WETHERSPOON", 5813, 6), ("THE CROWN PUB", 5813, 5), ("LEON", 5814, 3)]
DELIVERY = [("DELIVEROO", 5814, 10), ("JUST EAT", 5814, 9), ("UBER EATS", 5814, 8), ("DOMINOS PIZZA", 5814, 6)]
SHOPPING = [("AMAZON", 5999, 20), ("EBAY", 5999, 4), ("NEXT RETAIL", 5651, 7), ("PRIMARK", 5651, 7),
            ("ARGOS", 5311, 6), ("JOHN LEWIS", 5311, 4), ("IKEA", 5712, 5), ("CURRYS", 5732, 4),
            ("TK MAXX", 5651, 4), ("H&M", 5651, 4), ("WHSMITH", 5942, 3), ("SPORTS DIRECT", 5941, 4),
            ("APPLE STORE", 5732, 3)]
HEALTH = [("BOOTS", 5912, 12), ("SUPERDRUG", 5912, 6), ("LLOYDS PHARMACY", 5912, 3), ("SPECSAVERS", 8043, 2)]
LEISURE = [("ODEON CINEMAS", 7832, 5), ("CINEWORLD", 7832, 3), ("VUE CINEMAS", 7832, 3), ("TICKETMASTER", 7922, 4),
           ("NATIONAL TRUST", 7991, 2), ("BOWLPLEX", 7933, 2), ("WATERSTONES", 5942, 3)]
MISC = [("UBER", 4121, 8), ("NCP PARKING", 7523, 6), ("RINGGO PARKING", 7523, 6), ("ROYAL MAIL", 9402, 3),
        ("YOUR LOCAL CONVENIENCE STORE", 5499, 6), ("POST OFFICE", 9402, 3)]
TRAVEL = [("BOOKING.COM", 7011, 6), ("EASYJET", 4511, 6), ("RYANAIR", 4511, 5), ("BRITISH AIRWAYS", 4511, 4),
          ("AIRBNB", 6513, 5), ("JET2 HOLIDAYS", 4722, 4), ("TUI UK", 4722, 3)]
SUBS = [("NETFLIX.COM", 4899, 10.99), ("SPOTIFY", 5815, 11.99), ("AMAZON PRIME", 5968, 8.99), ("DISNEY PLUS", 4899, 7.99),
        ("APPLE.COM/BILL", 5818, 2.99), ("NOW TV", 4899, 9.99), ("YOUTUBE PREMIUM", 4899, 12.99)]
GYMS = [("PUREGYM", 7997, 22.99), ("THE GYM GROUP", 7997, 19.99), ("DAVID LLOYD CLUBS", 7997, 79.00)]
DIY = [("B&Q", 5200, 22), ("WICKES", 5211, 16), ("SCREWFIX", 5251, 14), ("HOMEBASE", 5200, 8),
       ("TOOLSTATION", 5251, 6), ("TRAVIS PERKINS", 5211, 8), ("DOBBIES GARDEN CENTRE", 5261, 5)]
TRADES = [("A J PLUMBING & HEATING LTD", 1711), ("ROOFING & CLADDING LTD", 1761), ("BATHROOM FITTERS UK LTD", 1520),
          ("KITCHEN DESIGN STUDIO LTD", 1520), ("WINDOWS & DOORS LTD", 1799), ("ELECTRICAL CONTRACTORS LTD", 1731),
          ("PROPERTY MAINTENANCE SERVICES", 1799)]
GARAGES = [("KWIK FIT", 7538), ("HALFORDS AUTOCENTRE", 7538), ("LOCAL GARAGE LTD", 7538), ("ATS EUROMASTER", 7538),
           ("MAIN DEALER SERVICING", 7538), ("QUICKFIT AUTO SERVICES", 7538)]
PARTS = [("HALFORDS", 5533, 10), ("EURO CAR PARTS", 5533, 10), ("AUTODOC", 5533, 3), ("GSF CAR PARTS", 5533, 5)]
INSURERS_CAR = ["ADMIRAL INSURANCE", "AVIVA CAR INSURANCE", "DIRECT LINE MOTOR", "LV= CAR INSURANCE", "CHURCHILL INSURANCE"]
INSURERS_HOME = ["AVIVA HOME INSURANCE", "DIRECT LINE HOME", "LV= HOME INSURANCE", "ADMIRAL HOME INSURANCE", "HASTINGS DIRECT"]

TX_COLS = ["transaction_id", "customer_id", "timestamp", "amount", "merchant_name", "mcc",
           "transaction_type", "category_ground_truth"]
SPEND_CATS = {"Fuel": "avg_monthly_fuel_spend", "Electricity": "avg_monthly_electricity_spend",
              "Gas": "avg_monthly_gas_spend", "Public Transport": "avg_monthly_public_transport_spend",
              "Home Improvement": "avg_monthly_home_improvement_spend", "Automotive": "avg_monthly_automotive_spend"}


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def weighted_choice(rng, labels, W):
    W = np.asarray(W, dtype=float)
    W = W / W.sum(axis=1, keepdims=True)
    c = W.cumsum(axis=1)
    u = rng.random((len(W), 1))
    idx = np.minimum((u > c).sum(axis=1), len(labels) - 1)
    return np.array(labels, dtype=object)[idx]


def net_annual(gross, retired):
    """Approximate rUK 2026/27 take-home (income tax + employee NI)."""
    gross = np.asarray(gross, dtype=float)
    pa = np.where(gross > 100000, np.maximum(0, 12570 - (gross - 100000) / 2), 12570)
    tx = np.maximum(0, gross - pa)
    tax = 0.20 * np.minimum(tx, 37700) + 0.40 * np.clip(tx - 37700, 0, 125140 - 37700) + 0.45 * np.maximum(tx - 125140, 0)
    ni = 0.08 * np.clip(gross - 12570, 0, 50270 - 12570) + 0.02 * np.maximum(gross - 50270, 0)
    ni = np.where(retired, 0, ni)
    return gross - tax - ni


def annuity_pmt(principal, i, n):
    return principal * i / (1 - (1 + i) ** (-n))


def month_days(y, m):
    return calendar.monthrange(y, m)[1]


def bizday(d, forward=True):
    """Shift a weekend date to a business day without leaving the month."""
    step = 1 if forward else -1
    e = d
    while e.weekday() >= 5:
        e += timedelta(days=step)
    if e.month != d.month:           # spilled over the month boundary -> shift the other way
        e = d
        while e.weekday() >= 5 or e.month != d.month:
            e -= timedelta(days=step)
    return e


# ----------------------------------------------------------------------------------------------
# STEP 1: customers
# ----------------------------------------------------------------------------------------------
def build_customers(n, rng):
    ids = np.array([f"C{i:05d}" for i in range(1, n + 1)], dtype=object)
    region = rng.choice(REGIONS, size=n, p=REG_P)
    london = region == "London"

    age = np.round(rng.normal(46, 17, n)).astype(int)
    age = np.where(age < 18, rng.integers(18, 31, n), age)
    age = np.minimum(age, 90)

    # employment & income
    p_ret = sigmoid((age - 65) / 3.0)
    emp = np.where(rng.random(n) < p_ret, "RETIRED", np.where(rng.random(n) < 0.13, "SELF_EMPLOYED", "EMPLOYED"))
    age_f = np.interp(age, [18, 22, 28, 35, 45, 52, 60, 70], [.55, .70, .95, 1.10, 1.20, 1.20, 1.05, .90])
    reg_m = np.array([REG_INC[r] for r in region])
    emp_inc = 31000 * age_f * reg_m * np.exp(rng.normal(0, .45, n))
    se_inc = 28000 * age_f * reg_m * np.exp(rng.normal(0, .60, n))
    state_pen = 11500 * rng.uniform(.85, 1.0, n)
    private = np.where(rng.random(n) < .70, 8000 * np.exp(rng.normal(0, .9, n)) * reg_m, 0.0)
    income = np.where(emp == "RETIRED", state_pen + private, np.where(emp == "SELF_EMPLOYED", se_inc, emp_inc))
    income = np.clip(income, 9000, 250000)
    income = np.where(age < 21, np.minimum(income, 32000), income)
    income = np.round(income / 100) * 100
    net_m = net_annual(income, emp == "RETIRED") / 12
    state_m = np.where(emp == "RETIRED", np.minimum(state_pen, income) / 12, 0.0)
    private_m = np.where(emp == "RETIRED", income / 12 - state_m, 0.0)

    # tenure
    max_t = np.maximum(6, (age - 16) * 12)
    tenure = np.clip(np.round(rng.beta(1.4, 2.0, n) * max_t), 1, 480).astype(int)

    # housing
    logit_o = -1.7 + 0.09 * (age - 20) + 0.000018 * (income - 30000) - np.where(london, 0.6, 0)
    owner = rng.random(n) < sigmoid(logit_o)
    outright = rng.random(n) < sigmoid(0.12 * (age - 58) + np.where(emp == "RETIRED", 0.5, 0))
    home = np.where(~owner, "RENTER", np.where(outright, "OWNER_OUTRIGHT", "OWNER_MORTGAGED")).astype(object)

    pt_labels = ["DETACHED", "SEMI_DETACHED", "TERRACED", "FLAT_APARTMENT"]
    W = np.where(owner[:, None], np.array([.28, .30, .26, .16]), np.array([.06, .20, .26, .48])).astype(float)
    W[london, 0] *= .35; W[london, 1] *= .7; W[london, 3] *= 2.2
    W[age < 30, 3] *= 1.4; W[age < 30, 0] *= .5
    ptype = weighted_choice(rng, pt_labels, W)
    flat = ptype == "FLAT_APARTMENT"
    det = ptype == "DETACHED"

    epc_l = list("ABCDEFG")
    W = np.tile(np.array([.007, .09, .40, .35, .12, .03, .003]), (n, 1))
    W[flat] *= np.array([2, 1.6, 1.2, .9, .6, .4, .3])
    W[det] *= np.array([.6, .7, .9, 1.0, 1.3, 1.6, 2.0])
    epc = weighted_choice(rng, epc_l, W)

    # heating
    heats = ["GAS_BOILER", "ELECTRIC", "OIL", "HEAT_PUMP", "OTHER"]
    W = np.tile(np.array([.76, .11, .06, .03, .04]), (n, 1))
    ni = region == "Northern Ireland"
    W[ni] = np.array([.12, .10, .65, .04, .09])
    oilr = np.isin(region, list(OIL_REGIONS))
    W[oilr, 2] *= 1.5
    W[london, 2] *= .1
    W[flat, 1] *= 2.5; W[flat, 2] *= .2; W[flat, 4] *= 1.5
    good_epc = np.isin(epc, ["A", "B"])
    W[good_epc, 3] *= 4; W[det, 3] *= 1.5
    heat = weighted_choice(rng, heats, W)

    # household, floor area, parking
    hh = np.zeros(n, dtype=int)
    for lo, hi, p in [(18, 29, [.35, .40, .15, .07, .03]), (30, 55, [.15, .30, .20, .25, .10]), (56, 100, [.35, .50, .08, .05, .02])]:
        m = (age >= lo) & (age <= hi)
        hh[m] = rng.choice([1, 2, 3, 4, 5], size=m.sum(), p=p)
    area_mean = np.select([ptype == "DETACHED", ptype == "SEMI_DETACHED", ptype == "TERRACED"], [130, 95, 85], 62)
    area = area_mean * np.exp(rng.normal(0, .18, n)) * (0.9 + 0.05 * hh) * np.where(london, .9, 1.0)
    area = np.round(np.clip(area, 28, 300), 1)

    p_park = np.select([ptype == "DETACHED", ptype == "SEMI_DETACHED", ptype == "TERRACED"], [.95, .85, .50], .35)
    p_park = p_park * np.where(london, .4, 1.0) * np.where(~owner, .9, 1.0)
    park = rng.random(n) < p_park

    # solar
    p_solar = (.03 + .05 * det + .02 * (ptype == "SEMI_DETACHED") + .02 * (income > 60000) + .02 * good_epc)
    solar_true = owner & ~flat & (rng.random(n) < p_solar)

    # vehicle
    p_none = np.clip(.12 + .22 * london + .15 * (age < 24) + .14 * (age > 78) + .12 * (income < 20000) + .06 * flat, .02, .70)
    has_veh = rng.random(n) >= p_none
    p_bev = np.clip((.045 + .035 * (income > 60000) + .02 * ((age >= 28) & (age <= 55))) * np.where(park, 1.8, .35), 0, .2)
    p_hyb = .12 + .04 * (income > 50000)
    p_dsl = np.clip(.27 - .10 * (age < 30) + .05 * (age >= 55), .05, .5)
    u = rng.random(n)
    veh = np.where(~has_veh, "NONE", np.where(u < p_bev, "BEV", np.where(u < p_bev + p_hyb, "HYBRID",
                   np.where(u < p_bev + p_hyb + p_dsl, "DIESEL", "PETROL")))).astype(object)
    is_bev = veh == "BEV"
    is_ice = np.isin(veh, ["PETROL", "DIESEL", "HYBRID"])

    miles = 6800 * np.exp(rng.normal(0, .55, n))
    miles *= np.where(veh == "DIESEL", 1.35, np.where(veh == "HYBRID", 1.05, 1.0))
    miles *= np.where(emp == "RETIRED", .70, 1.0) * np.where(london, .65, 1.0) * np.where(income > 60000, 1.1, 1.0)
    miles = np.where(has_veh, np.round(np.clip(miles, 1500, 30000) / 100) * 100, 0).astype(int)
    v_age = np.where(is_bev, np.clip(rng.gamma(2, 1.4, n), 0, 9), np.clip(rng.gamma(3, 2.8, n), 0, 22))
    v_age = np.floor(v_age).astype(int)

    mpg = np.where(veh == "PETROL", 42, np.where(veh == "DIESEL", 50, 58))
    price = np.where(veh == "DIESEL", P_DIESEL, P_PETROL)
    fuel_ice_m = np.where(is_ice, miles / mpg * 4.546 * price / 12, 0) * np.exp(rng.normal(0, .12, n))
    ev_kwh = np.where(is_bev, miles * EV_KWH_MILE, 0.0)
    ev_home_share = np.where(park, .9, 0.0)
    ev_home_kwh = ev_kwh * ev_home_share
    ev_pub_m = ev_kwh * (1 - ev_home_share) * P_EVPUB / 12
    fuel_m = fuel_ice_m + ev_pub_m

    # auto loan
    p_loan = np.where(v_age <= 4, .38, np.where(v_age <= 8, .12, .03)) * np.where(is_bev | (veh == "HYBRID"), 1.3, 1.0)
    p_loan = np.clip(p_loan * np.where(income < 18000, .5, 1.0), 0, .6)
    has_loan = has_veh & (rng.random(n) < p_loan)
    term = rng.choice([36, 48, 60], size=n, p=[.25, .35, .40])
    apr = rng.uniform(5.9, 12.9, n)
    i_l = apr / 1200
    principal = 14000 * np.exp(rng.normal(0, .45, n)) * np.where(is_bev, 1.4, 1.0) * (income / 30000) ** .25
    principal = np.clip(principal, 3000, 60000)
    pm = annuity_pmt(principal, i_l, term)
    principal = np.maximum(principal * np.minimum(1.0, .22 * net_m / pm), 1500)
    loan_pmt = annuity_pmt(principal, i_l, term)
    upper = np.clip((v_age + 1) * 12, 2, term)
    elapsed = 1 + np.floor(rng.random(n) * (upper - 1)).astype(int)
    rem = term - elapsed
    loan_bal = loan_pmt * (1 - (1 + i_l) ** (-rem)) / i_l

    # mortgage
    mort = home == "OWNER_MORTGAGED"
    rem_years = rng.uniform(5, np.maximum(6, np.minimum(32, 75 - age)))
    rem_n = np.round(rem_years * 12)
    i_m = rng.uniform(.02, .058, n) / 12
    frac = np.clip(1.05 - (age - 30) / 50, .15, 1.0)
    m_bal = income * np.exp(rng.normal(np.log(3.0), .35, n)) * frac * np.array([REG_MORT[r] for r in region])
    m_bal = np.clip(m_bal, 12000, 900000)
    m_pm = annuity_pmt(m_bal, i_m, rem_n)
    m_bal = np.maximum(m_bal * np.minimum(1.0, .36 * net_m / m_pm), 5000)
    mort_pmt = annuity_pmt(m_bal, i_m, rem_n)

    # investments & cash
    logit_i = -3.2 + 0.000035 * income + 0.03 * (age - 30) + 0.6 * (home == "OWNER_OUTRIGHT")
    has_inv = rng.random(n) < np.clip(sigmoid(logit_i), .01, .9)
    inv_bal = 6000 * (income / 35000) ** .9 * np.maximum(.5, 1 + (age - 30) / 25) * np.exp(rng.normal(0, 1.0, n))
    inv_bal = np.clip(inv_bal, 50, 1_500_000)
    cur_bal = net_m * np.exp(rng.normal(np.log(.9), .9, n))
    cur_bal = np.where(rng.random(n) < .03, rng.uniform(0, 80, n), cur_bal)
    cur_bal = np.clip(cur_bal, 0, 400000)

    # digital behaviour
    p_paper = np.interp(age, [18, 30, 45, 60, 70, 85], [.06, .10, .14, .30, .50, .65])
    paper = rng.random(n) < p_paper
    lam_app = np.interp(age, [18, 25, 35, 45, 55, 65, 80], [32, 30, 26, 20, 13, 7, 3]) * np.exp(rng.normal(0, .4, n))
    lam_app = np.where(paper, lam_app * .55, lam_app)
    app = rng.poisson(lam_app)
    lam_web = np.interp(age, [18, 35, 50, 65, 80], [2, 4, 5, 6, 5]) * np.exp(rng.normal(0, .5, n)) * np.where(paper, .8, 1.0)
    web = rng.poisson(lam_web)
    stmts = np.where(paper, np.where(rng.random(n) < .8, 12, 4), 0)

    # sustainability preference
    asked = rng.random(n) < .60
    p_green = sigmoid(-1.4 + .4 * (age < 40) + .000012 * (income - 30000) + .7 * has_inv)
    green = rng.random(n) < p_green

    # energy demand (latent annual GBP)
    epc_f = np.array([EPC_F[e] for e in epc])
    kwh = 2600 * (area / 90) ** .6 * (.75 + .1 * hh)
    kwh += np.where(heat == "ELECTRIC", 6500 * (area / 90) * epc_f, np.where(heat == "HEAT_PUMP", 2500 * (area / 90) * epc_f, 0))
    kwh += ev_home_kwh
    kwh = np.where(solar_true, kwh * .65, kwh)
    elec_annual = kwh * P_ELEC * np.exp(rng.normal(0, .12, n)) + P_ELEC_SC
    gas_kwh = 11000 * (area / 90) * epc_f * (.8 + .06 * hh)
    gas_annual = np.where(heat == "GAS_BOILER", gas_kwh * P_GAS * np.exp(rng.normal(0, .12, n)) + P_GAS_SC, 0.0)
    oil_annual = np.where(heat == "OIL", 1800 * (area / 90) * epc_f * .68 * np.exp(rng.normal(0, .12, n)), 0.0)
    heatnet_m = np.where(heat == "OTHER", 65 * (area / 90) * epc_f * np.exp(rng.normal(0, .12, n)), 0.0)

    # public transport latent
    pt_med = np.where(has_veh, np.where(london, 35, 14), np.where(london, 115, 60))
    pt_m = pt_med * np.exp(rng.normal(0, .6, n)) * np.where(emp == "RETIRED", .5, 1.0)
    pt_m = np.where(has_veh & ~london & (rng.random(n) < .25), 0.0, pt_m)
    pt_m = np.clip(pt_m, 0, 350)

    # rent
    rent = np.array([REG_RENT[r] for r in region]) * np.exp(rng.normal(0, .22, n)) * (.85 + .1 * hh)
    rent = np.maximum(np.minimum(rent, .42 * net_m), 300)
    rent = np.round(rent / 25) * 25

    # ---- observed (output) columns, with realistic nulls
    ptype_obs = np.where(rng.random(n) < .03, None, ptype).astype(object)
    epc_obs = np.where(rng.random(n) < .08, None, epc).astype(object)
    area_obs = np.where(rng.random(n) < .06, None, area).astype(object)
    park_obs = pd.Series([None if rng.random() < .03 else bool(p) for p in park], dtype=object)
    solar_obs = []
    for i in range(n):
        if ptype_obs[i] is None:
            solar_obs.append(None)
        elif not owner[i]:
            solar_obs.append(None if rng.random() < .05 else False)
        elif flat[i]:
            solar_obs.append(False)
        else:
            solar_obs.append(None if rng.random() < .04 else bool(solar_true[i]))
    green_obs = [bool(green[i]) if asked[i] else None for i in range(n)]

    def nn(mask, vals, rnd=None, as_int=False):
        out = []
        for m, v in zip(mask, vals):
            if not m:
                out.append(None)
            else:
                out.append(int(v) if as_int else (round(float(v), rnd) if rnd is not None else float(v)))
        return np.array(out, dtype=object)

    split = rng.choice(["train", "validation", "test"], size=n, p=[.70, .15, .15])

    C = pd.DataFrame({
        "customer_id": ids, "age": age, "income_annual_estimated": income.astype(float), "region": region,
        "tenure_months": tenure, "paper_statement_opt_in": paper, "has_current_account": True,
        "has_mortgage": mort, "has_auto_loan": has_loan, "has_investment_account": has_inv,
        "home_ownership_status": home, "property_type": ptype_obs, "epc_rating": epc_obs,
        "solar_installed": pd.Series(solar_obs, dtype=object), "current_vehicle_type": veh,
        "estimated_annual_mileage": miles, "auto_loan_months_remaining": nn(has_loan, rem, as_int=True),
        "current_account_balance": np.round(cur_bal, 2), "investment_account_balance": nn(has_inv, inv_bal, 2),
        "mortgage_balance_remaining": nn(mort, m_bal, 2), "auto_loan_balance_remaining": nn(has_loan, loan_bal, 2),
        "mobile_app_logins_monthly": app, "web_portal_logins_monthly": web,
        "avg_monthly_fuel_spend": 0.0, "avg_monthly_electricity_spend": 0.0, "avg_monthly_gas_spend": 0.0,
        "avg_monthly_public_transport_spend": 0.0, "avg_monthly_home_improvement_spend": 0.0,
        "avg_monthly_automotive_spend": 0.0,
        "heating_type": heat, "floor_area_sqm": area_obs, "has_off_street_parking": park_obs,
        "vehicle_age_years": nn(has_veh, v_age, as_int=True), "statements_per_year": stmts,
        "sustainability_preference_flag": pd.Series(green_obs, dtype=object),
        "data_as_of": AS_OF.isoformat(), "split": split,
    })
    # latent (true) drivers used by the transaction generator - NOT written to disk
    L = pd.DataFrame({
        "emp": emp, "net_m": net_m, "state_m": state_m, "private_m": private_m, "hh": hh, "owner": owner,
        "ptype": ptype, "epc": epc, "veh": veh, "vage": v_age, "fuel_m": fuel_m, "is_bev": is_bev,
        "has_loan": has_loan, "loan_pmt": loan_pmt, "loan_elapsed": elapsed, "mort": mort, "mort_pmt": mort_pmt,
        "elec_annual": elec_annual, "gas_annual": gas_annual, "oil_annual": oil_annual, "heatnet_m": heatnet_m,
        "pt_m": pt_m, "rent": rent, "heat": heat, "age": age, "income": income, "park": park,
        "london": london, "has_inv": has_inv, "cur_bal": cur_bal, "active": np.minimum(12, tenure),
        "region": region, "area": area, "solar": solar_true,
    })
    return C, L


# ----------------------------------------------------------------------------------------------
# STEP 2: transactions (conditioned on customers)
# ----------------------------------------------------------------------------------------------
class TxBuilder:
    def __init__(self, g):
        self.g = g
        self.rows = []

    def pick(self, items):
        w = np.array([it[-1] if len(it) == 3 else 1 for it in items], dtype=float)
        return items[int(self.g.choice(len(items), p=w / w.sum()))]

    def add(self, cid, dt, amount, merchant, mcc, ttype, cat):
        amt = round(float(amount), 2)
        if cat == "Income":
            amt = abs(amt) if abs(amt) >= .01 else .01
        else:
            amt = -abs(amt) if abs(amt) >= .01 else -.01
        self.rows.append((cid, dt, amt, merchant, mcc, ttype, cat))

    def dd(self, cid, y, m, day, amount, merchant, cat):
        d = bizday(date(y, m, min(day, 28)), forward=True)
        dt = datetime(d.year, d.month, d.day, int(self.g.integers(0, 7)), int(self.g.integers(0, 60)), int(self.g.integers(0, 60)))
        self.add(cid, dt, amount, merchant, None, "DIRECT_DEBIT", cat)

    def card_dt(self, y, m, mu=14, sd=4):
        g = self.g
        d = int(g.integers(1, month_days(y, m) + 1))
        h = int(np.clip(round(g.normal(mu, sd)), 7, 22))
        return datetime(y, m, d, h, int(g.integers(0, 60)), int(g.integers(0, 60)))

    def card(self, cid, y, m, amount, item, cat, mu=14, sd=4, suffix=True):
        name, mcc = item[0], item[1]
        if suffix and self.g.random() < .4:
            name = f"{name} {int(self.g.integers(1000, 9999))}"
        self.add(cid, self.card_dt(y, m, mu, sd), amount, name, f"{mcc:04d}", "CARD_PAYMENT", cat)

    def split(self, total, n, sigma=.5):
        w = np.exp(self.g.normal(0, sigma, n)); w = w / w.sum()
        return np.maximum(total * w, .5)

    def count(self, total, avg):
        if total < 1:
            return 0
        return max(1, int(self.g.poisson(total / avg)))


def gen_transactions(C, L, seed):
    g = np.random.default_rng(seed)
    T = TxBuilder(g)
    ri = lambda a, b: int(g.integers(a, b + 1))  # inclusive
    recs = C.to_dict("records")
    lats = L.to_dict("records")

    for c, l in tqdm(list(zip(recs, lats)), desc="transactions", unit="cust"):
        cid = c["customer_id"]
        months = WINDOW[-int(l["active"]):]
        nm = len(months)
        region, london = l["region"], bool(l["london"])
        net_m, hh, veh = l["net_m"], l["hh"], l["veh"]
        has_veh = veh != "NONE"
        epc_f = EPC_F[l["epc"]]

        # ---------- persistent per-customer choices
        employer = EMPLOYERS[ri(0, len(EMPLOYERS) - 1)]
        pay_day = int(g.choice([15, 20, 25, 26, 27, 28], p=[.08, .07, .2, .1, .15, .4]))
        sup_e = SUPPLIERS[ri(0, len(SUPPLIERS) - 1)]
        sup_g = sup_e if g.random() < .6 else SUPPLIERS[ri(0, len(SUPPLIERS) - 1)]
        e_name = ["{} ELEC DD", "{} ELECTRICITY", "{} - ELEC"][ri(0, 2)].format(sup_e)
        g_name = ["{} GAS DD", "{} GAS", "{} - GAS"][ri(0, 2)].format(sup_g)
        variable_pay = g.random() < .25   # variable (seasonal) vs fixed monthly energy direct debit
        d_elec, d_gas, d_water, d_bb, d_mob, d_ctax, d_tv, d_house, d_ins, d_loan, d_card, d_sub = [ri(1, 28) for _ in range(12)]

        # ---------- fixed monthly amounts
        water_m = max(15.0, g.normal(34, 8) * (1.15 if london else 1.0))
        bb_m = float(g.uniform(24, 48)) if g.random() < .92 else 0.0
        mob_m = float(g.uniform(8, 38)) if g.random() < .95 else 0.0
        tv_m = 14.54 if g.random() < .80 else 0.0
        ctax_m = REG_CTAX[region] * float(np.exp(g.normal(0, .25))) * (.75 if hh == 1 else 1.0)
        car_ins_m = float(np.exp(g.normal(np.log(62 * (1.5 if l["age"] < 25 else 1.0)), .3))) if has_veh else 0.0
        if l["owner"]:
            home_ins_m = float(np.exp(g.normal(np.log(20), .3))) if g.random() < .9 else 0.0
        else:
            home_ins_m = float(np.exp(g.normal(np.log(9), .3))) if g.random() < .5 else 0.0
        n_subs = int(g.choice([0, 1, 2, 3, 4], p=[.15, .25, .30, .20, .10]))
        sub_idx = g.choice(len(SUBS), size=n_subs, replace=False) if n_subs else []
        subs = [SUBS[int(i)] for i in sub_idx]
        if g.random() < .30:
            gname, gmcc, gamt = GYMS[ri(0, len(GYMS) - 1)]
            subs.append((gname, gmcc, gamt))
        has_cc = g.random() < .45
        cc_issuer = CARD_ISSUERS[ri(0, len(CARD_ISSUERS) - 1)]
        cc_m = min(float(np.exp(g.normal(np.log(.08 * net_m), .5))), .30 * net_m) if has_cc else 0.0
        bnpl = (l["age"] < 45) and g.random() < .10
        bnpl_m = float(g.uniform(25, 120)) if bnpl else 0.0
        bnpl_months = set(g.choice(nm, size=min(3, nm), replace=False).tolist()) if bnpl else set()
        mort_lender = MORT_LENDERS[int(g.choice(len(MORT_LENDERS), p=np.array(MORT_W) / sum(MORT_W)))]
        loan_lender = LOAN_LENDERS[ri(0, len(LOAN_LENDERS) - 1)]
        letting = LETTINGS[ri(0, len(LETTINGS) - 1)]
        council = COUNCILS[region][ri(0, len(COUNCILS[region]) - 1)]
        council_name = f"{council} COUNCIL TAX" if region != "Northern Ireland" else council
        bb_name, mob_name = BROADBAND[ri(0, len(BROADBAND) - 1)], MOBILE[ri(0, len(MOBILE) - 1)]
        car_ins_name, home_ins_name = INSURERS_CAR[ri(0, 4)], INSURERS_HOME[ri(0, 4)]
        pension_name = PENSION_PROV[ri(0, len(PENSION_PROV) - 1)]
        client_name = CLIENTS[ri(0, len(CLIENTS) - 1)]

        elec_const = l["elec_annual"] / 12
        gas_const = l["gas_annual"] / 12
        loan_pay_months = set(range(nm - min(nm, int(l["loan_elapsed"])), nm)) if l["has_loan"] else set()

        # ---------- expected committed spend -> discretionary budget
        committed = (l["rent"] if not l["owner"] else 0) + (l["mort_pmt"] if l["mort"] else 0) + \
                    (l["loan_pmt"] if l["has_loan"] else 0) + cc_m + elec_const + gas_const + l["oil_annual"] / 12 + \
                    l["heatnet_m"] + water_m + bb_m + mob_m + tv_m + ctax_m + car_ins_m + home_ins_m + \
                    sum(s[2] for s in subs) + l["fuel_m"] + l["pt_m"] + (l["fuel_m"] * 0 + (300 / 12 if has_veh else 0)) + 40
        spend_factor = float(g.uniform(.30, .70))
        disc_base = max(80.0, spend_factor * (net_m - committed))

        # ---------- yearly events (spread randomly over active months)
        def event_months(rate_per_year):
            k = int(g.poisson(rate_per_year * nm / 12))
            return [int(x) for x in g.integers(0, nm, size=k)]

        diy_rate = 3.0 if not l["owner"] else (10.0 if l["epc"] in "EFG" else 8.0)
        diy_events = event_months(diy_rate)
        big_p = (0.25 if (l["epc"] in "EFG" or l["income"] > 60000) else 0.12) * nm / 12 if l["owner"] else 0.0
        big_events = [int(g.integers(0, nm))] if g.random() < big_p else []
        auto_events = []
        if has_veh:
            bev_f = 0.6 if l["is_bev"] else 1.0
            if l["vage"] >= 3:
                auto_events.append((int(g.integers(0, nm)), "MOT", 54.85))
            if g.random() < nm / 12:
                auto_events.append((int(g.integers(0, nm)), "SERVICE", float(np.exp(g.normal(np.log(180 + 8 * l["vage"]), .35))) * bev_f))
            if g.random() < .28 * nm / 12:
                auto_events.append((int(g.integers(0, nm)), "TYRES", float(np.exp(g.normal(np.log(320), .3)))))
            if g.random() < min(.6, .12 + .03 * l["vage"]) * nm / 12:
                auto_events.append((int(g.integers(0, nm)), "REPAIR", float(np.exp(g.normal(np.log(450), .6)))))
            for k in event_months(2.5 * (1 + l["vage"] / 12)):
                auto_events.append((k, "PARTS", float(np.exp(g.normal(np.log(22), .5)))))
            for k in event_months(3):
                auto_events.append((k, "WASH", float(np.exp(g.normal(np.log(8), .3)))))
        travel_events = event_months(1.0) if g.random() < .55 else []
        oil_months = set(k for k in range(nm) if g.random() < 2 / 12) if l["oil_annual"] > 0 else set()

        # ---------- month loop
        for k, (y, m) in enumerate(months):
            # INCOME
            if l["emp"] == "EMPLOYED":
                d = bizday(date(y, m, pay_day), forward=False)
                T.add(cid, datetime(d.year, d.month, d.day, ri(0, 7), ri(0, 59), ri(0, 59)),
                      net_m * float(g.normal(1, .01)), f"{employer} PAYROLL", None, "CREDIT", "Income")
            elif l["emp"] == "SELF_EMPLOYED":
                tot = net_m * float(np.exp(g.normal(0, .25)))
                npay = ri(1, 4)
                for amt in T.split(tot, npay, .6):
                    d = date(y, m, ri(1, 28))
                    T.add(cid, datetime(d.year, d.month, d.day, ri(8, 18), ri(0, 59), ri(0, 59)), amt,
                          f"{client_name} {ri(1000, 9999)}", None, "CREDIT", "Income")
            else:
                d = bizday(date(y, m, 28 if pay_day >= 28 else pay_day), forward=False)
                if l["state_m"] > 0:
                    T.add(cid, datetime(d.year, d.month, d.day, ri(0, 6), ri(0, 59), ri(0, 59)), l["state_m"], "DWP STATE PENSION", None, "CREDIT", "Income")
                if l["private_m"] > 10:
                    T.add(cid, datetime(d.year, d.month, d.day, ri(0, 6), ri(0, 59), ri(0, 59)), l["private_m"], pension_name, None, "CREDIT", "Income")

            # HOUSING
            if not l["owner"]:
                T.dd(cid, y, m, d_house, l["rent"], letting, "Housing")
            elif l["mort"]:
                T.dd(cid, y, m, d_house, l["mort_pmt"], mort_lender, "Housing")

            # DEBT
            if k in loan_pay_months:
                T.dd(cid, y, m, d_loan, l["loan_pmt"], f"{loan_lender} DD", "Debt")
            if has_cc:
                T.dd(cid, y, m, d_card, cc_m * float(np.exp(g.normal(0, .25))), f"{cc_issuer} DD", "Debt")
            if k in bnpl_months:
                T.dd(cid, y, m, d_card, bnpl_m, "KLARNA DD", "Debt")

            # ENERGY
            fe = ELEC_S[m - 1] if variable_pay else 1.0
            T.dd(cid, y, m, d_elec, elec_const * fe * float(g.normal(1, .03)), e_name, "Electricity")
            if l["gas_annual"] > 0:
                fg = GAS_S[m - 1] if variable_pay else 1.0
                T.dd(cid, y, m, d_gas, gas_const * fg * float(g.normal(1, .03)), g_name, "Gas")
            if k in oil_months:
                T.card(cid, y, m, l["oil_annual"] / 2 * float(np.exp(g.normal(0, .15))), ("BOILERJUICE HEATING OIL", 5983), "Utilities", suffix=False)
            if l["heatnet_m"] > 0:
                T.dd(cid, y, m, d_gas, l["heatnet_m"] * float(g.normal(1, .02)), "HEAT NETWORK UTILITY DD", "Utilities")

            # UTILITIES
            T.dd(cid, y, m, d_water, water_m, f"{WATER[region]} DD", "Utilities")
            if bb_m:
                T.dd(cid, y, m, d_bb, bb_m, f"{bb_name} DD", "Utilities")
            if mob_m:
                T.dd(cid, y, m, d_mob, mob_m, f"{mob_name} DD", "Utilities")
            T.dd(cid, y, m, d_ctax, ctax_m, council_name, "Utilities")
            if tv_m:
                T.dd(cid, y, m, d_tv, tv_m, "TV LICENSING DD", "Utilities")

            # INSURANCE & SUBSCRIPTIONS (Other)
            if car_ins_m:
                T.dd(cid, y, m, d_ins, car_ins_m, f"{car_ins_name} DD", "Other")
            if home_ins_m:
                T.dd(cid, y, m, d_ins, home_ins_m, f"{home_ins_name} DD", "Other")
            for (sn, smcc, samt) in subs:
                dt = datetime(y, m, min(28, d_sub), ri(0, 23), ri(0, 59), ri(0, 59))
                T.add(cid, dt, samt, sn, f"{smcc:04d}", "CARD_PAYMENT", "Other")

            # FUEL / EV CHARGING
            if l["fuel_m"] > 5:
                tgt = l["fuel_m"] * float(np.exp(g.normal(0, .18)))
                if l["is_bev"]:
                    nfill, avg = T.count(tgt, 18), 18
                    items = EV_CH
                else:
                    nfill, avg = T.count(tgt, 55), 55
                    items = FUEL_ST
                if nfill:
                    for amt in T.split(tgt, nfill, .25):
                        T.card(cid, y, m, amt, T.pick(items), "Fuel", mu=13, sd=4)

            # PUBLIC TRANSPORT
            if l["pt_m"] > 3:
                tgt = l["pt_m"] * float(np.exp(g.normal(0, .25)))
                bus_part, rail_part = (.6 * tgt, .4 * tgt) if tgt > 40 else (tgt, 0.0)
                nb = min(16, max(1, int(round(bus_part / 4.5))))
                for amt in T.split(bus_part, nb, .35):
                    if london:
                        item = ("TFL TRAVEL CH", 4111)
                    else:
                        item = [("STAGECOACH", 4131), ("FIRST BUS", 4131), ("ARRIVA", 4131)][ri(0, 2)]
                    T.card(cid, y, m, amt, item, "Public Transport", mu=8, sd=3, suffix=False)
                if rail_part > 0:
                    nr = ri(1, 2)
                    for amt in T.split(rail_part, nr, .3):
                        T.card(cid, y, m, amt, [("TRAINLINE", 4112), ("NATIONAL RAIL", 4112), ("LNER", 4112), ("AVANTI WEST COAST", 4112)][ri(0, 3)], "Public Transport", mu=10, sd=4, suffix=False)

            # HOME IMPROVEMENT
            for ev in diy_events:
                if ev == k:
                    amt = float(np.exp(g.normal(np.log(35), .8)))
                    T.card(cid, y, m, amt, T.pick(DIY), "Home Improvement", mu=12, sd=3)
            for ev in big_events:
                if ev == k:
                    amt = float(np.clip(np.exp(g.normal(np.log(3500), .6)), 800, 25000))
                    if amt < .5 * l["income"]:
                        tr = TRADES[ri(0, len(TRADES) - 1)]
                        T.card(cid, y, m, amt, tr, "Home Improvement", mu=11, sd=2, suffix=False)

            # AUTOMOTIVE
            for (ek, kind, amt) in auto_events:
                if ek == k:
                    if kind == "MOT":
                        T.card(cid, y, m, amt, ("MOT TEST CENTRE", 7538), "Automotive", mu=11, sd=2)
                    elif kind in ("SERVICE", "REPAIR"):
                        T.card(cid, y, m, amt, GARAGES[ri(0, len(GARAGES) - 1)], "Automotive", mu=12, sd=2)
                    elif kind == "TYRES":
                        T.card(cid, y, m, amt, [("KWIK FIT", 5532), ("ATS EUROMASTER", 5532), ("NATIONAL TYRES", 5532)][ri(0, 2)], "Automotive", mu=12, sd=2)
                    elif kind == "PARTS":
                        T.card(cid, y, m, amt, T.pick(PARTS), "Automotive")
                    else:
                        T.card(cid, y, m, amt, ("SPLASH N GO CAR WASH", 7542), "Automotive", mu=13, sd=3)

            # DISCRETIONARY ('Other')
            season = 1.2 if m == 12 else 1.0
            disc = disc_base * season * float(np.exp(g.normal(0, .12)))
            for share, avg, items, mu in [(.38, 52, GROCERY, 15), (.14, 24, DINING, 13), (.07, 28, DELIVERY, 19),
                                          (.18, 65, SHOPPING, 15), (.05, 20, HEALTH, 13), (.08, 30, LEISURE, 17),
                                          (.10, 20, MISC, 14)]:
                tgt = share * disc
                cnt = T.count(tgt, avg)
                if cnt:
                    for amt in T.split(tgt, cnt, .55):
                        T.card(cid, y, m, amt, T.pick(items), "Other", mu=mu, sd=4)
            for ev in travel_events:
                if ev == k:
                    amt = float(np.clip(np.exp(g.normal(np.log(.06 * l["income"]), .4)), 80, 4000))
                    T.card(cid, y, m, amt, T.pick(TRAVEL), "Other", mu=12, sd=4, suffix=False)
            if g.random() < .04:   # occasional dentist / one-off health
                T.card(cid, y, m, float(g.uniform(60, 250)), ("DENTAL PRACTICE", 8021), "Other", mu=11, sd=2, suffix=False)

    df = pd.DataFrame(T.rows, columns=["customer_id", "dt", "amount", "merchant_name", "mcc", "transaction_type", "category_ground_truth"])
    df = df.sort_values(["dt", "customer_id"], kind="mergesort").reset_index(drop=True)
    df["transaction_id"] = ["TX{:07d}".format(i) for i in range(1, len(df) + 1)]
    df["timestamp"] = df["dt"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return df[TX_COLS]


# ----------------------------------------------------------------------------------------------
# STEP 3: product labels
# ----------------------------------------------------------------------------------------------
PRODUCTS = [("GP01", "EV Finance"), ("GP02", "Green Home Loan"), ("GP03", "Solar Finance"), ("GP04", "Paperless Banking")]
TARGET_TAKEUP = {"GP01": .14, "GP02": .10, "GP03": .07, "GP04": .35}


def eligibility(r):
    """Documented product-catalogue eligibility rules (business rules, NOT ML). Returns {pid: reason or None}."""
    owner = r["home_ownership_status"] != "RENTER"
    inc = r["income_annual_estimated"]
    out = {}
    # GP01 EV Finance
    if r["current_vehicle_type"] == "BEV":
        out["GP01"] = "ALREADY_BEV"
    elif r["current_vehicle_type"] == "NONE":
        out["GP01"] = "NO_VEHICLE"
    elif inc < 15000:
        out["GP01"] = "BELOW_INCOME_THRESHOLD"
    else:
        out["GP01"] = None
    # GP02 Green Home Loan
    if not owner:
        out["GP02"] = "NOT_HOME_OWNER"
    elif r["epc_rating"] is None:
        out["GP02"] = "NO_DATA"
    elif r["epc_rating"] in ("A", "B"):
        out["GP02"] = "NO_USE_CASE"
    elif inc < 20000:
        out["GP02"] = "BELOW_INCOME_THRESHOLD"
    else:
        out["GP02"] = None
    # GP03 Solar Finance
    if not owner:
        out["GP03"] = "NOT_HOME_OWNER"
    elif r["property_type"] is None:
        out["GP03"] = "NO_DATA"
    elif r["property_type"] == "FLAT_APARTMENT":
        out["GP03"] = "UNSUITABLE_PROPERTY"
    elif r["solar_installed"] is True:
        out["GP03"] = "SOLAR_ALREADY_INSTALLED"
    elif r["solar_installed"] is None:
        out["GP03"] = "NO_DATA"
    elif inc < 25000:
        out["GP03"] = "BELOW_INCOME_THRESHOLD"
    else:
        out["GP03"] = None
    # GP04 Paperless
    out["GP04"] = None if r["paper_statement_opt_in"] else "ALREADY_DIGITAL"
    return out


def raw_score(pid, r):
    """Feature-based latent score (before noise / intercept). Never uses eligibility."""
    inc = max(r["income_annual_estimated"], 5000)
    green = r["sustainability_preference_flag"] is True
    if pid == "GP01":
        fuel, miles = r["avg_monthly_fuel_spend"], r["estimated_annual_mileage"] or 0
        va = r["vehicle_age_years"]
        loan_soon = r["has_auto_loan"] and r["auto_loan_months_remaining"] is not None and r["auto_loan_months_remaining"] <= 12
        z = (1.1 * min(fuel / 100, 3) + .7 * min(miles / 8000, 3) + .5 * (va is not None and va >= 8) +
             .9 * loan_soon + .3 * r["has_auto_loan"] + .6 * (r["has_off_street_parking"] is True) + .5 * green +
             .4 * math.log(inc / 30000) - .02 * (r["age"] - 40) + (.3 if r["current_vehicle_type"] == "HYBRID" else 0))
        z -= 6 if r["current_vehicle_type"] == "BEV" else 0
        z -= 4 if r["current_vehicle_type"] == "NONE" else 0
        return z
    if pid == "GP02":
        epc_pts = {"G": .9, "F": .9, "E": .7, "D": .3, "C": 0, "B": -.5, "A": -1.0, None: 0}
        z = (epc_pts[r["epc_rating"]] + .7 * min(r["avg_monthly_gas_spend"] / 100, 3) +
             .5 * min(r["avg_monthly_home_improvement_spend"] / 60, 3) + .4 * r["has_mortgage"] + .3 * green +
             (.6 if r["heating_type"] == "OIL" else .2 if r["heating_type"] == "GAS_BOILER" else 0) +
             .3 * math.log(inc / 35000) - .015 * (r["age"] - 45))
        z -= 6 if r["home_ownership_status"] == "RENTER" else 0
        return z
    if pid == "GP03":
        pt = r["property_type"]
        z = (1.0 * min(r["avg_monthly_electricity_spend"] / 100, 3) + (.5 if pt == "DETACHED" else .2 if pt == "SEMI_DETACHED" else 0) +
             .6 * (r["home_ownership_status"] == "OWNER_OUTRIGHT") +
             .5 * (r["current_vehicle_type"] == "BEV" and r["has_off_street_parking"] is True) + .5 * green +
             .5 * math.log(inc / 40000) - .01 * (r["age"] - 50))
        z -= 6 if r["home_ownership_status"] == "RENTER" else 0
        z -= 4 if pt == "FLAT_APARTMENT" else 0
        z -= 6 if r["solar_installed"] is True else 0
        return z
    z = (1.0 * min(r["mobile_app_logins_monthly"] / 15, 3) + .4 * min(r["web_portal_logins_monthly"] / 6, 3) + .4 * (r["age"] < 45))
    z -= 6 if not r["paper_statement_opt_in"] else 0
    return z


def reason_codes(pid, r):
    codes = []
    g = r["sustainability_preference_flag"] is True
    if pid == "GP01":
        if r["avg_monthly_fuel_spend"] >= 100: codes.append("HIGH_FUEL_SPEND")
        if (r["estimated_annual_mileage"] or 0) >= 9000: codes.append("HIGH_ANNUAL_MILEAGE")
        t = r["current_vehicle_type"]
        codes.append({"PETROL": "PETROL_VEHICLE", "DIESEL": "DIESEL_VEHICLE", "HYBRID": "HYBRID_VEHICLE"}[t])
        if (r["vehicle_age_years"] or 0) >= 8: codes.append("OLDER_VEHICLE")
        if r["has_auto_loan"]:
            codes.append("AUTO_LOAN_ENDING_SOON" if (r["auto_loan_months_remaining"] or 99) <= 12 else "EXISTING_AUTO_LOAN")
        if r["has_off_street_parking"] is True: codes.append("HOME_CHARGING_AVAILABLE")
        if r["avg_monthly_automotive_spend"] >= 60: codes.append("HIGH_AUTOMOTIVE_SPEND")
    elif pid == "GP02":
        codes.append("OWNER_OCCUPIER")
        if r["has_mortgage"]: codes.append("MORTGAGE_HOLDER")
        if r["epc_rating"] in ("E", "F", "G"): codes.append("LOW_EPC_RATING")
        elif r["epc_rating"] == "D": codes.append("MID_EPC_RATING")
        if r["avg_monthly_gas_spend"] >= 100: codes.append("HIGH_GAS_SPEND")
        if r["avg_monthly_home_improvement_spend"] >= 80: codes.append("HIGH_HOME_IMPROVEMENT_SPEND")
        if r["heating_type"] == "OIL": codes.append("OIL_HEATING")
        elif r["heating_type"] == "GAS_BOILER": codes.append("GAS_BOILER_HEATING")
    elif pid == "GP03":
        if r["avg_monthly_electricity_spend"] >= 110: codes.append("HIGH_ELECTRICITY_SPEND")
        if r["property_type"] in ("DETACHED", "SEMI_DETACHED"): codes.append("SUITABLE_ROOF_PROPERTY")
        if r["home_ownership_status"] == "OWNER_OUTRIGHT": codes.append("OWNER_OUTRIGHT")
        if r["income_annual_estimated"] >= 50000: codes.append("HIGH_INCOME")
        if r["current_vehicle_type"] == "BEV" and r["has_off_street_parking"] is True: codes.append("EV_OWNER_HOME_CHARGING")
    else:
        codes.append("PAPER_STATEMENTS")
        if r["mobile_app_logins_monthly"] >= 20: codes.append("HIGH_APP_ACTIVITY")
        if r["web_portal_logins_monthly"] >= 10: codes.append("HIGH_WEB_ACTIVITY")
        if r["mobile_app_logins_monthly"] + r["web_portal_logins_monthly"] >= 25: codes.append("HIGH_DIGITAL_ACTIVITY")
    if g and pid != "GP04": codes.append("GREEN_INTEREST")
    if pid != "GP04" and len(codes) == 0: codes = ["NO_STRONG_SIGNAL"]
    return "|".join(codes)


def build_labels(C, rng):
    recs = C.replace({np.nan: None}).to_dict("records")
    for r in recs:  # normalise missing -> None
        for k, v in list(r.items()):
            if v is pd.NA or (isinstance(v, float) and math.isnan(v)):
                r[k] = None
    n = len(recs)
    out = []
    for pid, pname in tqdm(PRODUCTS, desc="labels", unit="product"):
        elig = np.array([eligibility(r)[pid] is None for r in recs])
        z = np.array([raw_score(pid, r) for r in recs]) + rng.normal(0, .5, n)
        lo, hi = -20.0, 20.0                      # calibrate intercept: mean propensity among eligible = target
        for _ in range(80):
            mid = (lo + hi) / 2
            if sigmoid(z[elig] + mid).mean() < TARGET_TAKEUP[pid]:
                lo = mid
            else:
                hi = mid
        p = sigmoid(z + (lo + hi) / 2)
        take = np.where(elig, (rng.random(n) < p).astype(int), 0)
        for i, r in enumerate(recs):
            reason = eligibility(r)[pid]
            out.append({
                "customer_id": r["customer_id"], "product_id": pid, "product_name": pname,
                "eligible_for_recommendation": bool(elig[i]), "ineligibility_reason": reason,
                "propensity_label": round(float(p[i]), 4), "simulated_takeup": int(take[i]),
                "reason_codes": reason_codes(pid, r) if elig[i] else f"NOT_ELIGIBLE|{reason}",
                "label_version": "v1.0", "data_as_of": AS_OF.isoformat(),
            })
    df = pd.DataFrame(out)
    df["_o"] = df["customer_id"] + df["product_id"]
    return df.sort_values("_o").drop(columns="_o").reset_index(drop=True)


# ----------------------------------------------------------------------------------------------
# Validation (reads the CSVs back from disk, checks against schema JSON + cross-table rules)
# ----------------------------------------------------------------------------------------------
def _check_table(path, schema, errors):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    cols = list(schema["columns"].keys())
    if list(df.columns) != cols:
        errors.append(f"{path}: column order/names differ from schema")
        return df
    for name, spec in schema["columns"].items():
        types = spec["type"] if isinstance(spec["type"], list) else [spec["type"]]
        nullable = "null" in types
        base = [t for t in types if t != "null"][0]
        s = df[name]
        empty = s == ""
        if empty.any() and not nullable:
            errors.append(f"{path}.{name}: {int(empty.sum())} empty values but not nullable")
        v = s[~empty]
        if base == "integer":
            bad = ~v.str.fullmatch(r"-?\d+")
        elif base == "number":
            bad = ~v.str.fullmatch(r"-?\d+(\.\d+)?")
        elif base == "boolean":
            bad = ~v.isin(["True", "False"])
        else:
            bad = pd.Series(False, index=v.index)
        if bad.any():
            errors.append(f"{path}.{name}: {int(bad.sum())} values not {base}, e.g. {v[bad].iloc[0]!r}")
            continue
        if "pattern" in spec:
            bad = ~v.str.fullmatch(spec["pattern"])
            if bad.any(): errors.append(f"{path}.{name}: {int(bad.sum())} pattern violations e.g. {v[bad].iloc[0]!r}")
        if "enum" in spec:
            allowed = {str(e) for e in spec["enum"] if e is not None}
            bad = ~v.isin(allowed)
            if bad.any(): errors.append(f"{path}.{name}: {int(bad.sum())} enum violations e.g. {v[bad].iloc[0]!r}")
        if base in ("integer", "number"):
            x = v.astype(float)
            if "minimum" in spec and (x < spec["minimum"]).any(): errors.append(f"{path}.{name}: below minimum")
            if "maximum" in spec and (x > spec["maximum"]).any(): errors.append(f"{path}.{name}: above maximum")
        if "const" in spec and (v != str(spec["const"])).any():
            errors.append(f"{path}.{name}: const violated")
        if spec.get("format") == "date":
            try:
                [date.fromisoformat(z) for z in v.unique()]
            except Exception:
                errors.append(f"{path}.{name}: bad date")
        if spec.get("format") == "date-time":
            try:
                [datetime.fromisoformat(z) for z in v.unique()]
            except Exception:
                errors.append(f"{path}.{name}: bad date-time")
    if "primary_key" in schema:
        pk = schema["primary_key"]
        pk = [pk] if isinstance(pk, str) else pk
        if df.duplicated(pk).any():
            errors.append(f"{path}: duplicate primary key")
    return df


def validate(out_dir, schema_dir):
    errors = []
    sch = {n: json.load(open(os.path.join(schema_dir, f"{n}.schema.json"))) for n in ["customers", "transactions", "product_labels"]}
    cd = _check_table(os.path.join(out_dir, "customers.csv"), sch["customers"], errors)
    td = _check_table(os.path.join(out_dir, "transactions.csv"), sch["transactions"], errors)
    ld = _check_table(os.path.join(out_dir, "product_labels.csv"), sch["product_labels"], errors)
    if errors:
        return errors

    B = lambda s: s.map({"True": True, "False": False})
    N = lambda s: pd.to_numeric(s.replace("", np.nan))
    c = cd.copy()
    for col in ["paper_statement_opt_in", "has_mortgage", "has_auto_loan", "has_investment_account"]:
        c[col] = B(c[col])
    c["solar"] = c["solar_installed"].map({"True": True, "False": False, "": None})
    for col in ["age", "income_annual_estimated", "tenure_months", "estimated_annual_mileage", "auto_loan_months_remaining",
                "investment_account_balance", "mortgage_balance_remaining", "auto_loan_balance_remaining", "vehicle_age_years",
                "statements_per_year", "avg_monthly_gas_spend", "avg_monthly_fuel_spend"] + list(SPEND_CATS.values()):
        c[col] = N(c[col])
    assert_ = lambda cond, msg: errors.append(msg) if not bool(cond) else None

    # ---- customers cross-column rules
    assert_((c.has_mortgage == (c.home_ownership_status == "OWNER_MORTGAGED")).all(), "has_mortgage vs home_ownership_status")
    assert_((c.has_mortgage == c.mortgage_balance_remaining.notna()).all(), "has_mortgage vs mortgage balance")
    assert_((c.has_auto_loan == c.auto_loan_balance_remaining.notna()).all(), "has_auto_loan vs balance")
    assert_((c.has_auto_loan == c.auto_loan_months_remaining.notna()).all(), "has_auto_loan vs months remaining")
    assert_((c.has_investment_account == c.investment_account_balance.notna()).all(), "has_investment_account vs balance")
    assert_(~((c.home_ownership_status == "RENTER") & (c.solar == True)).any(), "renter with solar installed")
    none = c.current_vehicle_type == "NONE"
    assert_((c.loc[none, "estimated_annual_mileage"] == 0).all(), "NONE vehicle with mileage")
    assert_(c.loc[none, "vehicle_age_years"].isna().all(), "NONE vehicle with vehicle age")
    assert_(~(none & c.has_auto_loan).any(), "NONE vehicle with auto loan")
    assert_((c.loc[none, "avg_monthly_fuel_spend"] == 0).all(), "NONE vehicle with fuel spend")
    assert_((c.loc[~none, "estimated_annual_mileage"] > 0).all(), "vehicle owner with zero mileage")
    assert_((c.loc[c.paper_statement_opt_in, "statements_per_year"] >= 1).all(), "paper customer without statements")
    assert_((c.loc[~c.paper_statement_opt_in, "statements_per_year"] == 0).all(), "digital customer with statements")
    assert_(((c.heating_type == "GAS_BOILER") == (c.avg_monthly_gas_spend > 0)).all(), "gas spend vs heating_type")
    assert_(c.customer_id.is_unique, "customer ids not unique")

    # ---- transactions vs customers
    t = td.copy()
    t["amount"] = t["amount"].astype(float)
    assert_(t.customer_id.isin(c.customer_id).all(), "transaction customer not in customers")
    assert_(c.customer_id.isin(t.customer_id).all(), "customer without transactions")
    assert_((t.amount != 0).all(), "zero amount")
    inc = t.category_ground_truth == "Income"
    assert_((t.loc[inc, "amount"] > 0).all(), "Income with non-positive amount")
    assert_((t.loc[~inc, "amount"] < 0).all(), "spend with non-negative amount")
    assert_((t.loc[inc, "transaction_type"] == "CREDIT").all(), "Income not CREDIT")
    assert_((t.loc[t.transaction_type == "CREDIT", "category_ground_truth"] == "Income").all(), "CREDIT not Income")
    assert_((t.loc[t.transaction_type != "CARD_PAYMENT", "mcc"] == "").all(), "non-card txn with MCC")
    assert_((t.loc[t.transaction_type == "CARD_PAYMENT", "mcc"] != "").all(), "card txn without MCC")
    assert_((t.merchant_name.str.strip() != "").all(), "blank merchant")
    ts = pd.to_datetime(t.timestamp)
    assert_(ts.min() >= pd.Timestamp("2025-10-01") and ts.max() < pd.Timestamp("2026-10-01"), "timestamp outside window")
    assert_(t.transaction_id.is_unique, "tx ids not unique")
    # reconcile avg_monthly_* with transactions
    active = np.minimum(12, c.set_index("customer_id").tenure_months)
    sums = (-t[t.category_ground_truth.isin(SPEND_CATS)].groupby(["customer_id", "category_ground_truth"]).amount.sum()).unstack().reindex(c.customer_id).fillna(0.0)
    for cat, col in SPEND_CATS.items():
        base = sums[cat] if cat in sums else pd.Series(0.0, index=sums.index)
        e = (base / active.reindex(sums.index)).round(2).values
        diff = float(np.abs(e - c[col].values).max())
        assert_(diff <= 0.011, f"{col} does not reconcile with transactions (max diff {diff})")
    # sanity on cash-flow
    flow = t.groupby("customer_id").amount.agg(lambda s: (s[s < 0].sum() / s[s > 0].sum()) if (s > 0).any() else np.inf)
    assert_(flow.median() < 1.0, "median outflow/inflow ratio >= 1")

    # ---- labels
    l = ld.copy()
    l["elig"] = B(l["eligible_for_recommendation"])
    l["take"] = l["simulated_takeup"].astype(int)
    l["prop"] = l["propensity_label"].astype(float)
    assert_(l.customer_id.isin(c.customer_id).all() and c.customer_id.isin(l.customer_id).all(), "label/customer id mismatch")
    assert_((l.groupby("customer_id").size() == 4).all(), "customer without exactly 4 label rows")
    assert_(l.groupby("customer_id").product_id.nunique().eq(4).all(), "duplicate products per customer")
    assert_(l.product_id.map(dict(PRODUCTS)).eq(l.product_name).all(), "product id/name mismatch")
    assert_((l.loc[~l.elig, "take"] == 0).all(), "takeup on ineligible row")
    assert_(((l.ineligibility_reason == "") == l.elig).all(), "ineligibility_reason vs eligible flag")
    assert_(l.loc[l.elig, "reason_codes"].ne("").all() and l.loc[l.elig, "reason_codes"].str.startswith("NOT_ELIGIBLE").eq(False).all(), "eligible row reason codes")
    j = l.merge(c, on="customer_id")
    assert_(not ((j.product_id == "GP01") & (j.current_vehicle_type == "BEV") & j.elig).any(), "BEV owner eligible for GP01")
    assert_(not ((j.product_id == "GP01") & (j.current_vehicle_type == "NONE") & j.elig).any(), "no-vehicle customer eligible for GP01")
    assert_(not ((j.product_id == "GP03") & (j.solar == True) & j.elig).any(), "solar owner eligible for GP03")
    assert_(not ((j.product_id.isin(["GP02", "GP03"])) & (j.home_ownership_status == "RENTER") & j.elig).any(), "renter eligible for GP02/GP03")
    assert_(not ((j.product_id == "GP04") & (~j.paper_statement_opt_in) & j.elig).any(), "digital customer eligible for GP04")
    assert_(not ((j.product_id == "GP03") & (j.property_type == "FLAT_APARTMENT") & j.elig).any(), "flat eligible for GP03")
    # recompute eligibility from the CSV values and compare with the stored reasons
    exp = {}
    nz = lambda v: None if v == "" else v
    for r in cd.to_dict("records"):
        rr = {"home_ownership_status": r["home_ownership_status"], "income_annual_estimated": float(r["income_annual_estimated"]),
              "current_vehicle_type": r["current_vehicle_type"], "epc_rating": nz(r["epc_rating"]),
              "property_type": nz(r["property_type"]), "paper_statement_opt_in": r["paper_statement_opt_in"] == "True",
              "solar_installed": {"True": True, "False": False, "": None}[r["solar_installed"]]}
        for pid, reason in eligibility(rr).items():
            exp[(r["customer_id"], pid)] = reason or ""
    got = {(a, b): d for a, b, d in zip(l.customer_id, l.product_id, l.ineligibility_reason)}
    assert_(exp == got, "ineligibility reasons differ from rules recomputed on CSV values")
    assert_(set(c.split.unique()) <= {"train", "validation", "test"}, "bad split")
    return errors


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="./data")
    ap.add_argument("--schema-dir", default="./schemas")
    ap.add_argument("--validate-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if not a.validate_only:
        rng = np.random.default_rng(a.seed)
        print("1/3 customers ...")
        C, L = build_customers(a.n, rng)
        print("2/3 transactions ...")
        T = gen_transactions(C, L, a.seed + 1)
        # recompute avg_monthly_* from the ledger so both tables reconcile exactly
        active = pd.Series(L["active"].values, index=C["customer_id"].values)
        s = (-T[T.category_ground_truth.isin(SPEND_CATS)].groupby(["customer_id", "category_ground_truth"]).amount.sum()).unstack().reindex(C["customer_id"]).fillna(0.0)
        for cat, col in SPEND_CATS.items():
            C[col] = (s[cat] / active.reindex(s.index)).round(2).values if cat in s else 0.0
        print("3/3 product labels ...")
        P = build_labels(C, np.random.default_rng(a.seed + 2))
        C.to_csv(os.path.join(a.out, "customers.csv"), index=False, na_rep="")
        T.to_csv(os.path.join(a.out, "transactions.csv"), index=False, na_rep="")
        P.to_csv(os.path.join(a.out, "product_labels.csv"), index=False, na_rep="")
        print(f"wrote {len(C)} customers, {len(T)} transactions, {len(P)} label rows -> {a.out}")

    print("validating ...")
    errs = validate(a.out, a.schema_dir)
    if errs:
        print("VALIDATION FAILED:")
        for e in errs:
            print(" -", e)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
