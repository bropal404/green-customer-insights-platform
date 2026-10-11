# Data card: synthetic bank for the Green Customer Insights Platform

30,000 synthetic customers, January 2025 to September 2026, seed 20261010. No real personal or banking data is used (brief slide 06): customers, accounts and transactions are generated; public UK statistics only calibrate them.

## Source systems

The bank's data arrives as extracts from seven systems, each with its own format, keys, units, dates, codes and quality problems (configs/source_systems.yaml). FR1 ingestion (src/gcip/fr1/ingest.py) checks each file against the control totals, harmonises it, resolves keys through crosswalks, removes duplicates and quarantines records that match no customer.

| system | files | format | keys | dates and units | known issues |
| --- | --- | --- | --- | --- | --- |
| core_banking | customers.csv, accounts.csv, balances.csv | CSV | CUST_ID; accounts by ACCT_NO (sort code and account number, e.g. 001004-12345678) | YYYYMMDD; money in pence, interest rates in basis points | Region names typed inconsistently (upper and lower case variants). |
| card_processor | card_registry.csv, part-*.parquet | Parquet (transactions), CSV (registry) | CARD_TOKEN (the processor never sees account numbers); TXN_REF | ISO 8601 timestamps; amounts in pence (AMOUNT_MINOR), direction C/D | The feed occasionally replays a batch: about 0.2% of rows are exact duplicates. |
| payments_hub | part-*.csv.gz | gzip CSV | PAYMENT_ID; ACCT_NO | DD/MM/YYYY value date; signed amounts in pounds (negative = money out) | About 1% of rows carry the account number in the legacy 14-digit form without a hyphen. Narratives are space-padded to 35 characters. |
| crm_marketing | contacts.csv, campaign_responses.csv | CSV | CONTACT_ID; CUST_REF (blank for prospects) | DD/MM/YYYY; product names instead of product codes | About 1% of customers have two contact ids (one from a web sign-up). About 1% of campaign contacts are prospects who are not customers. Outcomes are recorded in several spellings (e.g. 'Clicked', 'CLICKED', 'click-through'). |
| digital_analytics | monthly_engagement.jsonl | JSON Lines | uid = first 16 hex characters of SHA-256('<salt>:<customer id>'), pseudonymised by the analytics team | YYYY-MM; counts | About 0.3% of records belong to prospects who never became customers (no matching uid). |
| lending_assets | property_valuations.xml, vehicle_records.csv | XML (properties), CSV (vehicles) | mortgage or motor loan ACCT_NO | year of first registration; values in pence | EPC details are missing when the property did not match the EPC register. |
| product_system | holdings.csv, statement_preference_events.csv | CSV | ACCT_NO; CUSTOMER_REF | ISO dates and timestamps; no amounts | - |

### Ingestion result

- 7 source systems, 71 files; 71 passed size and SHA-256 checks; row counts all match.
- Transactions: 20,691,226 card-feed rows (41,301 replayed duplicates removed) and 7,616,570 payments-hub rows (75,989 legacy account numbers normalised) -> 28,266,495 transactions in 30 parts.
- Quarantined from campaign_responses: 451 (prospect: no customer reference).
- Quarantined from digital_activity: 1,890 (uid matches no customer).
- Audit: 8 of 8 tables identical to the bank's system of record.
- Audit: 28,266,495 of 28,266,495 transactions recovered; 0 missing, 0 unexpected, 0 on the wrong account, 0 on the wrong customer, 0 with any field different.

| file | rows in | rows out | handled |
| --- | --- | --- | --- |
| core_banking/customers.csv | 30,000 | 30,000 | region names re-cased to the ONS spelling: 2,379 |
| core_banking/accounts.csv | 80,269 | 80,269 | - |
| core_banking/balances.csv | 1,617,888 | 1,617,888 | - |
| product_system/holdings.csv | 81,118 | 81,118 | - |
| product_system/statement_preference_events.csv | 849 | 849 | - |
| lending_assets/property_valuations.xml | 3,296 | 3,296 | properties without an EPC register match: 1,264 |
| lending_assets/vehicle_records.csv | 2,232 | 2,232 | - |
| crm_marketing/contacts.csv | 30,742 | 30,291 | customers with more than one contact id (merged): 291; prospect contacts (no customer reference): 451 |
| crm_marketing/campaign_responses.csv | 45,590 | 45,139 | responses from prospects quarantined: 451; responses resolved through a second contact id: 229; outcome spellings harmonised: 21,302 |
| digital_analytics/monthly_engagement.jsonl | 631,890 | 630,000 | records with no matching customer (prospects) quarantined: 1,890 |
| card_processor/card_registry.csv | 46,717 | 46,717 | - |

### Ingestion audit

Each harmonised table is fingerprinted (every cell in a canonical form, order-independent) and compared with the fingerprint of the bank's system of record kept in the truth vault; every transaction's record hash is compared one by one.

| table | rows expected | rows ingested | identical | columns differing |
| --- | --- | --- | --- | --- |
| customers | 30,000 | 30,000 | yes | - |
| accounts | 80,269 | 80,269 | yes | - |
| balances | 1,617,888 | 1,617,888 | yes | - |
| product_holdings | 81,118 | 81,118 | yes | - |
| preference_changes | 849 | 849 | yes | - |
| assets | 5,528 | 5,528 | yes | - |
| digital_activity | 630,000 | 630,000 | yes | - |
| campaign_responses | 45,139 | 45,139 | yes | - |

### Column lineage

Where every warehouse column comes from and how it was transformed.

| table | column | source | source_field | transformation |
| --- | --- | --- | --- | --- |
| customers | customer_id | core_banking/customers.csv | CUST_ID | as delivered; duplicates quarantined |
| customers | birth_year | core_banking/customers.csv | BIRTH_YR | integer |
| customers | la_code, la_name | core_banking/customers.csv | LA_CD, LA_NM | as delivered |
| customers | region | core_banking/customers.csv | REGION_NM | trimmed and re-cased to the ONS region name |
| customers | declared_residential_status | core_banking/customers.csv | RES_STATUS | code table residential_status |
| customers | declared_employment_status | core_banking/customers.csv | EMP_STATUS | code table employment_status |
| customers | declared_income_band | core_banking/customers.csv | INC_BAND | code table income_band |
| customers | customer_since | core_banking/customers.csv | CUST_SINCE | YYYYMMDD to date |
| customers | marketing_consent, vulnerability_flag | core_banking/customers.csv | MKT_CONSENT, VULN_FLAG | Y/N to boolean |
| customers | preferred_contact_channel | core_banking/customers.csv | PREF_CHANNEL | code table contact_channel |
| customers | stated_sustainability_preference | core_banking/customers.csv | SUST_PREF | Y/N/blank to 1/0/null |
| customers | declared_occupation | core_banking/customers.csv | OCCUPATION_CD | code table occupation (SOC 2020 major group or status) |
| customers | declared_dependants, declared_years_at_address | core_banking/customers.csv | DEPENDANTS, YRS_AT_ADDR | number |
| accounts | account_id | core_banking/accounts.csv | CUST_ID, ACCT_TYPE | harmonised key '<customer>-<account type>' |
| accounts | customer_id | core_banking/accounts.csv | CUST_ID | must exist in customers, else quarantined |
| accounts | account_type | core_banking/accounts.csv | ACCT_FAMILY | lower case |
| accounts | product_code | core_banking/accounts.csv | PRODUCT_CD | as delivered |
| accounts | opened_date, closed_date | core_banking/accounts.csv | OPEN_DT, CLOSE_DT | YYYYMMDD to date |
| accounts | status | core_banking/accounts.csv | CLOSE_DT | closed if the close date is on or before the extract date |
| accounts | statement_preference | core_banking/accounts.csv | STMT_PREF | code table statement_preference |
| accounts | credit_limit, original_amount, collateral_value_at_origination | core_banking/accounts.csv | CREDIT_LIMIT_PENCE, ORIG_AMT_PENCE, COLLATERAL_VALUE_PENCE | pence to GBP |
| accounts | interest_rate | core_banking/accounts.csv | RATE_BPS | basis points to a fraction |
| accounts | term_months, loan_purpose, provider | core_banking/accounts.csv | TERM_MTHS, LOAN_PURPOSE, PROVIDER | number; lower case; as delivered |
| balances | account_id | core_banking/balances.csv | ACCT_NO | account-number crosswalk from accounts.csv |
| balances | month_end | core_banking/balances.csv | MONTH_END | YYYYMMDD to date |
| balances | balance | core_banking/balances.csv | BAL_PENCE | pence to GBP |
| balances | arrears_months | core_banking/balances.csv | ARREARS_MTHS | integer |
| transactions | transaction_id | card_processor/part-*.parquet / payments_hub/part-*.csv.gz | TXN_REF / PAYMENT_ID | replayed duplicates removed |
| transactions | account_id | card_processor/part-*.parquet / payments_hub/part-*.csv.gz | CARD_TOKEN / ACCT_NO | card token -> card_registry.csv -> account number -> crosswalk / account number normalised (legacy 14-digit form) -> crosswalk |
| transactions | booking_datetime | card_processor / payments_hub | AUTH_TS / VALUE_DATE | ISO timestamp / DD/MM/YYYY to datetime |
| transactions | amount, credit_debit_indicator | card_processor / payments_hub | AMOUNT_MINOR, DIRECTION / AMOUNT | pence to GBP, C/D to Credit/Debit / absolute value of signed pounds, sign to Credit/Debit |
| transactions | currency | card_processor / payments_hub | CURRENCY | as delivered |
| transactions | transaction_information | card_processor / payments_hub | MERCHANT_DESCRIPTOR + CARD_LAST4 / NARRATIVE | statement form '<descriptor> CD <last 4 digits>' / right-trimmed |
| transactions | merchant_name, merchant_category_code | card_processor | MERCHANT_NAME, MCC | card payments only; null for payments |
| transactions | proprietary_bank_transaction_code | card_processor / payments_hub | ENTRY_TYPE / SCHEME | code tables card_entry_type / payment_scheme |
| transactions | bank_transaction_code | derived | proprietary_bank_transaction_code | code table open_banking_code |
| transactions | status | derived | - | 'Booked' (both feeds carry settled items only) |
| product_holdings | customer_id, account_id | product_system/holdings.csv | ACCT_NO, CUSTOMER_REF | account-number crosswalk; owner checked against CUSTOMER_REF |
| product_holdings | product_code, product_family | product_system/holdings.csv | PRODUCT_CD, PRODUCT_FAMILY | as delivered |
| product_holdings | green_product | product_system/holdings.csv | PRODUCT_CD | product codes in configs/products.yaml |
| product_holdings | start_date, end_date | product_system/holdings.csv | START_DATE, END_DATE | ISO date |
| preference_changes | account_id | product_system/statement_preference_events.csv | ACCT_NO | account-number crosswalk |
| preference_changes | change_date | product_system/statement_preference_events.csv | EVENT_TS | timestamp to date |
| preference_changes | old_preference, new_preference | product_system/statement_preference_events.csv | OLD_PREF, NEW_PREF | lower case |
| assets | asset_id, customer_id | lending_assets/property_valuations.xml / vehicle_records.csv | mortgage_account / MOTOR_ACCT_NO | loan account crosswalk to the owner; asset id '<customer>-PROP' or '<customer>-VEH' |
| assets | property_type, floor_area_m2 | lending_assets/property_valuations.xml | property/@type, property/@floor_area_m2 | lower case; number |
| assets | property_detail, construction_period, bedrooms | lending_assets/property_valuations.xml | property/@detail, @built, @bedrooms | as delivered; number (from the valuation report) |
| assets | epc_band, epc_main_heating, solar_pv_on_record | lending_assets/property_valuations.xml | epc/@rating, @main_fuel, @solar_pv | as delivered; code table epc_main_fuel; Y/N to boolean (null when no EPC match) |
| assets | vehicle_fuel_type, vehicle_first_registered_year, vehicle_registration | lending_assets/vehicle_records.csv | FUEL_TYPE, YEAR_OF_FIRST_REG, VRM | code table dvla_fuel; number; as delivered |
| assets | valuation_at_origination | lending_assets/property_valuations.xml / vehicle_records.csv | value/@pence / VALUATION_PENCE | pence to GBP |
| digital_activity | customer_id | digital_analytics/monthly_engagement.jsonl | uid | re-derive each customer's uid with the shared salt; unmatched uids quarantined |
| digital_activity | month | digital_analytics/monthly_engagement.jsonl | month | YYYY-MM to first day of month |
| digital_activity | app_logins, web_logins, branch_visits, estatement_views | digital_analytics/monthly_engagement.jsonl | app_sessions, web_sessions, branch_visits, estatement_views | integer |
| digital_activity | paperless_all_accounts | digital_analytics/monthly_engagement.jsonl | paperless | boolean |
| campaign_responses | customer_id | crm_marketing/campaign_responses.csv | CONTACT_ID | contacts.csv crosswalk (several contact ids per customer allowed); prospects quarantined |
| campaign_responses | campaign_id, channel | crm_marketing/campaign_responses.csv | CAMPAIGN, CHANNEL | as delivered; lower case |
| campaign_responses | product | crm_marketing/campaign_responses.csv | PRODUCT_NAME | product names in configs/products.yaml to codes |
| campaign_responses | response | crm_marketing/campaign_responses.csv | OUTCOME | spellings in code table campaign_outcome |
| campaign_responses | sent_date, response_date | crm_marketing/campaign_responses.csv | SENT, OUTCOME_DATE | DD/MM/YYYY to date |

## Warehouse tables

The harmonised tables FR1 ingestion writes. They stay separate, as a bank keeps them; FR1's Customer 360 then joins them into one point-in-time row per customer.

### customers (30,000 rows)

One row per customer: what the bank recorded at onboarding.

| column | meaning |
| --- | --- |
| customer_id | Synthetic customer key. |
| birth_year | Year of birth. |
| la_code / la_name / region | Local authority (April 2023 boundaries) and region of the home address. |
| declared_residential_status | Residential status given at onboarding; about 10% is out of date. |
| declared_employment_status | Employment status given at onboarding. |
| declared_income_band | Banded gross annual income; about 12% is one band out. |
| customer_since | Date the relationship began. |
| marketing_consent | Whether the customer accepts marketing. |
| vulnerability_flag | Bank-recorded vulnerability (drives human review and blocks). |
| preferred_contact_channel | app, email or post. |
| stated_sustainability_preference | Investment fact-find answer (1 yes, 0 no); blank for most customers. |
| declared_occupation | Occupation group given at onboarding (SOC 2020 major group, or retired, student, not in work); about 8% out of date. |
| declared_dependants | Number of dependent children given in credit applications. |
| declared_years_at_address | Whole years at the current address. |

### accounts (80,269 rows)

One row per account the customer holds with the bank.

| column | meaning |
| --- | --- |
| account_id | Customer id plus a suffix: CA current, SV savings, CC card, MG mortgage, ML motor loan, PL personal loan, IV investment; green products EV, GH, SF, SI, GL (N = opened in the label window). |
| account_type / product_code | Product family and product code (green codes: MOTOR-EV-GREEN, MTG-GREEN-FA, LOAN-SOLAR, INV-SUSTAIN-FOCUS, PL-GREEN). |
| opened_date / closed_date / status | Account lifecycle. |
| statement_preference | paper or online at extraction (history in preference_changes). |
| credit_limit, original_amount, interest_rate, term_months | Credit terms where relevant. Rates follow the Bank of England quoted rate for the product when it started (mortgages by fixed period and LTV, personal loans by size, cards, savings). |
| collateral_value_at_origination | Property or vehicle value when the loan began (PCAF attribution denominator). |
| loan_purpose | Stated purpose of personal loans. |

### balances (1,617,888 rows)

Month-end balance per account, January 2025 to September 2026.

| column | meaning |
| --- | --- |
| account_id, month_end | Key. |
| balance | GBP; liabilities are negative. |
| arrears_months | Months in arrears on credit products. |

### transactions (28,266,495 rows)

Every booked transaction on current and credit card accounts, in the UK Open Banking OBTransaction shape.

| column | meaning |
| --- | --- |
| transaction_id, account_id, booking_datetime | Keys and booking time. |
| amount, currency, credit_debit_indicator | Positive amount in GBP, Credit or Debit. |
| transaction_information | Statement narrative as the customer sees it (truncated and noisy). |
| merchant_name | Card payments only. |
| merchant_category_code | ISO 18245 MCC, card payments only (about 97% present). |
| proprietary_bank_transaction_code | DEB card, DD direct debit, SO standing order, FPO/FPI faster payment out/in, BGC credit, CPT cash, TFR internal transfer, CHG charge. |
| bank_transaction_code | ISO 20022-style family of the payment type. |

### product_holdings (81,118 rows)

Products held and when, including green products and the paperless service.

| column | meaning |
| --- | --- |
| customer_id, account_id, product_code, product_family | What is held. |
| green_product | GP01-GP06 when the holding is one of the six green products. |
| start_date, end_date | Holding period. |

### preference_changes (849 rows)

Statement preference switches (paper to online).

| column | meaning |
| --- | --- |
| account_id, change_date, old_preference, new_preference | History needed for point-in-time features. |

### assets (5,528 rows)

Property and vehicle records the bank holds as lender.

| column | meaning |
| --- | --- |
| property_type, property_detail, construction_period, bedrooms | From the valuation report for mortgages with us. |
| floor_area_m2, epc_band, epc_main_heating, solar_pv_on_record | From the EPC register match (blank when the home has no certificate or was not matched). |
| vehicle_fuel_type, vehicle_first_registered_year, vehicle_registration | For motor loans with us. |
| valuation_at_origination | Property or vehicle value at loan origination. |

### digital_activity (630,000 rows)

Monthly digital engagement per customer.

| column | meaning |
| --- | --- |
| app_logins, web_logins, branch_visits, estatement_views | Counts per month. |
| paperless_all_accounts | Whether every account was on online statements that month. |

### campaign_responses (45,139 rows)

Earlier marketing contacts and responses (seed history for customer memory).

| column | meaning |
| --- | --- |
| campaign_id, sent_date, product, channel | What was sent. |
| response, response_date | no_response, viewed, clicked, dismissed, not_interested or already_have. |

## Truth vault

Training labels come from here for training customers only: transaction categories (FR2) and product take-up (FR3), as a bank would use a labelled history. Everything else (hidden traits, true km and kWh, p_true, validation and test labels) is read only by the evaluation harness; models, rules, the carbon engine and the LLM never see it.

| table | contents |
| --- | --- |
| state | Every customer's true situation: hidden traits (z_*), their observable parts (zmu_*), true km, litres, kWh, heating, EPC, solar, billing visibility, life events, prior holdings, take-up dates, split. |
| labels | Per customer and product: true eligibility, p_true (Bayes-optimal take-up probability), took_up, takeup_date. |
| transaction_truth | Per transaction: true category and sub-category, merchant, local or national merchant, physical quantity (litres or kWh) and fuel type; plus the true account and a hash of the booked record (audit_account_id, audit_fingerprint) for the ingestion audit. |
| merchant_directory | Every payee the simulator can produce, with its true category. |
| system_of_record | Fingerprints (row count, order-independent hash, one hash per column) of each table as the bank's systems held it before extraction; FR1 ingestion must reproduce them exactly. |

## How the data is generated

1. **Population.** Households are sampled per local authority from Census 2021 tenure, accommodation, car, heating, household-size and commute tables. The age of the household reference person matches Census RM201 (tenure by age) and economic activity matches RM133 (tenure by activity) in each authority; dependants follow TS003, occupation TS063, self-employment TS066, bedrooms TS050. Income deciles follow ONS Family Spending.
2. **Physical world.** Each home is a real home from the DESNZ NEED 2025 sample (4 million homes) of the same region, type and heating: its construction period, floor-area band, EPC band, insulation and metered 2023 gas and electricity, re-anchored to the authority's DESNZ averages. Homes NEED records without an EPC have no certificate the bank can match. Cars get a fuel type from DfT local licensing shares, mileage from NTS bands and an efficiency around the DESNZ 2026 average car; solar follows DESNZ sites per household.
3. **Money.** Fuel is priced at that week's DESNZ pump price; energy bills use the Ofgem cap in force each quarter; everyday spend follows ONS spend by income decile. Homes are valued at the UK House Price Index for their authority, type and purchase month; mortgage, loan, card and savings rates are the Bank of England quoted rates for the product and month.
4. **Statements.** Each payment becomes a UK statement line with real merchant names, store numbers, references and truncation; about 11% of spend goes to small local businesses that are unique to each area.
5. **Messiness.** Some customers pay energy from another bank, buy fuel with cash or on another bank's card, or have stale onboarding data.
6. **Life events.** House moves, new cars, babies and retirements are drawn monthly and change later months.
7. **Labels.** Take-up of each green product is drawn from the causal model below, only for truly eligible customers.

## Label-generation rules

For product *p* and customer *c*: logit P(take-up between 1 April and 30 September 2026) = b0 + Σ beta × driver + Σ gamma × hidden trait.
Drivers are the customer's true situation on 31 March 2026; hidden traits are standard-normal variables the platform never sees, so no rule on observed data can reproduce the labels. b0 is solved so take-up among eligible customers hits the target rate. Products are independent and may overlap: a customer can take up several.

### GP01 (b0 = -4.5062, eligible 19,051, take-ups 1,056, rate 0.055, target 0.055)

| driver | beta | definition |
| --- | --- | --- |
| log_km | 0.7 | log(annual petrol or diesel km / 11,000) |
| log_km_sq | -0.3 | square of log_km (very high mileage lowers take-up: range concerns) |
| car_age | 0.35 | car age in years / 10 |
| loan_ending_12m | 1.1 | current car finance ends within 12 months of the cutoff |
| offstreet | 0.6 | home has off-street parking for charging |
| income | 0.25 | (income decile - 5.5) / 2.87 |
| multi_car | 0.2 | household has two or more cars |
| clean_air_charges | 0.8 | pays ULEZ or clean-air-zone charges |
| area_ev_share | 0.3 | (local battery-electric share of private cars - 0.032) / 0.012 |

Hidden traits: green +0.90, liquidity +0.50.

### GP02 (b0 = -5.6231, eligible 6,884, take-ups 327, rate 0.048, target 0.045)

| driver | beta | definition |
| --- | --- | --- |
| epc_low | 0.9 | EPC band D to G |
| heat_intensity | 0.55 | log(heat demand per m2 / 140) |
| recent_move | 0.8 | moved home in the 12 months before the cutoff |
| diy | 0.45 | log of DIY spending intensity |
| income | 0.2 | (income decile - 5.5) / 2.87 |
| mortgage_with_us | 0.5 | mortgage is held with this bank |
| household_size | 0.15 | (household size - 2.4) / 1.3 |

Hidden traits: green +0.70, stay +0.60, liquidity +0.20.

### GP03 (b0 = -4.8877, eligible 9,133, take-ups 293, rate 0.032, target 0.035)

| driver | beta | definition |
| --- | --- | --- |
| log_elec | 0.65 | log(electricity demand / 3,000 kWh) |
| solar_yield | 0.35 | (regional PV yield kWh/kWp - 980) / 50 |
| ev_at_home | 0.7 | household drives a battery-electric or plug-in hybrid car |
| daytime_home | 0.45 | retired or works from home |
| income | 0.3 | (income decile - 5.5) / 2.87 |
| detached | 0.35 | lives in a detached house |

Hidden traits: green +0.80, liquidity +0.50.

### GP04 (b0 = -2.8449999999999998, eligible 7,043, take-ups 849, rate 0.121, target 0.12)

| driver | beta | definition |
| --- | --- | --- |
| log_logins | 0.85 | log(1 + online/app log-ins per month) |
| app_user | 0.7 | uses the mobile app |
| age | -0.45 | (age - 48) / 17 |
| paper_accounts | 0.2 | number of accounts on paper statements - 2 |

Hidden traits: digital +0.90.

### GP05 (b0 = -5.3525, eligible 8,671, take-ups 502, rate 0.058, target 0.06)

| driver | beta | definition |
| --- | --- | --- |
| invests_with_us | 0.6 | holds an investment account with this bank |
| log_investment | 0.35 | log(1 + investment balance / 1,000) / 2 |
| regular_investor | 0.45 | invests every month |
| stated_pref | 1.2 | stated sustainability preference (1, 0, or 0 when unknown) |
| green_supplier | 0.5 | energy supplier is a green-leaning supplier |
| income | 0.25 | (income decile - 5.5) / 2.87 |

Hidden traits: green +1.00, risk +0.55.

### GP06 (b0 = -4.6754999999999995, eligible 27,450, take-ups 948, rate 0.035, target 0.035)

| driver | beta | definition |
| --- | --- | --- |
| existing_borrowing | 0.6 | has a personal loan or credit card borrowing |
| purchase_intent | 1.3 | has a green purchase in mind (shows as bike and appliance shop spend) |
| cyclist | 0.7 | cycles or uses bike share |
| owner_outright_high_gas | 0.5 | owns outright and uses more gas than the local mean |
| income | -0.15 | (income decile - 5.5) / 2.87 |
| age | -0.25 | (age - 48) / 17 |

Hidden traits: green +0.70, liquidity -0.50.

## Transaction categories and boundary rules (FR2)

Every transaction has exactly one of eight categories; sub-categories roll up to one category.

- **B1** Fuel vs Automotive: energy that moves the vehicle is Fuel; anything that keeps a vehicle owned, financed, insured, taxed, maintained or parked is Automotive.
- **B2** Public EV charging is Fuel (transport energy). Home EV charging is part of the household Electricity bill and is not separable.
- **B3** Shop purchases at a fuel forecourt are Other (forecourt_shop). Forecourt terminals report the same merchant and MCC for fuel and shop, so the model separates them using the amount.
- **B4** Electricity or Gas only when the payment is for a single-fuel supply. One payment covering both fuels (dual fuel) is Utilities, as are water and delivered heating oil or LPG.
- **B5** Prepayment top-ups take the category of the meter they top up (Electricity or Gas). Generic top-up networks without a fuel tag are still labelled by the meter.
- **B6** Congestion, ULEZ, clean-air-zone and toll charges are Automotive even when paid to a transport authority; fares paid to the same authority are Public Transport.
- **B7** Taxis and ride-hailing are Other: they are private hire, not shared or scheduled transport. Bike and e-scooter share schemes are Public Transport.
- **B8** Large appliance retailers are Home Improvement (white goods change home energy use); furniture and homeware are Other.
- **B9** Motor insurance and breakdown cover are Automotive; home and life insurance are Other.
- **B10** Refunds and credits take the category of the merchant that paid them (an energy supplier refund stays in its energy category). Income credits are Other.
- **B11** Mortgage and rent payments are Other (housing) but are used as evidence of tenure.

Examples from this dataset:

**B1 Fuel vs Automotive**

| transaction_information | amount | true_category | true_sub_category |
| --- | --- | --- | --- |
| SHELL UK COLCHESTER CD 5975 | 76.88 | Fuel | road_fuel |
| ESURE INSURANCE | 51.19 | Automotive | motor_insurance |
| JET COLCHESTER CD 4770 | 29.86 | Fuel | road_fuel |
| GULF SERVICE STN COLCHEST CD 5975 | 63.2 | Fuel | road_fuel |
| NATIONAL TYRES COLCHESTER CD 5975 | 152.61 | Automotive | maintenance |
| SAINSBURYS PFS 3503 CD 5975 | 58.26 | Fuel | road_fuel |

**B2 Public EV charging is Fuel**

| transaction_information | amount | true_category | true_sub_category |
| --- | --- | --- | --- |
| MER UK CHARGING CD 4147 | 7.16 | Fuel | ev_public_charging |
| INSTAVOLT CD 9973 | 6.05 | Fuel | ev_public_charging |
| BP PULSE RUGBY CD 9973 | 3.91 | Fuel | ev_public_charging |
| CONNECTED KERBS CD 9973 | 7.95 | Fuel | ev_public_charging |
| UBITRICITY CD 9973 | 22.81 | Fuel | ev_public_charging |
| SHELL RECHARGE CD 2537 | 11.87 | Fuel | ev_public_charging |

**B3 Forecourt shop is Other**

| transaction_information | amount | true_category | true_sub_category |
| --- | --- | --- | --- |
| MORRISONS PETROL COLCHEST CD 5975 | 4.47 | Other | forecourt_shop |
| SHELL COLCHESTER CD 5975 | 3.38 | Other | forecourt_shop |
| JET PETROL 9296 CD 4770 | 2.34 | Other | forecourt_shop |
| MOTOR FUEL GROUP TRAFFORD CD 7011 | 5.87 | Other | forecourt_shop |
| ASDA PFS 2322 CD 7415 | 5.47 | Other | forecourt_shop |
| BP 6114 CD 9973 | 4.78 | Other | forecourt_shop |

**B4 Dual fuel and delivered fuels are Utilities**

| transaction_information | amount | true_category | true_sub_category |
| --- | --- | --- | --- |
| OVO ENERGY 19Y1E1P7 | 96.0 | Utilities | dual_fuel_energy |
| E.ON NEXT | 209.0 | Utilities | dual_fuel_energy |
| CHESHIRE EAST OIL SUPPLIE CD 6297 | 240.71 | Utilities | heating_oil_lpg |
| ADAMS HEATING OILS CD 6297 | 253.61 | Utilities | heating_oil_lpg |
| EXETER OIL SUPPLIES CD 6297 | 257.86 | Utilities | heating_oil_lpg |
| JOHNSON HEATING OILS CD 6297 | 264.77 | Utilities | heating_oil_lpg |

**B5 Prepayment top-ups follow the meter**

| transaction_information | amount | true_category | true_sub_category |
| --- | --- | --- | --- |
| PAYPOINT BIRMINGHAM CD 9899 | 20.0 | Electricity | electricity_prepayment |
| OVO ENERGY PREPAY CD 9899 | 20.0 | Electricity | electricity_prepayment |
| PAYZONE BIRMINGHAM CD 9899 | 20.0 | Electricity | electricity_prepayment |
| POST OFFICE BIRMINGHAM CD 9899 | 25.0 | Electricity | electricity_prepayment |
| BG TOP UP ELEC CD 2038 | 10.0 | Electricity | electricity_prepayment |
| BG PAYG GAS CD 2038 | 25.0 | Gas | gas_prepayment |

**B6 Clean-air charges are Automotive**

| transaction_information | amount | true_category | true_sub_category |
| --- | --- | --- | --- |
| GOV.UK CLEAN AIR ZONE CD 9732 | 8.0 | Automotive | parking_charges |
| TFL CONGESTION CHARGE CD 4589 | 15.0 | Automotive | parking_charges |
| TFL.GOV.UK/ULEZ CD 6762 | 12.5 | Automotive | parking_charges |

**B7 Taxis are Other, bike share is Public Transport**

| transaction_information | amount | true_category | true_sub_category |
| --- | --- | --- | --- |
| BOLT.EU/O/1A0KHXWF CD 4770 | 31.51 | Other | taxi_ridehail |
| COLCHESTER TAXIS CD 5975 | 9.92 | Other | taxi_ridehail |
| UBER BV CD 4770 | 23.89 | Other | taxi_ridehail |
| COLCHESTER PRIVATE HIRE CD 5975 | 28.24 | Other | taxi_ridehail |
| TIER DOTT CD 4770 | 3.71 | Public Transport | micromobility |
| BIRMINGHAM TAXIS CD 9899 | 14.22 | Other | taxi_ridehail |

**B8 Appliances are Home Improvement**

| transaction_information | amount | true_category | true_sub_category |
| --- | --- | --- | --- |
| CURRYS BIRMINGHAM CD 9899 | 686.31 | Home Improvement | appliances |
| APPLIANCES DIRECT CD 3966 | 434.41 | Home Improvement | appliances |
| HUGHES ELECTRICAL CD 0966 | 486.91 | Home Improvement | appliances |
| NORWICH DOMESTIC APPLIANC CD 3763 | 711.54 | Home Improvement | appliances |
| AO.COM CD 6783 | 338.16 | Home Improvement | appliances |
| SOUTHAMPTON DOMESTIC APPL CD 1582 | 415.25 | Home Improvement | appliances |

**B9 Motor vs home insurance**

| transaction_information | amount | true_category | true_sub_category |
| --- | --- | --- | --- |
| AVIVA | 31.43 | Other | insurance_other |
| ESURE INSURANCE | 51.19 | Automotive | motor_insurance |
| ADMIRAL INSURANCE | 53.42 | Automotive | motor_insurance |
| AXA INSURANCE UK | 103.8 | Automotive | motor_insurance |
| AXA INSURANCE UK | 45.21 | Other | insurance_other |
| VITALITY HEALTH | 43.93 | Other | insurance_other |

**B10 Supplier credits keep the supplier's category**

| transaction_information | amount | true_category | true_sub_category |
| --- | --- | --- | --- |
| E.ON NEXT | 38.93 | Utilities | dual_fuel_energy |
| OCTOPUS ELEC ACC 19Y4UQT0 | 42.56 | Electricity | electricity_bill |
| OCTOPUS SEG PAYMENT | 73.68 | Electricity | electricity_bill |
| GOOD ENERGY ELECTRICITY | 102.73 | Electricity | electricity_bill |
| SO ENERGY ENERGY 19YAE9KO | 73.77 | Utilities | dual_fuel_energy |
| OVO SEG PAYMENT | 75.27 | Electricity | electricity_bill |

**B11 Housing payments are Other**

| transaction_information | amount | true_category | true_sub_category |
| --- | --- | --- | --- |
| ISLE OF ANGLESEY PROPERTY | 1346.0 | Other | housing |
| WORCESTER PROPERTY MGMT | 909.0 | Other | housing |
| BRISTOL PROPERTY MGMT | 814.0 | Other | housing |
| MTG 4377 | 385.69 | Other | housing |
| NATWEST MORTGAGE | 440.94 | Other | housing |
| ARTHUR BAKER RENT | 954.0 | Other | housing |

## Splits

Customer-level split from a salted hash of the customer id: train 21,062, validation 4,393, test 4,545. All of a customer's rows sit in one split. Features use data up to 31 March 2026; labels use the six months after.

## Validation against public statistics

36 of 36 checks within tolerance.

| check | simulated | reference | relative_gap | pass | source |
| --- | --- | --- | --- | --- | --- |
| tenure share: owned_outright | 0.3288 | 0.3283 | 0.0016 | True | Census 2021 TS054 |
| tenure share: mortgage | 0.2977 | 0.2971 | 0.0022 | True | Census 2021 TS054 |
| tenure share: social_rent | 0.1667 | 0.1706 | 0.0228 | True | Census 2021 TS054 |
| tenure share: private_rent | 0.2067 | 0.204 | 0.0133 | True | Census 2021 TS054 |
| households in flats | 0.2176 | 0.2169 | 0.0033 | True | Census 2021 TS044 |
| households with no car | 0.2327 | 0.2331 | 0.0017 | True | Census 2021 TS045 |
| mains gas central heating | 0.7643 | 0.781 | 0.0214 | True | Census 2021 TS046 |
| battery-electric share of primary cars | 0.0325 | 0.032 | 0.015 | True | DfT VEH0142/VEH0105 2026 Q2 |
| homes with solar PV | 0.0658 | 0.0667 | 0.0132 | True | DESNZ renewables by LA 2025 |
| mean annual mileage, primary car (miles) | 7546.3541 | 7100.0 | 0.0629 | True | NTS0901 2025 |
| mean domestic gas use, gas homes (kWh) | 11360.3848 | 11305.6519 | 0.0048 | True | DESNZ sub-national gas 2024 |
| median domestic electricity use (kWh) | 2585.0 | 2501.857 | 0.0332 | True | DESNZ sub-national electricity 2024 |
| household reference persons aged 65+ (owned outright) | 0.6302 | 0.6271 | 0.0049 | True | Census 2021 RM201 |
| household reference persons aged 65+ (social rent) | 0.2599 | 0.2659 | 0.0225 | True | Census 2021 RM201 |
| household reference persons in work | 0.6283 | 0.6254 | 0.0047 | True | Census 2021 RM133 |
| household reference persons retired | 0.256 | 0.2567 | 0.0027 | True | Census 2021 RM133 |
| self-employed share of workers | 0.1639 | 0.1697 | 0.0342 | True | Census 2021 TS066 |
| households with dependent children | 0.2855 | 0.2837 | 0.0061 | True | Census 2021 TS003 |
| homes with 4 or more bedrooms | 0.2085 | 0.2107 | 0.0103 | True | Census 2021 TS050 |
| homes with an EPC on the register | 0.6681 | 0.6759 | 0.0114 | True | NEED 2025 |
| rated homes in EPC band C or better | 0.4921 | 0.4899 | 0.0044 | True | NEED 2025 |
| mean rate on 2-year fixed mortgages (%) | 5.0585 | 5.0567 | 0.0004 | True | Bank of England quoted rates 2023-24 |
| median age of petrol cars aged 3+ (years) | 10.1 | 9.5 | 0.0632 | True | DVSA MOT 2024 |
| mileage of petrol cars aged 10-14 vs 3-5 (ratio) | 0.8807 | 0.8739 | 0.0077 | True | DVSA MOT 2024 |
| mean monthly fuel spend (GBP) | 84.5227 | 79.9067 | 0.0578 | True | ONS Family Spending FYE 2025 A6 |
| fuel spend gradient across income deciles (correlation) | 0.957 | 1.0 | 0.043 | True | ONS Family Spending FYE 2025 A6 |
| mean monthly food and drink spend (GBP) | 317.5152 | 294.5973 | 0.0778 | True | ONS Family Spending FYE 2025 A6 |
| food and drink spend gradient across income deciles (correlation) | 0.9998 | 1.0 | 0.0002 | True | ONS Family Spending FYE 2025 A6 |
| mean monthly home energy spend (GBP) | 150.8739 | 147.55 | 0.0225 | True | ONS Family Spending FYE 2025 A6 |
| home energy spend gradient across income deciles (correlation) | 0.9602 | 1.0 | 0.0398 | True | ONS Family Spending FYE 2025 A6 |
| GP01 take-up among eligible | 0.0554 | 0.055 | 0.0078 | True | label model target |
| GP02 take-up among eligible | 0.0475 | 0.045 | 0.0556 | True | label model target |
| GP03 take-up among eligible | 0.0321 | 0.035 | 0.0834 | True | label model target |
| GP04 take-up among eligible | 0.1205 | 0.12 | 0.0045 | True | label model target |
| GP05 take-up among eligible | 0.0579 | 0.06 | 0.0351 | True | label model target |
| GP06 take-up among eligible | 0.0345 | 0.035 | 0.0133 | True | label model target |

Monthly fuel spend by income decile (GBP):

| series | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| simulated | 52.4 | 55.9 | 65.8 | 71.3 | 81.1 | 87.5 | 96.9 | 105.2 | 114.3 | 121.4 |
| ons_a6 | 26.4 | 35.5 | 54.6 | 76.3 | 81.0 | 89.7 | 100.1 | 112.2 | 110.1 | 113.1 |

Monthly food and drink spend by income decile (GBP):

| series | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| simulated | 175.4 | 214.9 | 244.4 | 307.9 | 292.3 | 345.9 | 368.1 | 371.1 | 418.3 | 457.2 |
| ons_a6 | 170.2 | 203.1 | 227.8 | 283.6 | 268.7 | 315.1 | 337.7 | 340.1 | 383.4 | 416.3 |

Monthly home energy spend by income decile (GBP):

| series | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| simulated | 137.6 | 135.9 | 141.0 | 143.8 | 147.0 | 150.1 | 157.0 | 160.5 | 164.9 | 174.6 |
| ons_a6 | 108.3 | 123.9 | 127.8 | 140.8 | 147.3 | 143.9 | 152.1 | 163.4 | 167.7 | 200.2 |

## Partial observability

| measure | value |
| --- | --- |
| transactions_per_customer_month | 44.9 |
| card_share | 0.7058 |
| mcc_present_on_card | 0.97 |
| local_merchant_share | 0.1074 |
| customers_with_energy_bills_visible | 0.8559 |
| drivers_buying_some_fuel_in_cash | 0.0988 |
| drivers_fuel_on_external_card | 0.0787 |
| mortgages_visible | 0.9189 |
| customers_eligible_for_two_or_more_products | 0.7857 |

## Product overlap

Customers truly eligible for each pair of products (diagonal = eligible for that product):

| product | GP01 | GP02 | GP03 | GP04 | GP05 | GP06 |
| --- | --- | --- | --- | --- | --- | --- |
| GP01 | 19051 | 5508 | 7182 | 4647 | 6702 | 18521 |
| GP02 | 5508 | 6884 | 3732 | 868 | 3120 | 6805 |
| GP03 | 7182 | 3732 | 9133 | 2602 | 3310 | 8896 |
| GP04 | 4647 | 868 | 2602 | 7043 | 1809 | 6434 |
| GP05 | 6702 | 3120 | 3310 | 1809 | 8671 | 8544 |
| GP06 | 18521 | 6805 | 8896 | 6434 | 8544 | 27450 |

## Assumptions

Every parameter lives in configs/simulation.yaml with its source or the word 'assumption'. The main assumptions:

- Age and income given tenure: [0.02, 0.21, 0.31, 0.27, 0.15, 0.03, 0.01] (mortgagors' age bands, as an example).
- Energy billing: dual-fuel single direct debit 0.68, prepayment 0.12, bills paid from another bank 0.14.
- Heat pump efficiency 2.93 and boiler efficiency 0.84 (Electrification of Heat final report).
- Public charging {'slow': 0.55, 'rapid': 0.79} GBP/kWh; heating oil 0.66 GBP/litre.
- Take-up targets {'GP01': 0.055, 'GP02': 0.045, 'GP03': 0.035, 'GP04': 0.12, 'GP05': 0.06, 'GP06': 0.035}.
- Seasonal shapes of gas, electricity and driving (transactions.py constants).

## Limitations

- England and Wales only: Census 2021 covers these nations; Scottish and Northern Irish customers are not simulated.
- Energy prices use Ofgem GB-average Direct Debit cap rates; regional rate differences are not modelled.
- Each customer is one household's main account holder; joint accounts appear only as bills paid from elsewhere.
- Hidden traits and label coefficients are modelling assumptions, documented here; metrics measure performance on this world, not on real customers.
- Insurers that sell both motor and home cover produce identical statement text for both; the true category comes from the policy type.
- Dual-fuel and single-fuel bills from the same supplier can read identically (rule B4); only fuel-tagged references disclose the fuel.
- Heating oil and LPG prices, public charging prices and water bills are assumptions, not sourced series.

## Sources

- Census 2021 TS054 Tenure, England and Wales (ONS via Nomis, Open Government Licence v3.0): https://www.nomisweb.co.uk/output/census/2021/census2021-ts054.zip
- Census 2021 TS044 Accommodation type, England and Wales (ONS via Nomis, Open Government Licence v3.0): https://www.nomisweb.co.uk/output/census/2021/census2021-ts044.zip
- Census 2021 TS045 Car or van availability, England and Wales (ONS via Nomis, Open Government Licence v3.0): https://www.nomisweb.co.uk/output/census/2021/census2021-ts045.zip
- Census 2021 TS061 Method used to travel to work, England and Wales (ONS via Nomis, Open Government Licence v3.0): https://www.nomisweb.co.uk/output/census/2021/census2021-ts061.zip
- Census 2021 TS046 Central heating, England and Wales (ONS via Nomis, Open Government Licence v3.0): https://www.nomisweb.co.uk/output/census/2021/census2021-ts046.zip
- Census 2021 TS017 Household size, England and Wales (ONS via Nomis, Open Government Licence v3.0): https://www.nomisweb.co.uk/output/census/2021/census2021-ts017.zip
- Census 2021 TS003 Household composition, England and Wales (ONS via Nomis, Open Government Licence v3.0): https://www.nomisweb.co.uk/output/census/2021/census2021-ts003.zip
- Census 2021 TS050 Number of bedrooms, England and Wales (ONS via Nomis, Open Government Licence v3.0): https://www.nomisweb.co.uk/output/census/2021/census2021-ts050.zip
- Census 2021 TS066 Economic activity status, England and Wales (ONS via Nomis, Open Government Licence v3.0): https://www.nomisweb.co.uk/output/census/2021/census2021-ts066.zip
- Census 2021 TS062 National Statistics Socio-economic Classification (NS-SeC), England and Wales (ONS via Nomis, Open Government Licence v3.0): https://www.nomisweb.co.uk/output/census/2021/census2021-ts062.zip
- Census 2021 TS058 Distance travelled to work, England and Wales (ONS via Nomis, Open Government Licence v3.0): https://www.nomisweb.co.uk/output/census/2021/census2021-ts058.zip
- Census 2021 TS011 Households by deprivation dimensions, England and Wales (ONS via Nomis, Open Government Licence v3.0): https://www.nomisweb.co.uk/output/census/2021/census2021-ts011.zip
- Census 2021 TS037 General health, England and Wales (ONS via Nomis, Open Government Licence v3.0): https://www.nomisweb.co.uk/output/census/2021/census2021-ts037.zip
- Census 2021 TS038 Disability, England and Wales (ONS via Nomis, Open Government Licence v3.0): https://www.nomisweb.co.uk/output/census/2021/census2021-ts038.zip
- Census 2021 TS063 Occupation, England and Wales (ONS via Nomis, Open Government Licence v3.0): https://www.nomisweb.co.uk/output/census/2021/census2021-ts063.zip
- Census 2021 TS052 Occupancy rating for bedrooms, England and Wales (ONS via Nomis, Open Government Licence v3.0): https://www.nomisweb.co.uk/output/census/2021/census2021-ts052.zip
- Census 2021 RM201 Tenure by age of household reference person, local authorities (April 2023) (ONS via Nomis, Open Government Licence v3.0): https://www.nomisweb.co.uk/api/v01/dataset/NM_2301_1.data.csv?date=latest&geography=TYPE424&c2021_hhtenure_5=1,2,3,4&c2021_age_4b=1,2,3&measures=20100&select=geography_code,geography_name,c2021_hhtenure_5_name,c2021_age_4b_name,obs_value
- Census 2021 RM133 Tenure by economic activity status of household reference person, local authorities (April 2023) (ONS via Nomis, Open Government Licence v3.0): https://www.nomisweb.co.uk/api/v01/dataset/NM_2233_1.data.csv?date=latest&geography=TYPE424&c2021_hhtenure_5=1,2,3,4&measures=20100&select=geography_code,geography_name,c2021_hhtenure_5_name,c2021_eastat_10,c2021_eastat_10_name,obs_value
- National Energy Efficiency Data-Framework (NEED) anonymised data 2025: 4 million homes in England and Wales with property attributes and metered gas and electricity 2005-2023 (DESNZ, Open Government Licence v3.0): https://assets.publishing.service.gov.uk/media/685c04734cd6b03168709827/anon2025_4million.zip
- NEED anonymised data 2025: variable definitions (DESNZ, Open Government Licence v3.0): https://assets.publishing.service.gov.uk/media/685c02aa0433072fce0e1025/NEED-2025-anonymised-dataset-metadata.ods
- UK House Price Index full file (July 2026): average prices by local authority and property type, monthly (HM Land Registry, Open Government Licence v3.0): https://publicdata.landregistry.gov.uk/market-trend-data/house-price-index-data/UK-HPI-full-file-2026-07.csv
- Anonymised MOT test results 2024 (make, model, fuel, first-use date, odometer) for every MOT test in Great Britain (DVSA, Open Government Licence v3.0): https://edh-dvsa-data-gov-uk-files-prod.s3.eu-west-1.amazonaws.com/MOT+testing+data+results+(2024).zip
- Quoted household interest rates, monthly, Jan 2015 to Sep 2026: mortgages by LTV and fixed period, personal loans, credit cards, overdrafts, cash ISAs, fixed-rate bonds (Bank of England, Bank of England database terms of use): https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp?csv.x=yes&Datefrom=01/Jan/2015&Dateto=now&SeriesCodes=IUMZICQ,IUMBV34,IUMZICR,IUMB482,IUM2WTL,IUMBV37,IUMBV42,IUMBV45,IUMBX67,IUMHPTL,IUMCCTL,IUMODTL,IUMB6VL,IUMB6VM,IUMB6VN,IUMZID2,IUMWTFA,IUMB6RH,IUMB6RI&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N
- Sub-national electricity consumption by local authority, 2005-2024 (DESNZ, Open Government Licence v3.0): https://assets.publishing.service.gov.uk/media/69426e905431f4f94d7f0b5e/Subnational_electricity_consumption_statistics_2005-2024.xlsx
- Sub-national gas consumption by local authority (weather corrected), 2005-2024 (DESNZ, Open Government Licence v3.0): https://assets.publishing.service.gov.uk/media/69f9bf208f3213f7384ecae6/Subnational_gas_consumption_statistics_2005-2024.xlsx
- Distribution of domestic electricity consumption per meter, GB (DESNZ, Open Government Licence v3.0): https://assets.publishing.service.gov.uk/media/69426ec31ec67214e98f3058/Domestic_electricity_consumption_distribution_2015-2024.xlsx
- Distribution of domestic gas consumption per meter, GB (DESNZ, Open Government Licence v3.0): https://assets.publishing.service.gov.uk/media/69457621033693d5d50eb843/Domestic_gas_consumption_distribution_2015-2024.xlsx
- Renewable electricity sites and capacity by local authority, 2014-2025 (DESNZ, Open Government Licence v3.0): https://assets.publishing.service.gov.uk/media/6aba6ebafe72ed1e2b02f151/Renewable_electricity_by_local_authority_2014_2025.xlsx
- UK Government GHG conversion factors for company reporting 2026, flat file v1.2 (DESNZ, Open Government Licence v3.0): https://assets.publishing.service.gov.uk/media/6a6c9748862aaf18d9c62ac9/ghg-conversion-factors-2026-flat-format-revised.xlsx
- Weekly road fuel prices (pence per litre), 2018 to October 2026 (DESNZ, Open Government Licence v3.0): https://assets.publishing.service.gov.uk/media/6ac3b5951ef3e896de979385/CSV__2018_-__.csv
- VEH0105 licensed vehicles by body type, fuel type, keepership and local authority (DfT, Open Government Licence v3.0): https://assets.publishing.service.gov.uk/media/6aad561bce3f006bd4346ce1/veh0105.ods
- VEH0142 licensed plug-in vehicles (battery electric, plug-in hybrid) by body type, keepership and local authority (DfT, Open Government Licence v3.0): https://assets.publishing.service.gov.uk/media/6aad561b086ce7c57b818707/veh0142.ods
- NTS0901 annual mileage of cars by fuel type, England (DfT, Open Government Licence v3.0): https://assets.publishing.service.gov.uk/media/6a9ed42af36e1f225ecb7eab/nts0901.ods
- NTS0904 annual mileage bands of cars, England (DfT, Open Government Licence v3.0): https://assets.publishing.service.gov.uk/media/6a9ed42a474b8101ece46448/nts0904.ods
- Family spending workbook 1, FYE 2025 (table A6 by gross income decile) (ONS, Open Government Licence v3.0): https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/personalandhouseholdfinances/expenditure/datasets/familyspendingworkbook1detailedexpenditureandtrends/fye2025/workbook1detailedexpenditureandtrends.xlsx
- Atmospheric emissions: greenhouse gas emissions intensity by industry (SIC sections), 1990-2024 (ONS, Open Government Licence v3.0): https://www.ons.gov.uk/file?uri=/economy/environmentalaccounts/datasets/ukenvironmentalaccountsatmosphericemissionsgreenhousegasemissionsintensitybyeconomicsectorunitedkingdom/current/04atmosphericemissionsghgintensity.xlsx
- ISO 18245 merchant category codes with descriptions (greggles/mcc-codes (GitHub), The Unlicense): https://raw.githubusercontent.com/greggles/mcc-codes/main/mcc_codes.csv
- 68,000 synthetic US bank transaction descriptions in 17 categories (DoDataThings (Hugging Face), MIT): https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2/resolve/main/transactions-synthetic.csv
- Price cap announcement 2025Q1 (Ofgem, Public web page): https://www.ofgem.gov.uk/news/changes-energy-price-cap-between-1-january-and-31-march-2025
- Price cap announcement 2025Q2 (Ofgem, Public web page): https://www.ofgem.gov.uk/news/changes-energy-price-cap-between-1-april-and-30-june-2025
- Price cap announcement 2025Q3 (Ofgem, Public web page): https://www.ofgem.gov.uk/news/changes-energy-price-cap-between-1-july-and-30-september-2025
- Price cap announcement 2025Q4 (Ofgem, Public web page): https://www.ofgem.gov.uk/news/changes-energy-price-cap-between-1-october-and-31-december-2025
- Price cap announcement 2026Q1 (Ofgem, Public web page): https://www.ofgem.gov.uk/news/changes-energy-price-cap-between-1-january-and-31-march-2026
- Price cap announcement 2026Q2 (Ofgem, Public web page): https://www.ofgem.gov.uk/news/changes-energy-price-cap-between-1-april-and-30-june-2026
- Price cap announcement 2026Q3 (Ofgem, Public web page): https://www.ofgem.gov.uk/news/changes-energy-price-cap-between-1-july-and-30-september-2026
- Price cap announcement 2026Q4 (Ofgem, Public web page): https://www.ofgem.gov.uk/news/changes-energy-price-cap-between-1-october-and-31-december-2026
- PV yield at North East reference point (European Commission JRC, Free use (EU JRC PVGIS 5.3)): https://re.jrc.ec.europa.eu/api/v5_3/PVcalc?lat=54.973&lon=-1.614&peakpower=1&loss=14&optimalangles=1&outputformat=json
- PV yield at North West reference point (European Commission JRC, Free use (EU JRC PVGIS 5.3)): https://re.jrc.ec.europa.eu/api/v5_3/PVcalc?lat=53.48&lon=-2.242&peakpower=1&loss=14&optimalangles=1&outputformat=json
- PV yield at Yorkshire and The Humber reference point (European Commission JRC, Free use (EU JRC PVGIS 5.3)): https://re.jrc.ec.europa.eu/api/v5_3/PVcalc?lat=53.8&lon=-1.549&peakpower=1&loss=14&optimalangles=1&outputformat=json
- PV yield at East Midlands reference point (European Commission JRC, Free use (EU JRC PVGIS 5.3)): https://re.jrc.ec.europa.eu/api/v5_3/PVcalc?lat=52.955&lon=-1.149&peakpower=1&loss=14&optimalangles=1&outputformat=json
- PV yield at West Midlands reference point (European Commission JRC, Free use (EU JRC PVGIS 5.3)): https://re.jrc.ec.europa.eu/api/v5_3/PVcalc?lat=52.486&lon=-1.89&peakpower=1&loss=14&optimalangles=1&outputformat=json
- PV yield at East reference point (European Commission JRC, Free use (EU JRC PVGIS 5.3)): https://re.jrc.ec.europa.eu/api/v5_3/PVcalc?lat=52.205&lon=0.122&peakpower=1&loss=14&optimalangles=1&outputformat=json
- PV yield at London reference point (European Commission JRC, Free use (EU JRC PVGIS 5.3)): https://re.jrc.ec.europa.eu/api/v5_3/PVcalc?lat=51.507&lon=-0.128&peakpower=1&loss=14&optimalangles=1&outputformat=json
- PV yield at South East reference point (European Commission JRC, Free use (EU JRC PVGIS 5.3)): https://re.jrc.ec.europa.eu/api/v5_3/PVcalc?lat=51.236&lon=-0.57&peakpower=1&loss=14&optimalangles=1&outputformat=json
- PV yield at South West reference point (European Commission JRC, Free use (EU JRC PVGIS 5.3)): https://re.jrc.ec.europa.eu/api/v5_3/PVcalc?lat=50.9&lon=-3.2&peakpower=1&loss=14&optimalangles=1&outputformat=json
- PV yield at Wales reference point (European Commission JRC, Free use (EU JRC PVGIS 5.3)): https://re.jrc.ec.europa.eu/api/v5_3/PVcalc?lat=51.62&lon=-3.4&peakpower=1&loss=14&optimalangles=1&outputformat=json
