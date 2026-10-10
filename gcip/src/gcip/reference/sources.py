"""Every public source the platform uses: where it lives, its licence and its cache path.

The URLs are the exact files read for the October 2026 build. GOV.UK and ONS
content is published under the Open Government Licence v3.0.
"""

from __future__ import annotations

from dataclasses import dataclass

OGL = "Open Government Licence v3.0"


@dataclass(frozen=True)
class Source:
    key: str
    url: str
    cache_path: str  # relative to the raw cache directory
    licence: str
    publisher: str
    description: str


CENSUS_TABLES = {
    "ts054": "Tenure",
    "ts044": "Accommodation type",
    "ts045": "Car or van availability",
    "ts061": "Method used to travel to work",
    "ts046": "Central heating",
    "ts017": "Household size",
}

# Further Census 2021 topics used for households and area context (counts per 2021 district).
CENSUS_EXTRA_TABLES = {
    "ts003": "Household composition",
    "ts050": "Number of bedrooms",
    "ts066": "Economic activity status",
    "ts062": "National Statistics Socio-economic Classification (NS-SeC)",
    "ts058": "Distance travelled to work",
    "ts011": "Households by deprivation dimensions",
    "ts037": "General health",
    "ts038": "Disability",
    "ts063": "Occupation",
    "ts052": "Occupancy rating for bedrooms",
}

BOE_QUOTED_RATES = {          # Bank of England quoted household interest rates (monthly, %)
    "IUMZICQ": "mortgage_2y_fixed_60ltv",
    "IUMBV34": "mortgage_2y_fixed_75ltv",
    "IUMZICR": "mortgage_2y_fixed_85ltv",
    "IUMB482": "mortgage_2y_fixed_90ltv",
    "IUM2WTL": "mortgage_2y_fixed_95ltv",
    "IUMBV37": "mortgage_3y_fixed_75ltv",
    "IUMBV42": "mortgage_5y_fixed_75ltv",
    "IUMBV45": "mortgage_10y_fixed_75ltv",
    "IUMBX67": "personal_loan_5k",
    "IUMHPTL": "personal_loan_10k",
    "IUMCCTL": "credit_card",
    "IUMODTL": "overdraft",
    "IUMB6VL": "cash_isa_variable",
    "IUMB6VM": "cash_isa_variable_excl_bonus",
    "IUMB6VN": "cash_isa_fixed_1y",
    "IUMZID2": "cash_isa_fixed_2y",
    "IUMWTFA": "bond_fixed_1y",
    "IUMB6RH": "bond_fixed_2y",
    "IUMB6RI": "bond_fixed_3y",
}
BOE_IADB_CSV = ("https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp?csv.x=yes&Datefrom=01/Jan/2015"
                "&Dateto=now&SeriesCodes={codes}&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N")
NOMIS_API = "https://www.nomisweb.co.uk/api/v01/dataset/{dataset}.data.csv?date=latest&geography=TYPE424&{query}"

SOURCES: list[Source] = [
    *[
        Source(
            key=f"census_{table}",
            url=f"https://www.nomisweb.co.uk/output/census/2021/census2021-{table}.zip",
            cache_path=f"census/census2021-{table}.zip",
            licence=OGL,
            publisher="ONS via Nomis",
            description=f"Census 2021 {table.upper()} {label}, England and Wales",
        )
        for table, label in CENSUS_TABLES.items()
    ],
    *[
        Source(
            key=f"census_{table}",
            url=f"https://www.nomisweb.co.uk/output/census/2021/census2021-{table}.zip",
            cache_path=f"census/census2021-{table}.zip",
            licence=OGL,
            publisher="ONS via Nomis",
            description=f"Census 2021 {table.upper()} {label}, England and Wales",
        )
        for table, label in CENSUS_EXTRA_TABLES.items()
    ],
    Source(
        "census_rm201",
        NOMIS_API.format(dataset="NM_2301_1", query="c2021_hhtenure_5=1,2,3,4&c2021_age_4b=1,2,3&measures=20100"
                         "&select=geography_code,geography_name,c2021_hhtenure_5_name,c2021_age_4b_name,obs_value"),
        "census_rm/rm201_tenure_by_age_hrp.csv", OGL, "ONS via Nomis",
        "Census 2021 RM201 Tenure by age of household reference person, local authorities (April 2023)",
    ),
    Source(
        "census_rm133",
        NOMIS_API.format(dataset="NM_2233_1", query="c2021_hhtenure_5=1,2,3,4&measures=20100"
                         "&select=geography_code,geography_name,c2021_hhtenure_5_name,c2021_eastat_10,c2021_eastat_10_name,obs_value"),
        "census_rm/rm133_tenure_by_activity_hrp.csv", OGL, "ONS via Nomis",
        "Census 2021 RM133 Tenure by economic activity status of household reference person, local authorities (April 2023)",
    ),
    Source(
        "desnz_need_2025_4m",
        "https://assets.publishing.service.gov.uk/media/685c04734cd6b03168709827/anon2025_4million.zip",
        "need/anon2025_4million.zip", OGL, "DESNZ",
        "National Energy Efficiency Data-Framework (NEED) anonymised data 2025: 4 million homes in England and Wales "
        "with property attributes and metered gas and electricity 2005-2023",
    ),
    Source(
        "desnz_need_2025_metadata",
        "https://assets.publishing.service.gov.uk/media/685c02aa0433072fce0e1025/NEED-2025-anonymised-dataset-metadata.ods",
        "need/NEED-2025-anonymised-dataset-metadata.ods", OGL, "DESNZ",
        "NEED anonymised data 2025: variable definitions",
    ),
    Source(
        "hmlr_uk_hpi_2026_07",
        "https://publicdata.landregistry.gov.uk/market-trend-data/house-price-index-data/UK-HPI-full-file-2026-07.csv",
        "hpi/UK-HPI-full-file-2026-07.csv", OGL, "HM Land Registry",
        "UK House Price Index full file (July 2026): average prices by local authority and property type, monthly",
    ),
    Source(
        "dvsa_mot_results_2024",
        "https://edh-dvsa-data-gov-uk-files-prod.s3.eu-west-1.amazonaws.com/MOT+testing+data+results+(2024).zip",
        "mot/mot_results_2024.zip", OGL, "DVSA",
        "Anonymised MOT test results 2024 (make, model, fuel, first-use date, odometer) for every MOT test in Great Britain",
    ),
    Source(
        "boe_quoted_household_rates",
        BOE_IADB_CSV.format(codes=",".join(BOE_QUOTED_RATES)),
        "boe/quoted_household_rates.csv", "Bank of England database terms of use", "Bank of England",
        "Quoted household interest rates, monthly, Jan 2015 to Sep 2026: mortgages by LTV and fixed period, personal loans, "
        "credit cards, overdrafts, cash ISAs, fixed-rate bonds",
    ),
    Source(
        "desnz_elec_la",
        "https://assets.publishing.service.gov.uk/media/69426e905431f4f94d7f0b5e/Subnational_electricity_consumption_statistics_2005-2024.xlsx",
        "desnz/elec_la.xlsx", OGL, "DESNZ",
        "Sub-national electricity consumption by local authority, 2005-2024",
    ),
    Source(
        "desnz_gas_la",
        "https://assets.publishing.service.gov.uk/media/69f9bf208f3213f7384ecae6/Subnational_gas_consumption_statistics_2005-2024.xlsx",
        "desnz/gas_la.xlsx", OGL, "DESNZ",
        "Sub-national gas consumption by local authority (weather corrected), 2005-2024",
    ),
    Source(
        "desnz_elec_dist",
        "https://assets.publishing.service.gov.uk/media/69426ec31ec67214e98f3058/Domestic_electricity_consumption_distribution_2015-2024.xlsx",
        "desnz/elec_dist.xlsx", OGL, "DESNZ",
        "Distribution of domestic electricity consumption per meter, GB",
    ),
    Source(
        "desnz_gas_dist",
        "https://assets.publishing.service.gov.uk/media/69457621033693d5d50eb843/Domestic_gas_consumption_distribution_2015-2024.xlsx",
        "desnz/gas_dist.xlsx", OGL, "DESNZ",
        "Distribution of domestic gas consumption per meter, GB",
    ),
    Source(
        "desnz_renewables_la",
        "https://assets.publishing.service.gov.uk/media/6aba6ebafe72ed1e2b02f151/Renewable_electricity_by_local_authority_2014_2025.xlsx",
        "desnz/renew_la.xlsx", OGL, "DESNZ",
        "Renewable electricity sites and capacity by local authority, 2014-2025",
    ),
    Source(
        "desnz_ghg_factors_2026",
        "https://assets.publishing.service.gov.uk/media/6a6c9748862aaf18d9c62ac9/ghg-conversion-factors-2026-flat-format-revised.xlsx",
        "desnz/ghg_factors_2026_flat.xlsx", OGL, "DESNZ",
        "UK Government GHG conversion factors for company reporting 2026, flat file v1.2",
    ),
    Source(
        "desnz_fuel_prices",
        "https://assets.publishing.service.gov.uk/media/6ac3b5951ef3e896de979385/CSV__2018_-__.csv",
        "desnz/weekly_fuel_prices_2018_2026.csv", OGL, "DESNZ",
        "Weekly road fuel prices (pence per litre), 2018 to October 2026",
    ),
    Source(
        "dft_veh0105",
        "https://assets.publishing.service.gov.uk/media/6aad561bce3f006bd4346ce1/veh0105.ods",
        "dft/veh0105.ods", OGL, "DfT",
        "VEH0105 licensed vehicles by body type, fuel type, keepership and local authority",
    ),
    Source(
        "dft_veh0142",
        "https://assets.publishing.service.gov.uk/media/6aad561b086ce7c57b818707/veh0142.ods",
        "dft/veh0142.ods", OGL, "DfT",
        "VEH0142 licensed plug-in vehicles (battery electric, plug-in hybrid) by body type, keepership and local authority",
    ),
    Source(
        "dft_nts0901",
        "https://assets.publishing.service.gov.uk/media/6a9ed42af36e1f225ecb7eab/nts0901.ods",
        "nts/nts0901.ods", OGL, "DfT",
        "NTS0901 annual mileage of cars by fuel type, England",
    ),
    Source(
        "dft_nts0904",
        "https://assets.publishing.service.gov.uk/media/6a9ed42a474b8101ece46448/nts0904.ods",
        "nts/nts0904.ods", OGL, "DfT",
        "NTS0904 annual mileage bands of cars, England",
    ),
    Source(
        "ons_family_spending_wb1",
        "https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/personalandhouseholdfinances/expenditure/datasets/familyspendingworkbook1detailedexpenditureandtrends/fye2025/workbook1detailedexpenditureandtrends.xlsx",
        "ons/fs_wb1_fye2025.xlsx", OGL, "ONS",
        "Family spending workbook 1, FYE 2025 (table A6 by gross income decile)",
    ),
    Source(
        "ons_ghg_intensity_by_industry",
        "https://www.ons.gov.uk/file?uri=/economy/environmentalaccounts/datasets/ukenvironmentalaccountsatmosphericemissionsgreenhousegasemissionsintensitybyeconomicsectorunitedkingdom/current/04atmosphericemissionsghgintensity.xlsx",
        "ons/ghg_intensity_by_industry.xlsx", OGL, "ONS",
        "Atmospheric emissions: greenhouse gas emissions intensity by industry (SIC sections), 1990-2024",
    ),
    Source(
        "mcc_codes",
        "https://raw.githubusercontent.com/greggles/mcc-codes/main/mcc_codes.csv",
        "mcc/mcc_codes.csv", "The Unlicense", "greggles/mcc-codes (GitHub)",
        "ISO 18245 merchant category codes with descriptions",
    ),
    Source(
        "bench_dodatathings",
        "https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2/resolve/main/transactions-synthetic.csv",
        "bench/dodatathings_transactions_synthetic.csv", "MIT", "DoDataThings (Hugging Face)",
        "68,000 synthetic US bank transaction descriptions in 17 categories",
    ),
]

# Ofgem announces each quarterly cap on a news page; rates are GB averages for
# Direct Debit customers including VAT (electricity VAT is 0% from 1 Oct 2026).
OFGEM_QUARTERS = [
    ("2025Q1", "2025-01-01", "2025-03-31", "1-january-and-31-march-2025"),
    ("2025Q2", "2025-04-01", "2025-06-30", "1-april-and-30-june-2025"),
    ("2025Q3", "2025-07-01", "2025-09-30", "1-july-and-30-september-2025"),
    ("2025Q4", "2025-10-01", "2025-12-31", "1-october-and-31-december-2025"),
    ("2026Q1", "2026-01-01", "2026-03-31", "1-january-and-31-march-2026"),
    ("2026Q2", "2026-04-01", "2026-06-30", "1-april-and-30-june-2026"),
    ("2026Q3", "2026-07-01", "2026-09-30", "1-july-and-30-september-2026"),
    ("2026Q4", "2026-10-01", "2026-12-31", "1-october-and-31-december-2026"),
]
OFGEM_URL = "https://www.ofgem.gov.uk/news/changes-energy-price-cap-between-{slug}"

# One representative point per English region and Wales for the PVGIS solar yield
# (the largest city or the population centre). Solar yield varies smoothly with
# latitude, so a regional value is close enough for portfolio-level estimates.
REGION_POINTS = {
    "North East": (54.973, -1.614),
    "North West": (53.480, -2.242),
    "Yorkshire and The Humber": (53.800, -1.549),
    "East Midlands": (52.955, -1.149),
    "West Midlands": (52.486, -1.890),
    "East": (52.205, 0.122),
    "London": (51.507, -0.128),
    "South East": (51.236, -0.570),
    "South West": (50.900, -3.200),
    "Wales": (51.620, -3.400),
}
PVGIS_URL = (
    "https://re.jrc.ec.europa.eu/api/v5_3/PVcalc?lat={lat}&lon={lon}"
    "&peakpower=1&loss=14&optimalangles=1&outputformat=json"
)

# Census 2021 lower-tier local authorities that were merged into new unitary
# authorities on 1 April 2023. All other datasets use the 2023 geography.
MERGED_2023 = {
    "Allerdale": "Cumberland",
    "Carlisle": "Cumberland",
    "Copeland": "Cumberland",
    "Barrow-in-Furness": "Westmorland and Furness",
    "Eden": "Westmorland and Furness",
    "South Lakeland": "Westmorland and Furness",
    "Craven": "North Yorkshire",
    "Hambleton": "North Yorkshire",
    "Harrogate": "North Yorkshire",
    "Richmondshire": "North Yorkshire",
    "Ryedale": "North Yorkshire",
    "Scarborough": "North Yorkshire",
    "Selby": "North Yorkshire",
    "Mendip": "Somerset",
    "Sedgemoor": "Somerset",
    "Somerset West and Taunton": "Somerset",
    "South Somerset": "Somerset",
}
