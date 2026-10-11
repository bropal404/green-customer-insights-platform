# Data sources

Every external dataset the platform uses, where it comes from, its licence and what we use it for. All are public and
contain no personal data; every customer in the platform is synthetic. Generated from `data/reference/manifest.json`
(which also records each file's size and SHA-256).

## How the data is organised

| Layer | Where | In the repo? |
|---|---|---|
| Raw public downloads (57 files below) | `raw_cache/<source>/`, fetched by `scripts/build_reference.py` | No (about 2 GB; re-downloadable from the publishers) |
| Processed public reference tables | `data/reference/` (+ `manifest.json`) | Yes (24 MB) |
| Merchant and narrative lexicon | `data/lexicon/` | Yes |
| External categoriser benchmark (68,000 US descriptions) | `data/benchmarks/` | Yes |
| Synthetic bank: 7 source-system extracts, warehouse, truth vault, Customer 360, features, categorised transactions, labels | `runs/full/` | No: release bundles `gcip_bank_core` and `gcip_bank_full` (see `release/manifest.json`) |
| Pipeline outputs: models, scores, impact, recommendations, explanations, insights, metrics | `runs/pipeline/` | No: release bundle `gcip_outputs` |
| Local LLM (Qwen/Qwen2.5-1.5B-Instruct) | Hugging Face cache | No: downloaded by `scripts/fetch_release.py --model` |

Geography: England and Wales, local authorities on April 2023 boundaries (Census 2021 merged where needed).

## Raw public sources

| Key | Publisher | What it is | Licence | Link |
|---|---|---|---|---|
| `census_ts054` | ONS via Nomis | Census 2021 TS054 Tenure, England and Wales | Open Government Licence v3.0 | [download](https://www.nomisweb.co.uk/output/census/2021/census2021-ts054.zip) |
| `census_ts044` | ONS via Nomis | Census 2021 TS044 Accommodation type, England and Wales | Open Government Licence v3.0 | [download](https://www.nomisweb.co.uk/output/census/2021/census2021-ts044.zip) |
| `census_ts045` | ONS via Nomis | Census 2021 TS045 Car or van availability, England and Wales | Open Government Licence v3.0 | [download](https://www.nomisweb.co.uk/output/census/2021/census2021-ts045.zip) |
| `census_ts061` | ONS via Nomis | Census 2021 TS061 Method used to travel to work, England and Wales | Open Government Licence v3.0 | [download](https://www.nomisweb.co.uk/output/census/2021/census2021-ts061.zip) |
| `census_ts046` | ONS via Nomis | Census 2021 TS046 Central heating, England and Wales | Open Government Licence v3.0 | [download](https://www.nomisweb.co.uk/output/census/2021/census2021-ts046.zip) |
| `census_ts017` | ONS via Nomis | Census 2021 TS017 Household size, England and Wales | Open Government Licence v3.0 | [download](https://www.nomisweb.co.uk/output/census/2021/census2021-ts017.zip) |
| `census_ts003` | ONS via Nomis | Census 2021 TS003 Household composition, England and Wales | Open Government Licence v3.0 | [download](https://www.nomisweb.co.uk/output/census/2021/census2021-ts003.zip) |
| `census_ts050` | ONS via Nomis | Census 2021 TS050 Number of bedrooms, England and Wales | Open Government Licence v3.0 | [download](https://www.nomisweb.co.uk/output/census/2021/census2021-ts050.zip) |
| `census_ts066` | ONS via Nomis | Census 2021 TS066 Economic activity status, England and Wales | Open Government Licence v3.0 | [download](https://www.nomisweb.co.uk/output/census/2021/census2021-ts066.zip) |
| `census_ts062` | ONS via Nomis | Census 2021 TS062 National Statistics Socio-economic Classification (NS-SeC), England and Wales | Open Government Licence v3.0 | [download](https://www.nomisweb.co.uk/output/census/2021/census2021-ts062.zip) |
| `census_ts058` | ONS via Nomis | Census 2021 TS058 Distance travelled to work, England and Wales | Open Government Licence v3.0 | [download](https://www.nomisweb.co.uk/output/census/2021/census2021-ts058.zip) |
| `census_ts011` | ONS via Nomis | Census 2021 TS011 Households by deprivation dimensions, England and Wales | Open Government Licence v3.0 | [download](https://www.nomisweb.co.uk/output/census/2021/census2021-ts011.zip) |
| `census_ts037` | ONS via Nomis | Census 2021 TS037 General health, England and Wales | Open Government Licence v3.0 | [download](https://www.nomisweb.co.uk/output/census/2021/census2021-ts037.zip) |
| `census_ts038` | ONS via Nomis | Census 2021 TS038 Disability, England and Wales | Open Government Licence v3.0 | [download](https://www.nomisweb.co.uk/output/census/2021/census2021-ts038.zip) |
| `census_ts063` | ONS via Nomis | Census 2021 TS063 Occupation, England and Wales | Open Government Licence v3.0 | [download](https://www.nomisweb.co.uk/output/census/2021/census2021-ts063.zip) |
| `census_ts052` | ONS via Nomis | Census 2021 TS052 Occupancy rating for bedrooms, England and Wales | Open Government Licence v3.0 | [download](https://www.nomisweb.co.uk/output/census/2021/census2021-ts052.zip) |
| `census_rm201` | ONS via Nomis | Census 2021 RM201 Tenure by age of household reference person, local authorities (April 2023) | Open Government Licence v3.0 | [download](https://www.nomisweb.co.uk/api/v01/dataset/NM_2301_1.data.csv?date=latest&geography=TYPE424&c2021_hhtenure_5=1,2,3,4&c2021_age_4b=1,2,3&measures=20100&select=geography_code,geography_name,c2021_hhtenure_5_name,c2021_age_4b_name,obs_value) |
| `census_rm133` | ONS via Nomis | Census 2021 RM133 Tenure by economic activity status of household reference person, local authorities (April 2023) | Open Government Licence v3.0 | [download](https://www.nomisweb.co.uk/api/v01/dataset/NM_2233_1.data.csv?date=latest&geography=TYPE424&c2021_hhtenure_5=1,2,3,4&measures=20100&select=geography_code,geography_name,c2021_hhtenure_5_name,c2021_eastat_10,c2021_eastat_10_name,obs_value) |
| `desnz_need_2025_4m` | DESNZ | National Energy Efficiency Data-Framework (NEED) anonymised data 2025: 4 million homes in England and Wales with property attributes and metered gas and electricity 2005-2023 | Open Government Licence v3.0 | [download](https://assets.publishing.service.gov.uk/media/685c04734cd6b03168709827/anon2025_4million.zip) |
| `desnz_need_2025_metadata` | DESNZ | NEED anonymised data 2025: variable definitions | Open Government Licence v3.0 | [download](https://assets.publishing.service.gov.uk/media/685c02aa0433072fce0e1025/NEED-2025-anonymised-dataset-metadata.ods) |
| `hmlr_uk_hpi_2026_07` | HM Land Registry | UK House Price Index full file (July 2026): average prices by local authority and property type, monthly | Open Government Licence v3.0 | [download](https://publicdata.landregistry.gov.uk/market-trend-data/house-price-index-data/UK-HPI-full-file-2026-07.csv) |
| `dvsa_mot_results_2024` | DVSA | Anonymised MOT test results 2024 (make, model, fuel, first-use date, odometer) for every MOT test in Great Britain | Open Government Licence v3.0 | [download](https://edh-dvsa-data-gov-uk-files-prod.s3.eu-west-1.amazonaws.com/MOT+testing+data+results+(2024).zip) |
| `boe_quoted_household_rates` | Bank of England | Quoted household interest rates, monthly, Jan 2015 to Sep 2026: mortgages by LTV and fixed period, personal loans, credit cards, overdrafts, cash ISAs, fixed-rate bonds | Bank of England database terms of use | [download](https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp?csv.x=yes&Datefrom=01/Jan/2015&Dateto=now&SeriesCodes=IUMZICQ,IUMBV34,IUMZICR,IUMB482,IUM2WTL,IUMBV37,IUMBV42,IUMBV45,IUMBX67,IUMHPTL,IUMCCTL,IUMODTL,IUMB6VL,IUMB6VM,IUMB6VN,IUMZID2,IUMWTFA,IUMB6RH,IUMB6RI&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N) |
| `desnz_elec_la` | DESNZ | Sub-national electricity consumption by local authority, 2005-2024 | Open Government Licence v3.0 | [download](https://assets.publishing.service.gov.uk/media/69426e905431f4f94d7f0b5e/Subnational_electricity_consumption_statistics_2005-2024.xlsx) |
| `desnz_gas_la` | DESNZ | Sub-national gas consumption by local authority (weather corrected), 2005-2024 | Open Government Licence v3.0 | [download](https://assets.publishing.service.gov.uk/media/69f9bf208f3213f7384ecae6/Subnational_gas_consumption_statistics_2005-2024.xlsx) |
| `desnz_elec_dist` | DESNZ | Distribution of domestic electricity consumption per meter, GB | Open Government Licence v3.0 | [download](https://assets.publishing.service.gov.uk/media/69426ec31ec67214e98f3058/Domestic_electricity_consumption_distribution_2015-2024.xlsx) |
| `desnz_gas_dist` | DESNZ | Distribution of domestic gas consumption per meter, GB | Open Government Licence v3.0 | [download](https://assets.publishing.service.gov.uk/media/69457621033693d5d50eb843/Domestic_gas_consumption_distribution_2015-2024.xlsx) |
| `desnz_renewables_la` | DESNZ | Renewable electricity sites and capacity by local authority, 2014-2025 | Open Government Licence v3.0 | [download](https://assets.publishing.service.gov.uk/media/6aba6ebafe72ed1e2b02f151/Renewable_electricity_by_local_authority_2014_2025.xlsx) |
| `desnz_ghg_factors_2026` | DESNZ | UK Government GHG conversion factors for company reporting 2026, flat file v1.2 | Open Government Licence v3.0 | [download](https://assets.publishing.service.gov.uk/media/6a6c9748862aaf18d9c62ac9/ghg-conversion-factors-2026-flat-format-revised.xlsx) |
| `desnz_fuel_prices` | DESNZ | Weekly road fuel prices (pence per litre), 2018 to October 2026 | Open Government Licence v3.0 | [download](https://assets.publishing.service.gov.uk/media/6ac3b5951ef3e896de979385/CSV__2018_-__.csv) |
| `dft_veh0105` | DfT | VEH0105 licensed vehicles by body type, fuel type, keepership and local authority | Open Government Licence v3.0 | [download](https://assets.publishing.service.gov.uk/media/6aad561bce3f006bd4346ce1/veh0105.ods) |
| `dft_veh0142` | DfT | VEH0142 licensed plug-in vehicles (battery electric, plug-in hybrid) by body type, keepership and local authority | Open Government Licence v3.0 | [download](https://assets.publishing.service.gov.uk/media/6aad561b086ce7c57b818707/veh0142.ods) |
| `dft_nts0901` | DfT | NTS0901 annual mileage of cars by fuel type, England | Open Government Licence v3.0 | [download](https://assets.publishing.service.gov.uk/media/6a9ed42af36e1f225ecb7eab/nts0901.ods) |
| `dft_nts0904` | DfT | NTS0904 annual mileage bands of cars, England | Open Government Licence v3.0 | [download](https://assets.publishing.service.gov.uk/media/6a9ed42a474b8101ece46448/nts0904.ods) |
| `ons_family_spending_wb1` | ONS | Family spending workbook 1, FYE 2025 (table A6 by gross income decile) | Open Government Licence v3.0 | [download](https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/personalandhouseholdfinances/expenditure/datasets/familyspendingworkbook1detailedexpenditureandtrends/fye2025/workbook1detailedexpenditureandtrends.xlsx) |
| `ons_ghg_intensity_by_industry` | ONS | Atmospheric emissions: greenhouse gas emissions intensity by industry (SIC sections), 1990-2024 | Open Government Licence v3.0 | [download](https://www.ons.gov.uk/file?uri=/economy/environmentalaccounts/datasets/ukenvironmentalaccountsatmosphericemissionsgreenhousegasemissionsintensitybyeconomicsectorunitedkingdom/current/04atmosphericemissionsghgintensity.xlsx) |
| `mcc_codes` | greggles/mcc-codes (GitHub) | ISO 18245 merchant category codes with descriptions | The Unlicense | [download](https://raw.githubusercontent.com/greggles/mcc-codes/main/mcc_codes.csv) |
| `bench_dodatathings` | DoDataThings (Hugging Face) | 68,000 synthetic US bank transaction descriptions in 17 categories | MIT | [download](https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2/resolve/main/transactions-synthetic.csv) |
| `ofgem_2025Q1` | Ofgem | Price cap announcement 2025Q1 | Public web page | [download](https://www.ofgem.gov.uk/news/changes-energy-price-cap-between-1-january-and-31-march-2025) |
| `ofgem_2025Q2` | Ofgem | Price cap announcement 2025Q2 | Public web page | [download](https://www.ofgem.gov.uk/news/changes-energy-price-cap-between-1-april-and-30-june-2025) |
| `ofgem_2025Q3` | Ofgem | Price cap announcement 2025Q3 | Public web page | [download](https://www.ofgem.gov.uk/news/changes-energy-price-cap-between-1-july-and-30-september-2025) |
| `ofgem_2025Q4` | Ofgem | Price cap announcement 2025Q4 | Public web page | [download](https://www.ofgem.gov.uk/news/changes-energy-price-cap-between-1-october-and-31-december-2025) |
| `ofgem_2026Q1` | Ofgem | Price cap announcement 2026Q1 | Public web page | [download](https://www.ofgem.gov.uk/news/changes-energy-price-cap-between-1-january-and-31-march-2026) |
| `ofgem_2026Q2` | Ofgem | Price cap announcement 2026Q2 | Public web page | [download](https://www.ofgem.gov.uk/news/changes-energy-price-cap-between-1-april-and-30-june-2026) |
| `ofgem_2026Q3` | Ofgem | Price cap announcement 2026Q3 | Public web page | [download](https://www.ofgem.gov.uk/news/changes-energy-price-cap-between-1-july-and-30-september-2026) |
| `ofgem_2026Q4` | Ofgem | Price cap announcement 2026Q4 | Public web page | [download](https://www.ofgem.gov.uk/news/changes-energy-price-cap-between-1-october-and-31-december-2026) |
| `pvgis_north_east` | European Commission JRC | PV yield at North East reference point | Free use (EU JRC PVGIS 5.3) | [download](https://re.jrc.ec.europa.eu/api/v5_3/PVcalc?lat=54.973&lon=-1.614&peakpower=1&loss=14&optimalangles=1&outputformat=json) |
| `pvgis_north_west` | European Commission JRC | PV yield at North West reference point | Free use (EU JRC PVGIS 5.3) | [download](https://re.jrc.ec.europa.eu/api/v5_3/PVcalc?lat=53.48&lon=-2.242&peakpower=1&loss=14&optimalangles=1&outputformat=json) |
| `pvgis_yorkshire_and_the_humber` | European Commission JRC | PV yield at Yorkshire and The Humber reference point | Free use (EU JRC PVGIS 5.3) | [download](https://re.jrc.ec.europa.eu/api/v5_3/PVcalc?lat=53.8&lon=-1.549&peakpower=1&loss=14&optimalangles=1&outputformat=json) |
| `pvgis_east_midlands` | European Commission JRC | PV yield at East Midlands reference point | Free use (EU JRC PVGIS 5.3) | [download](https://re.jrc.ec.europa.eu/api/v5_3/PVcalc?lat=52.955&lon=-1.149&peakpower=1&loss=14&optimalangles=1&outputformat=json) |
| `pvgis_west_midlands` | European Commission JRC | PV yield at West Midlands reference point | Free use (EU JRC PVGIS 5.3) | [download](https://re.jrc.ec.europa.eu/api/v5_3/PVcalc?lat=52.486&lon=-1.89&peakpower=1&loss=14&optimalangles=1&outputformat=json) |
| `pvgis_east` | European Commission JRC | PV yield at East reference point | Free use (EU JRC PVGIS 5.3) | [download](https://re.jrc.ec.europa.eu/api/v5_3/PVcalc?lat=52.205&lon=0.122&peakpower=1&loss=14&optimalangles=1&outputformat=json) |
| `pvgis_london` | European Commission JRC | PV yield at London reference point | Free use (EU JRC PVGIS 5.3) | [download](https://re.jrc.ec.europa.eu/api/v5_3/PVcalc?lat=51.507&lon=-0.128&peakpower=1&loss=14&optimalangles=1&outputformat=json) |
| `pvgis_south_east` | European Commission JRC | PV yield at South East reference point | Free use (EU JRC PVGIS 5.3) | [download](https://re.jrc.ec.europa.eu/api/v5_3/PVcalc?lat=51.236&lon=-0.57&peakpower=1&loss=14&optimalangles=1&outputformat=json) |
| `pvgis_south_west` | European Commission JRC | PV yield at South West reference point | Free use (EU JRC PVGIS 5.3) | [download](https://re.jrc.ec.europa.eu/api/v5_3/PVcalc?lat=50.9&lon=-3.2&peakpower=1&loss=14&optimalangles=1&outputformat=json) |
| `pvgis_wales` | European Commission JRC | PV yield at Wales reference point | Free use (EU JRC PVGIS 5.3) | [download](https://re.jrc.ec.europa.eu/api/v5_3/PVcalc?lat=51.62&lon=-3.4&peakpower=1&loss=14&optimalangles=1&outputformat=json) |

## Processed reference tables (`data/reference/`)

| File | Rows | Columns |
|---|---|---|
| `reference/la_profile.csv` | 318 | 85 |
| `reference/pvgis_region_yield.csv` | 10 | 7 |
| `reference/consumption_distribution_2024.csv` | 414 | 4 |
| `reference/nts_mileage_by_fuel.csv` | 144 | 3 |
| `reference/nts_mileage_bands.csv` | 336 | 4 |
| `reference/fuel_prices_weekly.csv` | 458 | 3 |
| `reference/ofgem_price_cap.csv` | 8 | 9 |
| `reference/family_spending_a6_fye2025.csv` | 196 | 12 |
| `reference/ghg_factors_2026.csv` | 2622 | 9 |
| `reference/ons_ghg_intensity_by_industry.csv` | 735 | 6 |
| `reference/mcc_codes.csv` | 981 | 6 |
| `reference/census_tenure_age_hrp.csv` | 2544 | 4 |
| `reference/census_tenure_activity_hrp.csv` | 8904 | 4 |
| `reference/house_prices_la_monthly.csv` | 63282 | 8 |
| `reference/boe_quoted_rates_monthly.csv` | 141 | 20 |
| `reference/mot_car_profiles.csv` | 96 | 5 |
| `reference/need_homes.parquet` | 4000000 | 14 |
| `benchmarks/dodatathings_us_bank_categories.csv` | 68000 | 2 |

## Downloaded but not used

- EPC register bulk download (`raw_cache/epc/`, 7.8 GB): kept for a planned check of the simulated EPC bands; the
  homes in the simulator come from DESNZ NEED 2025 instead.
- Czech PKDD'99 bank (Berka) and a public UK current-account statement (MoneyData): kept for planned real-data proofs
  of FR1 and FR2; not used in any result.
