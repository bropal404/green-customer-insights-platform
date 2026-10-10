"""Download (with a local cache) and process every public reference source.

Outputs go to data/reference/ as tidy CSV files plus manifest.json, which
records the source URL, licence, SHA-256 and row count of every file. The
simulator and the carbon engine read only these processed files, so a run on
Ada never needs the raw downloads.

Usage (from the project root):
    python scripts/build_reference.py            # download what is missing, then build
    python scripts/build_reference.py --offline  # build from the raw cache only
"""

from __future__ import annotations

import io
import json
import logging
import re
import time
import urllib.request
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from gcip.io.files import sha256_file, write_json
from gcip.io.ods import read_ods_table
from gcip.reference.sources import (
    BOE_QUOTED_RATES,
    CENSUS_EXTRA_TABLES,
    CENSUS_TABLES,
    MERGED_2023,
    OFGEM_QUARTERS,
    OFGEM_URL,
    PVGIS_URL,
    REGION_POINTS,
    SOURCES,
    Source,
)

log = logging.getLogger("gcip.reference")
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

USER_AGENT = "Mozilla/5.0 (gcip reference builder; research use)"
LA_CODE_PATTERN = r"^(E06|E07|E08|E09|W06)\d{6}$"


# --------------------------------------------------------------------------- download


def _download(url: str, target: Path, retries: int = 3, timeout: int = 300) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            tmp = target.with_suffix(target.suffix + ".part")
            tmp.write_bytes(payload)
            tmp.replace(target)
            return
        except Exception as error:  # network errors vary by platform
            last_error = error
            log.warning("download failed (%s/%s) %s: %s", attempt, retries, url, error)
            time.sleep(2 * attempt)
    raise RuntimeError(f"could not download {url}: {last_error}")


def fetch_sources(raw_dir: Path, offline: bool) -> list[dict]:
    """Make sure every source is in the raw cache. Returns manifest rows."""
    records = []
    for source in SOURCES:
        path = raw_dir / source.cache_path
        if not path.exists():
            if offline:
                raise FileNotFoundError(f"{path} missing and --offline was given")
            log.info("downloading %s", source.url)
            _download(source.url, path)
        records.append(_raw_record(source, path))

    for quarter, _start, _end, slug in OFGEM_QUARTERS:
        path = raw_dir / "ofgem" / f"cap_{slug}.html"
        url = OFGEM_URL.format(slug=slug)
        if not path.exists():
            if offline:
                raise FileNotFoundError(f"{path} missing and --offline was given")
            _download(url, path)
        records.append(
            _raw_record(Source(f"ofgem_{quarter}", url, f"ofgem/cap_{slug}.html", "Public web page",
                               "Ofgem", f"Price cap announcement {quarter}"), path)
        )

    for region, (lat, lon) in REGION_POINTS.items():
        slug = region.lower().replace(" ", "_")
        path = raw_dir / "pvgis" / f"{slug}.json"
        url = PVGIS_URL.format(lat=lat, lon=lon)
        if not path.exists():
            if offline:
                raise FileNotFoundError(f"{path} missing and --offline was given")
            _download(url, path)
            time.sleep(0.2)  # PVGIS allows 30 calls a second; stay far below
        records.append(
            _raw_record(Source(f"pvgis_{slug}", url, f"pvgis/{slug}.json", "Free use (EU JRC PVGIS 5.3)",
                               "European Commission JRC", f"PV yield at {region} reference point"), path)
        )
    return records


def _raw_record(source: Source, path: Path) -> dict:
    return {
        "key": source.key,
        "url": source.url,
        "publisher": source.publisher,
        "licence": source.licence,
        "description": source.description,
        "cache_path": source.cache_path,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


# --------------------------------------------------------------------------- helpers


def _clean_header(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _xlsx_table(path: Path, sheet: str, first_cell: str) -> pd.DataFrame:
    """Read a sheet whose header row starts with ``first_cell`` (prefix match)."""
    raw = pd.read_excel(path, sheet_name=sheet, header=None, dtype=object)
    starts = raw.iloc[:, 0].map(_clean_header)
    hits = raw.index[starts.str.startswith(first_cell)]
    if len(hits) == 0:
        raise ValueError(f"no header row starting {first_cell!r} in {path.name}:{sheet}")
    row = hits[0]
    frame = raw.iloc[row + 1 :].copy()
    frame.columns = [_clean_header(c) or f"col_{i}" for i, c in enumerate(raw.iloc[row])]
    return frame.reset_index(drop=True)


def _num(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(cleaned.where(~cleaned.isin(["..", "[x]", "[c]", "[u]", "[low]", "-", "nan", "None"])),
                         errors="coerce")


def _col_suffix(frame: pd.DataFrame, prefix: str, suffix: str) -> str:
    for column in frame.columns:
        if column.lower().startswith(prefix.lower()) and column.lower().endswith(suffix.lower()):
            return column
    raise KeyError(f"no column starting {prefix!r} and ending {suffix!r} in {list(frame.columns)[:30]}")


def _col(frame: pd.DataFrame, *needles: str) -> str:
    for column in frame.columns:
        if all(n.lower() in column.lower() for n in needles):
            return column
    raise KeyError(f"no column containing {needles} in {list(frame.columns)[:30]}")


# --------------------------------------------------------------------------- census


_CENSUS_COLUMNS = {
    "ts054": {
        "households": "Total: All households",
        "tenure_owned_outright": "Owned: Owns outright",
        "tenure_mortgage": "Owned: Owns with a mortgage or loan",
        "tenure_shared_ownership": "Shared ownership: Shared ownership",
        "tenure_social_rent_council": "Social rented: Rents from council or Local Authority",
        "tenure_social_rent_other": "Social rented: Other social rented",
        "tenure_private_landlord": "Private rented: Private landlord or letting agency",
        "tenure_private_other": "Private rented: Other private rented",
        "tenure_rent_free": "Lives rent free",
    },
    "ts044": {
        "acc_total": "Total: All households",
        "acc_detached": "Detached",
        "acc_semi": "Semi-detached",
        "acc_terraced": "Terraced",
        "acc_flat_purpose_built": "In a purpose-built block of flats or tenement",
        "acc_flat_converted": "Part of a converted or shared house",
        "acc_flat_other_building": "Part of another converted building",
        "acc_flat_commercial": "In a commercial building",
        "acc_caravan": "A caravan or other mobile or temporary structure",
    },
    "ts045": {
        "cars_total": "Total: All households",
        "cars_0": "No cars or vans in household",
        "cars_1": "1 car or van in household",
        "cars_2": "2 cars or vans in household",
        "cars_3plus": "3 or more cars or vans in household",
    },
    "ts061": {
        "commute_total": "Total: All usual residents aged 16 years and over in employment",
        "commute_wfh": "Work mainly at or from home",
        "commute_metro_tram": "Underground, metro, light rail, tram",
        "commute_train": "Train",
        "commute_bus": "Bus, minibus or coach",
        "commute_taxi": "Taxi",
        "commute_motorcycle": "Motorcycle, scooter or moped",
        "commute_car_driver": "Driving a car or van",
        "commute_car_passenger": "Passenger in a car or van",
        "commute_bicycle": "Bicycle",
        "commute_foot": "On foot",
        "commute_other": "Other method of travel to work",
    },
    "ts046": {
        "heat_total": "Total: All households",
        "heat_none": "No central heating",
        "heat_mains_gas": "Mains gas only",
        "heat_lpg": "Tank or bottled gas only",
        "heat_electric": "Electric only",
        "heat_oil": "Oil only",
        "heat_wood": "Wood only",
        "heat_solid": "Solid fuel only",
        "heat_renewable": "Renewable energy only",
        "heat_district": "District or communal heat networks only",
        "heat_other": "Other central heating only",
        "heat_multi_no_renewable": "Two or more types of central heating (not including renewable energy)",
        "heat_multi_renewable": "Two or more types of central heating (including renewable energy)",
    },
    "ts017": {
        "hh_size_1": "1 person in household",
        "hh_size_2": "2 people in household",
        "hh_size_3": "3 people in household",
        "hh_size_4": "4 people in household",
        "hh_size_5": "5 people in household",
        "hh_size_6": "6 people in household",
        "hh_size_7": "7 people in household",
        "hh_size_8plus": "8 or more people in household",
    },
}


def build_census(raw_dir: Path) -> pd.DataFrame:
    """Counts per 2021 lower-tier local authority for the six Census tables."""
    merged = None
    for table in CENSUS_TABLES:
        with zipfile.ZipFile(raw_dir / f"census/census2021-{table}.zip") as archive:
            frame = pd.read_csv(archive.open(f"census2021-{table}-ltla.csv"))
        out = frame[["geography code", "geography"]].rename(columns={"geography code": "la_code_2021",
                                                                     "geography": "la_name_2021"})
        for short, needle in _CENSUS_COLUMNS[table].items():
            matches = [c for c in frame.columns if needle in c]
            if len(matches) != 1:
                raise KeyError(f"{table}: expected one column containing {needle!r}, got {matches}")
            out[short] = frame[matches[0]].astype(float)
        merged = out if merged is None else merged.merge(out, on=["la_code_2021", "la_name_2021"], how="inner")
    assert merged is not None and len(merged) == 331, f"expected 331 Census LTLAs, got {len(merged)}"
    return merged


def census_to_2023(census: pd.DataFrame, la_names_2023: dict[str, str]) -> pd.DataFrame:
    """Aggregate 2021 districts into the unitary authorities created in April 2023."""
    frame = census.copy()
    frame["la_name"] = frame["la_name_2021"].map(lambda n: MERGED_2023.get(n, n))
    name_to_code = {name: code for code, name in la_names_2023.items()}
    frame["la_code"] = np.where(
        frame["la_name_2021"].isin(MERGED_2023.keys()),
        frame["la_name"].map(name_to_code),
        frame["la_code_2021"],
    )
    if frame["la_code"].isna().any():
        missing = frame.loc[frame["la_code"].isna(), "la_name"].unique()
        raise KeyError(f"no 2023 code found for merged authorities {missing}")
    # Boundary changes also re-coded some unmerged authorities (Barnsley and Sheffield
    # became E08000038/E08000039 in 2023): re-key any stale code by its name.
    stale = ~frame["la_code"].isin(la_names_2023.keys())
    frame.loc[stale, "la_code"] = frame.loc[stale, "la_name"].map(name_to_code).fillna(frame.loc[stale, "la_code"])
    count_cols = [c for c in frame.columns if c not in ("la_code_2021", "la_name_2021", "la_name", "la_code")]
    grouped = frame.groupby(["la_code", "la_name"], as_index=False)[count_cols].sum()
    return grouped


# --------------------------------------------------------------------------- DESNZ energy


def build_electricity_la(raw_dir: Path) -> pd.DataFrame:
    frame = _xlsx_table(raw_dir / "desnz/elec_la.xlsx", "2024", "Code")
    out = pd.DataFrame({
        "la_code": frame["Code"].astype(str).str.strip(),
        "region": frame["Country or region"].astype(str).str.strip(),
        "la_name_desnz": frame["Local authority"].astype(str).str.strip(),
        "elec_dom_meters": _num(frame[_col(frame, "Number of meters", "All Domestic")]) * 1000,
        "elec_e7_meters": _num(frame[_col(frame, "Number of meters", "Domestic E7")]) * 1000,
        "elec_dom_mean_kwh": _num(frame[_col(frame, "Mean consumption", "All Domestic")]),
        "elec_dom_median_kwh": _num(frame[_col(frame, "Median consumption", "All Domestic")]),
    })
    out = out[out["la_code"].str.match(LA_CODE_PATTERN)].copy()
    out["elec_e7_share"] = out["elec_e7_meters"] / out["elec_dom_meters"]
    return out.drop(columns=["elec_e7_meters"])


def build_gas_la(raw_dir: Path) -> pd.DataFrame:
    frame = _xlsx_table(raw_dir / "desnz/gas_la.xlsx", "2024", "Code")
    out = pd.DataFrame({
        "la_code": frame["Code"].astype(str).str.strip(),
        "gas_dom_meters": _num(frame[_col_suffix(frame, "Number of meters", ": Domestic")]) * 1000,
        "gas_dom_mean_kwh": _num(frame[_col_suffix(frame, "Mean consumption", ": Domestic")]),
        "gas_dom_median_kwh": _num(frame[_col_suffix(frame, "Median consumption", ": Domestic")]),
    })
    return out[out["la_code"].str.match(LA_CODE_PATTERN)].copy()


def build_consumption_distribution(raw_dir: Path) -> pd.DataFrame:
    parts = []
    for fuel, name in (("electricity", "elec_dist.xlsx"), ("gas", "gas_dist.xlsx")):
        frame = _xlsx_table(raw_dir / "desnz" / name, "2024", "Lower consumption band")
        part = pd.DataFrame({
            "fuel": fuel,
            "lower_kwh": _num(frame.iloc[:, 0]),
            "upper_kwh": _num(frame.iloc[:, 1]),
            "meters": _num(frame.iloc[:, 2]),
        }).dropna(subset=["lower_kwh", "meters"])
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def build_renewables_la(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "desnz/renew_la.xlsx"
    sites = _xlsx_table(path, "LA - Sites 2025", "Local Authority Code")
    capacity = _xlsx_table(path, "LA - Capacity, 2025", "Local Authority Code")
    out = pd.DataFrame({
        "la_code": sites.iloc[:, 0].astype(str).str.strip(),
        "households_desnz_2025": _num(sites[_col(sites, "Estimated number of households")]),
        "pv_sites_2025": _num(sites["Solar PV"]),
    })
    cap = pd.DataFrame({
        "la_code": capacity.iloc[:, 0].astype(str).str.strip(),
        "pv_capacity_mw_2025": _num(capacity["Solar PV"]),
    })
    out = out.merge(cap, on="la_code", how="left")
    return out[out["la_code"].str.match(LA_CODE_PATTERN)].copy()


# --------------------------------------------------------------------------- DfT vehicles


def build_vehicles_la(raw_dir: Path) -> pd.DataFrame:
    """Private cars by fuel per local authority at 2026 Q2.

    VEH0105 splits cars into petrol, diesel, petrol hybrid and 'other fuel types';
    VEH0142 gives the plug-in cars (battery electric, plug-in hybrid) inside 'other'.
    """
    quarter = "2026 Q2"
    veh = read_ods_table(raw_dir / "dft/veh0105.ods", "VEH0105", header_row_contains="Units")
    veh = veh[(veh["BodyType"] == "Cars") & (veh["Keepership"] == "Private")]
    veh = veh[veh["ONS Code"].astype(str).str.match(LA_CODE_PATTERN)].copy()
    veh[quarter] = _num(veh[quarter])
    wide = veh.pivot_table(index="ONS Code", columns="Fuel", values=quarter, aggfunc="sum") * 1000.0

    plug = read_ods_table(raw_dir / "dft/veh0142.ods", "VEH0142", header_row_contains="Units")
    plug = plug[(plug["BodyType"] == "Cars") & (plug["Keepership"] == "Private")]
    plug = plug[plug["ONS Code"].astype(str).str.match(LA_CODE_PATTERN)].copy()
    plug[quarter] = _num(plug[quarter])
    plug["kind"] = np.select(
        [plug["Fuel"].str.contains("Battery electric|Range extended", case=False, regex=True),
         plug["Fuel"].str.contains("Plug-in hybrid", case=False, regex=True)],
        ["bev", "phev"], default="skip",
    )
    plug = plug[plug["kind"] != "skip"]
    plug_wide = plug.pivot_table(index="ONS Code", columns="kind", values=quarter, aggfunc="sum")

    out = pd.DataFrame(index=wide.index)
    out["cars_private_total"] = wide["Total"]
    out["cars_petrol"] = wide["Petrol"]
    out["cars_diesel"] = wide["Diesel"]
    out["cars_hev"] = wide["Hybrid electric (petrol)"]
    out = out.join(plug_wide.rename(columns={"bev": "cars_bev", "phev": "cars_phev"}), how="left").fillna(0.0)
    out["cars_other"] = (wide["Other fuel types"] - out["cars_bev"] - out["cars_phev"]).clip(lower=0.0)
    for kind in ("petrol", "diesel", "hev", "phev", "bev", "other"):
        out[f"car_share_{kind}"] = out[f"cars_{kind}"] / out["cars_private_total"]
    out.index.name = "la_code"
    return out.reset_index()


def build_nts(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    fuel = read_ods_table(raw_dir / "nts/nts0901.ods", "NTS0901a_fuel_type_mileage", header_row_contains="Year")
    fuel = fuel.rename(columns=lambda c: re.sub(r"\s*\[.*?\]", "", c).strip())
    fuel_tidy = fuel.melt(id_vars="Year", var_name="fuel", value_name="miles_per_year")
    fuel_tidy = fuel_tidy[~fuel_tidy["fuel"].str.startswith("Unweighted")]
    fuel_tidy["year"] = _num(fuel_tidy["Year"]).astype("Int64")
    fuel_tidy["miles_per_year"] = _num(fuel_tidy["miles_per_year"])
    fuel_tidy = fuel_tidy.dropna(subset=["year"])[["year", "fuel", "miles_per_year"]]

    bands = read_ods_table(raw_dir / "nts/nts0904.ods", "NTS0904", header_row_contains="Year")
    rows = []
    for column in bands.columns:
        match = re.match(r"([\d,]+) (?:to ([\d,]+) )?(?:and over )?miles", column)
        if not match:
            continue
        low = float(match.group(1).replace(",", ""))
        high = float(match.group(2).replace(",", "")) + 1 if match.group(2) else np.nan
        for year, share in zip(_num(bands["Year"]), _num(bands[column])):
            if pd.notna(year):
                rows.append({"year": int(year), "lower_miles": low, "upper_miles": high, "share_pct": share})
    return fuel_tidy.reset_index(drop=True), pd.DataFrame(rows)


# --------------------------------------------------------------------------- prices


def build_fuel_prices(raw_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(raw_dir / "desnz/weekly_fuel_prices_2018_2026.csv", encoding="utf-8-sig")
    out = pd.DataFrame({
        "week_start": pd.to_datetime(frame.iloc[:, 0], dayfirst=True),
        "petrol_ppl": pd.to_numeric(frame[_col(frame, "ULSP", "Pump price")], errors="coerce"),
        "diesel_ppl": pd.to_numeric(frame[_col(frame, "ULSD", "Pump price")], errors="coerce"),
    }).dropna()
    return out.sort_values("week_start").reset_index(drop=True)


def build_ofgem(raw_dir: Path) -> pd.DataFrame:
    from lxml import html as lxml_html

    rows = []
    for quarter, start, end, slug in OFGEM_QUARTERS:
        document = lxml_html.parse(str(raw_dir / "ofgem" / f"cap_{slug}.html")).getroot()
        for node in document.xpath("//script|//style"):
            node.drop_tree()
        text = re.sub(r"\s+", " ", document.text_content())
        values = re.findall(r"(\d{1,3}\.\d{1,2})\s*(?:pence|p)\s*per\s*(kilowatt hour|kWh|day)", text, flags=re.I)
        if len(values) < 4:
            raise ValueError(f"could not read the four rates for {quarter} from Ofgem's page")
        (elec_unit, u1), (elec_sc, u2), (gas_unit, u3), (gas_sc, u4) = values[:4]
        if not (u1.lower() != "day" and u2.lower() == "day" and u3.lower() != "day" and u4.lower() == "day"):
            raise ValueError(f"unexpected rate order on Ofgem page for {quarter}: {values[:4]}")
        rows.append({
            "quarter": quarter, "start": start, "end": end,
            "elec_unit_p_per_kwh": float(elec_unit), "elec_standing_p_per_day": float(elec_sc),
            "gas_unit_p_per_kwh": float(gas_unit), "gas_standing_p_per_day": float(gas_sc),
            "electricity_vat_pct": 0.0 if quarter >= "2026Q4" else 5.0,
            "source_url": OFGEM_URL.format(slug=slug),
        })
    return pd.DataFrame(rows)


def build_pvgis(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for region, (lat, lon) in REGION_POINTS.items():
        slug = region.lower().replace(" ", "_")
        payload = json.loads((raw_dir / "pvgis" / f"{slug}.json").read_text())
        mounting = payload["inputs"]["mounting_system"]["fixed"]
        rows.append({
            "region": region, "lat": lat, "lon": lon,
            "pv_kwh_per_kwp_year": payload["outputs"]["totals"]["fixed"]["E_y"],
            "optimal_slope_deg": mounting["slope"]["value"],
            "optimal_azimuth_deg": mounting["azimuth"]["value"],
            "radiation_db": payload["inputs"]["meteo_data"].get("radiation_db"),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- ONS, DESNZ factors, MCC


def build_family_spending(raw_dir: Path) -> pd.DataFrame:
    raw = pd.read_excel(raw_dir / "ons/fs_wb1_fye2025.xlsx", sheet_name="A6", header=None, dtype=object)
    decile_cols = list(range(4, 14))
    rows = []
    for _, row in raw.iterrows():
        first = _clean_header(row.iloc[0])
        code, label = None, None
        for position in range(0, 3):
            cell = _clean_header(row.iloc[position])
            if re.fullmatch(r"\d+(\.\d+)*", cell):
                code = cell
                label = next((_clean_header(row.iloc[j]) for j in range(position + 1, 4) if _clean_header(row.iloc[j])), "")
                break
        special = {
            "Lower boundary of group": "income_lower_boundary_gbp_week",
            "Mean of group": "income_mean_gbp_week",
            "Median of group": "income_median_gbp_week",
            "Weighted number of households": "weighted_households_thousands",
            "Total expenditure": "total_expenditure_gbp_week",
        }
        for prefix, key in special.items():
            if first.startswith(prefix):
                code, label = key, first
        if code is None:
            continue
        values = _num(pd.Series([row.iloc[c] for c in decile_cols]))
        record = {"code": code, "label": label}
        record.update({f"decile_{i + 1}": values.iloc[i] for i in range(10)})
        rows.append(record)
    frame = pd.DataFrame(rows).drop_duplicates(subset=["code"], keep="first")
    return frame.reset_index(drop=True)


def build_ghg_factors(raw_dir: Path) -> pd.DataFrame:
    frame = _xlsx_table(raw_dir / "desnz/ghg_factors_2026_flat.xlsx", "Factors by Category", "ID")
    frame = frame.rename(columns={
        "ID": "factor_id", "Scope": "scope", "Level 1": "level_1", "Level 2": "level_2", "Level 3": "level_3",
        "Level 4": "level_4", "Column Text": "column_text", "UOM": "unit", "GHG/Unit": "ghg_unit",
        "GHG Conversion Factor 2026": "kg_co2e",
    })
    frame = frame[frame["ghg_unit"] == "kg CO2e"].copy()
    frame["kg_co2e"] = _num(frame["kg_co2e"])
    keep = ["factor_id", "scope", "level_1", "level_2", "level_3", "level_4", "column_text", "unit", "kg_co2e"]
    return frame[keep].dropna(subset=["kg_co2e"]).reset_index(drop=True)


def build_ghg_intensity(raw_dir: Path) -> pd.DataFrame:
    """ONS GHG emissions intensity by SIC section (thousand tonnes CO2e per GBP million GVA)."""
    raw = pd.read_excel(raw_dir / "ons/ghg_intensity_by_industry.xlsx", sheet_name="GHG intensity", header=None, dtype=object)
    header_row = raw.index[raw.iloc[:, 0].map(_clean_header) == "SIC code"][0]
    title = _clean_header(raw.iloc[header_row - 1, 0])
    codes = [_clean_header(c) for c in raw.iloc[header_row]]
    names = [_clean_header(c) for c in raw.iloc[header_row + 1]]
    rows = []
    for r in range(header_row + 2, len(raw)):
        year = pd.to_numeric(raw.iloc[r, 0], errors="coerce")
        if pd.isna(year):
            break
        for c in range(1, len(codes)):
            if not names[c]:
                break
            value = raw.iloc[r, c]
            numeric = 0.004 if _clean_header(value) == "[low]" else pd.to_numeric(value, errors="coerce")
            rows.append({"year": int(year), "sic_section": codes[c] or "TOTAL", "industry": names[c], "intensity": numeric,
                         "units": "thousand tonnes CO2e per GBP million GVA", "table_title": title})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- Census: further topics and cross-tables

_TOTAL = "Total: All households"
_CENSUS_EXTRA_COLUMNS = {   # short name -> exact column headers in the district file (summed)
    "ts003": {
        "hc_total": ["Household composition: Total; measures: Value"],
        "hc_one_person": ["Household composition: One person household; measures: Value"],
        "hc_one_person_66plus": ["Household composition: One person household: Aged 66 years and over; measures: Value"],
        "hc_family_66plus": ["Household composition: Single family household: All aged 66 years and over; measures: Value"],
        "hc_couple_no_children": [
            "Household composition: Single family household: Married or civil partnership couple: No children; measures: Value",
            "Household composition: Single family household: Cohabiting couple family: No children; measures: Value"],
        "hc_couple_dependent": [
            "Household composition: Single family household: Married or civil partnership couple: Dependent children; measures: Value",
            "Household composition: Single family household: Cohabiting couple family: With dependent children; measures: Value"],
        "hc_couple_nondependent": [
            "Household composition: Single family household: Married or civil partnership couple: All children non-dependent; measures: Value",
            "Household composition: Single family household: Cohabiting couple family: All children non-dependent; measures: Value"],
        "hc_lone_dependent": ["Household composition: Single family household: Lone parent family: With dependent children; measures: Value"],
        "hc_lone_nondependent": ["Household composition: Single family household: Lone parent family: All children non-dependent; measures: Value"],
        "hc_other_dependent": ["Household composition: Other household types: With dependent children; measures: Value"],
    },
    "ts050": {
        "bed_total": ["Number of bedrooms: Total: All households"],
        "bed_1": ["Number of bedrooms: 1 bedroom"],
        "bed_2": ["Number of bedrooms: 2 bedrooms"],
        "bed_3": ["Number of bedrooms: 3 bedrooms"],
        "bed_4plus": ["Number of bedrooms: 4 or more bedrooms"],
    },
    "ts066": {
        "ea_employee": ["Economic activity status: Economically active (excluding full-time students):In employment:Employee"],
        "ea_self_employed": [
            "Economic activity status: Economically active (excluding full-time students):In employment:Self-employed with employees",
            "Economic activity status: Economically active (excluding full-time students):In employment:Self-employed without employees"],
    },
    "ts062": {
        "nssec_total": ["National Statistics Socio-economic Classification (NS-SEC): Total: All usual residents aged 16 years and over"],
        "nssec_higher": ["National Statistics Socio-economic Classification (NS-SEC): L1, L2 and L3 Higher managerial, administrative and professional occupations"],
        "nssec_routine": ["National Statistics Socio-economic Classification (NS-SEC): L12 Semi-routine occupations",
                          "National Statistics Socio-economic Classification (NS-SEC): L13 Routine occupations"],
    },
    "ts058": {
        "dist_total": ["Distance travelled to work: Total: All usual residents aged 16 years and over in employment the week before the census"],
        "dist_under_5km": ["Distance travelled to work: Less than 2km", "Distance travelled to work: 2km to less than 5km"],
        "dist_20km_plus": ["Distance travelled to work: 20km to less than 30km", "Distance travelled to work: 30km to less than 40km",
                           "Distance travelled to work: 40km to less than 60km", "Distance travelled to work: 60km and over"],
        "dist_home": ["Distance travelled to work: Works mainly from home"],
    },
    "ts011": {
        "dep_total": ["Household deprivation: Total: All households; measures: Value"],
        "dep_2plus": ["Household deprivation: Household is deprived in two dimensions; measures: Value",
                      "Household deprivation: Household is deprived in three dimensions; measures: Value",
                      "Household deprivation: Household is deprived in four dimensions; measures: Value"],
    },
    "ts037": {
        "health_total": ["General health: Total: All usual residents"],
        "health_bad": ["General health: Bad health", "General health: Very bad health"],
    },
    "ts038": {
        "dis_total": ["Disability: Total: All usual residents"],
        "dis_disabled": ["Disability: Disabled under the Equality Act"],
    },
    "ts063": {
        "occ_total": ["Occupation (current): Total: All usual residents aged 16 years and over in employment the week before the census"],
        **{f"occ_{i}": [f"Occupation (current): {i}. {name}"] for i, name in enumerate([
            "Managers, directors and senior officials", "Professional occupations", "Associate professional and technical occupations",
            "Administrative and secretarial occupations", "Skilled trades occupations", "Caring, leisure and other service occupations",
            "Sales and customer service occupations", "Process, plant and machine operatives", "Elementary occupations"], start=1)},
    },
    "ts052": {
        "occupancy_total": ["Occupancy rating for bedrooms: Total: All households"],
        "overcrowded": ["Occupancy rating for bedrooms: Occupancy rating of bedrooms: -1",
                        "Occupancy rating for bedrooms: Occupancy rating of bedrooms: -2 or less"],
    },
}
SOC_MAJOR_GROUPS = {1: "managers_directors", 2: "professional", 3: "associate_professional", 4: "administrative",
                    5: "skilled_trades", 6: "caring_leisure", 7: "sales_customer_service", 8: "process_plant", 9: "elementary"}


def build_census_extra(raw_dir: Path) -> pd.DataFrame:
    """Counts per 2021 district for the further Census topics (exact headers, summed where listed)."""
    merged = None
    for table, columns in _CENSUS_EXTRA_COLUMNS.items():
        with zipfile.ZipFile(raw_dir / f"census/census2021-{table}.zip") as archive:
            frame = pd.read_csv(archive.open(f"census2021-{table}-ltla.csv"))
        out = frame[["geography code", "geography"]].rename(columns={"geography code": "la_code_2021",
                                                                     "geography": "la_name_2021"})
        for short, headers in columns.items():
            missing = [h for h in headers if h not in frame.columns]
            if missing:
                raise KeyError(f"{table}: columns not found {missing}")
            out[short] = frame[headers].astype(float).sum(axis=1)
        merged = out if merged is None else merged.merge(out, on=["la_code_2021", "la_name_2021"], how="inner")
    assert merged is not None and len(merged) == 331, f"expected 331 Census LTLAs, got {len(merged)}"
    return merged


def census_extra_shares(c: pd.DataFrame) -> pd.DataFrame:
    """Shares per 2023 authority: households, bedrooms, workers, residents, as each table's base."""
    out = pd.DataFrame({"la_code": c["la_code"]})
    hh = c["hc_total"]
    out["share_hh_one_person"] = c["hc_one_person"] / hh
    out["share_hh_all_66plus"] = (c["hc_one_person_66plus"] + c["hc_family_66plus"]) / hh
    out["share_hh_couple_no_children"] = c["hc_couple_no_children"] / hh
    out["share_hh_dependent_children"] = (c["hc_couple_dependent"] + c["hc_lone_dependent"] + c["hc_other_dependent"]) / hh
    out["share_hh_lone_parent_dependent"] = c["hc_lone_dependent"] / hh
    out["share_hh_nondependent_children"] = (c["hc_couple_nondependent"] + c["hc_lone_nondependent"]) / hh
    for k in ("1", "2", "3", "4plus"):
        out[f"share_bedrooms_{k}"] = c[f"bed_{k}"] / c["bed_total"]
    out["self_employed_share_of_workers"] = c["ea_self_employed"] / (c["ea_employee"] + c["ea_self_employed"])
    out["share_nssec_higher_managerial"] = c["nssec_higher"] / c["nssec_total"]
    out["share_nssec_routine"] = c["nssec_routine"] / c["nssec_total"]
    out["share_commute_under_5km"] = c["dist_under_5km"] / c["dist_total"]
    out["share_commute_20km_plus"] = c["dist_20km_plus"] / c["dist_total"]
    out["share_hh_deprived_2plus_dims"] = c["dep_2plus"] / c["dep_total"]
    out["share_bad_health"] = c["health_bad"] / c["health_total"]
    out["share_disabled"] = c["dis_disabled"] / c["dis_total"]
    for i, name in SOC_MAJOR_GROUPS.items():
        out[f"share_occupation_{name}"] = c[f"occ_{i}"] / c["occ_total"]
    out["share_hh_overcrowded"] = c["overcrowded"] / c["occupancy_total"]
    return out


RM_TENURE = {"Owned: Owns outright": "owned_outright",
             "Owned: Owns with a mortgage or loan or shared ownership": "mortgage",
             "Rented: Social rented": "social_rent",
             "Rented: Private rented or lives rent free": "private_rent"}
RM_ACTIVITY = {1: "employed", 3: "employed", 2: "unemployed", 4: "unemployed", 5: "retired", 6: "student",
               7: "home_family", 8: "sick_disabled", 9: "other_inactive"}


def build_census_rm(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """RM201 (tenure x age) and RM133 (tenure x economic activity) of household reference persons, per 2023 authority."""
    age = pd.read_csv(raw_dir / "census_rm/rm201_tenure_by_age_hrp.csv")
    age = age[age["C2021_AGE_4B_NAME"] != "Aged 15 years and under"]
    age = pd.DataFrame({
        "la_code": age["GEOGRAPHY_CODE"],
        "tenure": age["C2021_HHTENURE_5_NAME"].map(RM_TENURE),
        "hrp_age": age["C2021_AGE_4B_NAME"].map({"Aged 16 to 64 years": "16-64", "Aged 65 years and over": "65+"}),
        "households": age["OBS_VALUE"].astype(float),
    })
    act = pd.read_csv(raw_dir / "census_rm/rm133_tenure_by_activity_hrp.csv")
    act = act[act["C2021_EASTAT_10"].isin(RM_ACTIVITY)]
    act = pd.DataFrame({
        "la_code": act["GEOGRAPHY_CODE"],
        "tenure": act["C2021_HHTENURE_5_NAME"].map(RM_TENURE),
        "hrp_activity": act["C2021_EASTAT_10"].map(RM_ACTIVITY),
        "households": act["OBS_VALUE"].astype(float),
    }).groupby(["la_code", "tenure", "hrp_activity"], as_index=False)["households"].sum()
    for frame, name in ((age, "RM201"), (act, "RM133")):
        if frame.isna().any().any():
            raise KeyError(f"{name}: unmapped categories")
    return age, act


# --------------------------------------------------------------------------- NEED, house prices, Bank of England rates

NEED_REGIONS = {"E12000001": "North East", "E12000002": "North West", "E12000003": "Yorkshire and The Humber",
                "E12000004": "East Midlands", "E12000005": "West Midlands", "E12000006": "East", "E12000007": "London",
                "E12000008": "South East", "E12000009": "South West", "W92000004": "Wales"}


def build_need(raw_dir: Path) -> pd.DataFrame:
    """NEED 2025 sample of 4 million homes: attributes plus 2023 metered use where the reading is valid."""
    columns = ["PROP_TYPE", "PROP_AGE_BAND", "FLOOR_AREA_BAND", "CONSERVATORY_FLAG", "COUNCIL_TAX_BAND", "IMD_BAND_ENG",
               "IMD_BAND_WALES", "REGION", "EPC", "LI_FLAG", "CWI_FLAG", "PV_FLAG", "MAIN_HEAT_FUEL", "Gcons2023",
               "GasValFlag2023", "Econs2023", "ElecValFlag2023"]
    with zipfile.ZipFile(raw_dir / "need/anon2025_4million.zip") as archive:
        frame = pd.read_csv(archive.open("anon2025_4million.csv"), usecols=columns,
                            dtype={"PROP_TYPE": "category", "COUNCIL_TAX_BAND": "category", "REGION": "category",
                                   "EPC": "category", "GasValFlag2023": "category", "ElecValFlag2023": "category"})
    out = pd.DataFrame({
        "region": frame["REGION"].astype(str).map(NEED_REGIONS),
        "prop_type": frame["PROP_TYPE"].astype(str),
        "age_band": frame["PROP_AGE_BAND"].astype("int8"),
        "floor_area_band": frame["FLOOR_AREA_BAND"].astype("int8"),
        "conservatory": frame["CONSERVATORY_FLAG"].astype("float32"),
        "council_tax_band": frame["COUNCIL_TAX_BAND"].astype(str),
        "imd_quintile": frame["IMD_BAND_ENG"].fillna(frame["IMD_BAND_WALES"]).astype("float32"),
        "epc": frame["EPC"].astype(str),
        "loft_insulation_scheme": frame["LI_FLAG"].astype(bool),
        "cavity_insulation_scheme": frame["CWI_FLAG"].astype(bool),
        "solar_pv": frame["PV_FLAG"].astype(bool),
        "gas_heated": frame["MAIN_HEAT_FUEL"] == 1,
        "gas_kwh_2023": np.where(frame["GasValFlag2023"].astype(str) == "V", frame["Gcons2023"], np.nan).astype("float32"),
        "elec_kwh_2023": np.where(frame["ElecValFlag2023"].astype(str) == "V", frame["Econs2023"], np.nan).astype("float32"),
    })
    if out["region"].isna().any():
        raise KeyError("NEED: unknown region codes")
    return out


def build_house_prices(raw_dir: Path, la_codes: list[str]) -> pd.DataFrame:
    """Monthly average prices by local authority and property type since 2010 (UK HPI)."""
    hpi = pd.read_csv(raw_dir / "hpi/UK-HPI-full-file-2026-07.csv",
                      usecols=["Date", "RegionName", "AreaCode", "AveragePrice", "DetachedPrice", "SemiDetachedPrice",
                               "TerracedPrice", "FlatPrice"])
    hpi["month"] = pd.to_datetime(hpi["Date"], format="%d/%m/%Y")
    hpi = hpi[hpi["AreaCode"].isin(la_codes) & (hpi["month"] >= "2010-01-01")]
    out = pd.DataFrame({
        "month": hpi["month"].dt.strftime("%Y-%m-%d"), "la_code": hpi["AreaCode"], "la_name_hmlr": hpi["RegionName"],
        "avg_price": hpi["AveragePrice"], "price_detached": hpi["DetachedPrice"], "price_semi": hpi["SemiDetachedPrice"],
        "price_terraced": hpi["TerracedPrice"], "price_flat": hpi["FlatPrice"],
    })
    fallbacks = {"E06000053": "E06000052"}  # Isles of Scilly has no HPI series: use Cornwall
    for code in sorted(set(la_codes) - set(out["la_code"])):
        if code not in fallbacks:
            raise KeyError(f"no house price series for {code}")
        out = pd.concat([out, out[out["la_code"] == fallbacks[code]].assign(la_code=code)], ignore_index=True)
    return out.sort_values(["la_code", "month"]).reset_index(drop=True)


def build_boe_rates(raw_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(raw_dir / "boe/quoted_household_rates.csv")
    frame["month"] = pd.to_datetime(frame["DATE"], format="%d %b %Y").dt.to_period("M").dt.to_timestamp().dt.strftime("%Y-%m-%d")
    missing = [code for code in BOE_QUOTED_RATES if code not in frame.columns]
    if missing:
        raise KeyError(f"Bank of England series missing: {missing}")
    return frame.rename(columns=BOE_QUOTED_RATES)[["month"] + list(BOE_QUOTED_RATES.values())]





# --------------------------------------------------------------------------- DVSA MOT results

MOT_FUEL = {"PE": "petrol", "DI": "diesel", "EL": "bev", "HY": "hybrid", "ED": "hybrid"}
MOT_MONTHS = ("202401", "202404", "202407", "202410")  # one month per quarter: seasonal spread at a quarter of the size


def build_mot_profiles(raw_dir: Path, months: tuple[str, ...] = MOT_MONTHS) -> pd.DataFrame:
    """Cars at their MOT in 2024 by fuel and age: counts and lifetime-average annual mileage (odometer / age).

    Class 4 normal tests only, one row per test; ages under 2.5 years are dropped (first MOT is at 3 years).
    """
    columns = ["test_id", "test_date", "test_class_id", "test_type", "test_mileage", "fuel_type", "first_use_date"]
    parts = []
    with zipfile.ZipFile(raw_dir / "mot/mot_results_2024.zip") as archive:
        names = [n for n in archive.namelist() if not n.startswith("__MACOSX") and n.endswith(".csv")
                 and any(n.endswith(f"test_result_{m}.csv") for m in months)]
        for name in sorted(names):
            for chunk in pd.read_csv(archive.open(name), usecols=columns, chunksize=2_000_000,
                                     dtype={"test_id": "string", "test_class_id": "string", "test_type": "string", "fuel_type": "string",
                                            "test_date": "string", "first_use_date": "string", "test_mileage": "string"}):
                # The monthly files repeat their header row part-way through; those rows fail every filter below.
                chunk = chunk[(chunk["test_class_id"] == "4") & (chunk["test_type"] == "NT") & chunk["fuel_type"].isin(list(MOT_FUEL))]
                chunk = chunk.drop_duplicates("test_id")
                test = pd.to_datetime(chunk["test_date"], errors="coerce")
                first = pd.to_datetime(chunk["first_use_date"], errors="coerce")
                age = (test - first).dt.days / 365.25
                miles = pd.to_numeric(chunk["test_mileage"], errors="coerce")
                ok = age.between(2.5, 40) & miles.between(100, 600_000)
                frame = pd.DataFrame({"fuel": chunk["fuel_type"].map(MOT_FUEL)[ok], "age_year": np.floor(age[ok]).astype(int).clip(upper=25),
                                      "annual_miles": (miles[ok] / age[ok]).clip(upper=60_000)})
                parts.append(frame.groupby(["fuel", "age_year"]).agg(cars=("annual_miles", "size"), miles_sum=("annual_miles", "sum")))
            log.info("MOT %s processed", name.rsplit("/", 1)[-1])
    table = pd.concat(parts).groupby(level=[0, 1]).sum().reset_index()
    table["mean_annual_miles"] = (table["miles_sum"] / table["cars"]).round(0)
    table["share_of_fuel"] = (table["cars"] / table.groupby("fuel")["cars"].transform("sum")).round(5)
    return table.drop(columns="miles_sum").sort_values(["fuel", "age_year"]).reset_index(drop=True)

# --------------------------------------------------------------------------- assemble


def assemble_la_profile(census23: pd.DataFrame, elec: pd.DataFrame, gas: pd.DataFrame, renew: pd.DataFrame,
                        cars: pd.DataFrame, pv: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frame = census23.merge(elec, on="la_code", how="left")
    frame = frame.merge(gas, on="la_code", how="left").merge(renew, on="la_code", how="left").merge(cars, on="la_code", how="left")
    stats = {
        "las": int(len(frame)),
        "missing_region": int(frame["region"].isna().sum()),
        "missing_electricity": int(frame["elec_dom_mean_kwh"].isna().sum()),
        "missing_gas": int(frame["gas_dom_mean_kwh"].isna().sum()),
        "missing_renewables": int(frame["pv_sites_2025"].isna().sum()),
        "missing_vehicles": int(frame["cars_private_total"].isna().sum()),
    }
    if frame["region"].isna().any():
        raise KeyError(f"local authorities without a region: {frame.loc[frame.region.isna(), 'la_name'].tolist()}")

    # Shares from Census counts.
    tenure_total = frame["households"]
    frame["share_owned_outright"] = frame["tenure_owned_outright"] / tenure_total
    frame["share_mortgage"] = (frame["tenure_mortgage"] + frame["tenure_shared_ownership"]) / tenure_total
    frame["share_social_rent"] = (frame["tenure_social_rent_council"] + frame["tenure_social_rent_other"]) / tenure_total
    frame["share_private_rent"] = (frame["tenure_private_landlord"] + frame["tenure_private_other"]
                                   + frame["tenure_rent_free"]) / tenure_total
    acc = frame["acc_total"]
    frame["share_detached"] = frame["acc_detached"] / acc
    frame["share_semi"] = frame["acc_semi"] / acc
    frame["share_terraced"] = frame["acc_terraced"] / acc
    frame["share_flat"] = (frame["acc_flat_purpose_built"] + frame["acc_flat_converted"] + frame["acc_flat_other_building"]
                           + frame["acc_flat_commercial"]) / acc
    frame["share_caravan"] = frame["acc_caravan"] / acc
    for k in ("0", "1", "2", "3plus"):
        frame[f"share_cars_{k}"] = frame[f"cars_{k}"] / frame["cars_total"]
    commute_total = frame["commute_total"]
    for k in ("wfh", "metro_tram", "train", "bus", "taxi", "motorcycle", "car_driver", "car_passenger", "bicycle", "foot", "other"):
        frame[f"share_commute_{k}"] = frame[f"commute_{k}"] / commute_total
    heat_total = frame["heat_total"]
    frame["share_heat_mains_gas"] = (frame["heat_mains_gas"] + 0.5 * frame["heat_multi_no_renewable"]) / heat_total
    frame["share_heat_electric"] = (frame["heat_electric"] + frame["heat_none"]) / heat_total
    frame["share_heat_oil"] = frame["heat_oil"] / heat_total
    frame["share_heat_lpg"] = frame["heat_lpg"] / heat_total
    frame["share_heat_heat_pump_or_renewable"] = (frame["heat_renewable"] + frame["heat_multi_renewable"]) / heat_total
    frame["share_heat_other"] = (frame["heat_wood"] + frame["heat_solid"] + frame["heat_district"] + frame["heat_other"]
                                 + 0.5 * frame["heat_multi_no_renewable"]) / heat_total
    size_cols = [f"hh_size_{k}" for k in ("1", "2", "3", "4", "5", "6", "7", "8plus")]
    size_total = frame[size_cols].sum(axis=1)
    for col in size_cols:
        frame[f"share_{col}"] = frame[col] / size_total

    # Regional fallbacks for the few authorities missing from a dataset.
    fill_cols = ["elec_dom_mean_kwh", "elec_dom_median_kwh", "elec_e7_share", "gas_dom_mean_kwh", "gas_dom_median_kwh",
                 "pv_sites_2025", "pv_capacity_mw_2025", "households_desnz_2025",
                 "car_share_petrol", "car_share_diesel", "car_share_hev", "car_share_phev", "car_share_bev",
                 "car_share_other"]
    for col in fill_cols:
        frame[col] = frame[col].fillna(frame.groupby("region")[col].transform("median"))
    frame["pv_rate_per_household"] = frame["pv_sites_2025"] / frame["households_desnz_2025"]
    frame["pv_kw_per_site"] = frame["pv_capacity_mw_2025"] * 1000.0 / frame["pv_sites_2025"]

    frame["region"] = frame["region"].replace({"East of England": "East", "Inner London": "London",
                                               "Outer London": "London"})
    # Official names such as "Bristol, City of" carry a comma; use the natural order so every output stays comma-free.
    frame["la_name"] = (frame["la_name"].str.replace(r"^(.+), (City of|County of)$", r"\2 \1", regex=True)
                        .str.replace(", ", " ", regex=False))
    frame = frame.merge(pv[["region", "pv_kwh_per_kwp_year"]], on="region", how="left")
    if frame["pv_kwh_per_kwp_year"].isna().any():
        raise KeyError(f"regions without PV yield: {frame.loc[frame.pv_kwh_per_kwp_year.isna(), 'region'].unique()}")
    keep = ["la_code", "la_name", "region", "households"] + [c for c in frame.columns if c.startswith("share_")] + [
        "elec_dom_mean_kwh", "elec_dom_median_kwh", "elec_e7_share", "gas_dom_mean_kwh", "gas_dom_median_kwh",
        "pv_rate_per_household", "pv_kw_per_site", "pv_kwh_per_kwp_year", "cars_private_total",
        "car_share_petrol", "car_share_diesel", "car_share_hev", "car_share_phev", "car_share_bev", "car_share_other",
    ]
    return frame[keep].sort_values("la_code").reset_index(drop=True), stats


def build_all(raw_dir: Path, out_dir: Path, offline: bool = False) -> dict:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    raw_records = fetch_sources(raw_dir, offline)

    log.info("processing DESNZ electricity and gas by local authority")
    elec = build_electricity_la(raw_dir)
    gas = build_gas_la(raw_dir)
    names_2023 = dict(zip(elec["la_code"], elec["la_name_desnz"]))
    log.info("processing Census 2021 (6 tables, 331 authorities -> 2023 geography)")
    census23 = census_to_2023(build_census(raw_dir), names_2023)
    log.info("processing DESNZ renewables, DfT vehicles, NTS, prices, PVGIS, ONS, factors")
    renew = build_renewables_la(raw_dir)
    cars = build_vehicles_la(raw_dir)
    pv = build_pvgis(raw_dir)
    la_profile, join_stats = assemble_la_profile(census23, elec.drop(columns=["la_name_desnz"]), gas, renew, cars, pv)
    log.info("processing further Census 2021 topics (%s tables) and tenure cross-tables", len(CENSUS_EXTRA_TABLES))
    extra23 = census_to_2023(build_census_extra(raw_dir), names_2023)
    la_profile = la_profile.merge(census_extra_shares(extra23), on="la_code", how="left")
    join_stats["missing_census_extra"] = int(la_profile["share_hh_dependent_children"].isna().sum())
    rm_age, rm_activity = build_census_rm(raw_dir)
    join_stats["census_rm_las"] = int(rm_age["la_code"].nunique())
    log.info("processing NEED homes, house prices and Bank of England rates")
    need = build_need(raw_dir)
    house_prices = build_house_prices(raw_dir, la_profile["la_code"].tolist())
    nts_fuel, nts_bands = build_nts(raw_dir)

    tables: dict[str, pd.DataFrame] = {
        "la_profile.csv": la_profile,
        "pvgis_region_yield.csv": pv,
        "consumption_distribution_2024.csv": build_consumption_distribution(raw_dir),
        "nts_mileage_by_fuel.csv": nts_fuel,
        "nts_mileage_bands.csv": nts_bands,
        "fuel_prices_weekly.csv": build_fuel_prices(raw_dir),
        "ofgem_price_cap.csv": build_ofgem(raw_dir),
        "family_spending_a6_fye2025.csv": build_family_spending(raw_dir),
        "ghg_factors_2026.csv": build_ghg_factors(raw_dir),
        "ons_ghg_intensity_by_industry.csv": build_ghg_intensity(raw_dir),
        "mcc_codes.csv": pd.read_csv(raw_dir / "mcc/mcc_codes.csv", dtype=str),
        "census_tenure_age_hrp.csv": rm_age,
        "census_tenure_activity_hrp.csv": rm_activity,
        "house_prices_la_monthly.csv": house_prices,
        "boe_quoted_rates_monthly.csv": build_boe_rates(raw_dir),
        "mot_car_profiles.csv": build_mot_profiles(raw_dir),
    }
    parquet_tables = {"need_homes.parquet": need}
    bench_dir = out_dir.parent / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    bench_tables = {
        "dodatathings_us_bank_categories.csv": pd.read_csv(raw_dir / "bench/dodatathings_transactions_synthetic.csv"),
    }

    processed = []
    for name, table in tables.items():
        path = out_dir / name
        table.to_csv(path, index=False)
        processed.append({"file": f"reference/{name}", "rows": int(len(table)), "columns": int(table.shape[1]),
                          "sha256": sha256_file(path)})
    for name, table in parquet_tables.items():
        path = out_dir / name
        table.to_parquet(path, index=False, compression="zstd")
        processed.append({"file": f"reference/{name}", "rows": int(len(table)), "columns": int(table.shape[1]),
                          "sha256": sha256_file(path)})
    for name, table in bench_tables.items():
        path = bench_dir / name
        table.to_csv(path, index=False)
        processed.append({"file": f"benchmarks/{name}", "rows": int(len(table)), "columns": int(table.shape[1]),
                          "sha256": sha256_file(path)})

    manifest = {
        "built_at_utc": started.isoformat(timespec="seconds"),
        "geography": "England and Wales, local authorities on April 2023 boundaries (Census 2021 merged where needed)",
        "join_stats": join_stats,
        "raw_sources": raw_records,
        "processed_files": processed,
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest
