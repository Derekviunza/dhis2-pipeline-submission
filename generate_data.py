"""
generate_data.py
----------------
PSI DISC · DHIS2 Health Data Pipeline Challenge
Associate Data Engineer Take-Home Assessment

Generates four synthetic DHIS2 JSON export files that mirror the real
structure returned by the DHIS2 Web API:

    data/
        metadata.json       — dataElements + categoryOptionCombos
        org_units.json      — organisationUnits with hierarchy (4 levels)
        programs.json       — health programs mapping to data elements
        data_values.json    — dataValueSets with injected quality issues

Intentional quality issues (all must be handled in pipeline.py):
  ┌─────────────────────────────────────────────────────────────────┐
  │  ~8%  duplicate dataValue rows                                  │
  │  ~5%  dataValues with UIDs absent from metadata (unresolvable)  │
  │  ~4%  dataValues referencing ghost orgUnit UIDs                 │
  │  ~12% late-reported (lastUpdated > 60d after period end)        │
  │  ~6%  explicit zero values (raw_value = "0")                    │
  │  ~3%  NULL / missing raw_value                                  │
  │  ~8%  facilities that skip one or more reporting periods        │
  │  Mixed value types (INTEGER, NUMBER, TEXT, BOOLEAN, PERCENTAGE) │
  │  Inconsistent categoryOptionCombo assignment                    │
  │  Orphaned COC UIDs not present in metadata                      │
  └─────────────────────────────────────────────────────────────────┘

Usage:
    pip install faker numpy
    python generate_data.py
    python generate_data.py --countries 6 --periods 12 --seed 42 --out ./data
"""

import argparse
import json
import os
import random
import string
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
from faker import Faker

# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic DHIS2 JSON exports.")
    p.add_argument("--countries",  type=int, default=5,     help="Number of PSI countries (default 5)")
    p.add_argument("--periods",    type=int, default=12,    help="Number of monthly periods (default 12)")
    p.add_argument("--seed",       type=int, default=42,    help="Random seed (default 42)")
    p.add_argument("--out",        type=str, default="./data", help="Output directory (default ./data)")
    p.add_argument("--pretty",     action="store_true",     help="Pretty-print JSON (larger files)")
    return p.parse_args()


# ── Seeded RNG ─────────────────────────────────────────────────────────────────

FAKE = Faker()

def init_rng(seed: int) -> np.random.Generator:
    Faker.seed(seed)
    random.seed(seed)
    return np.random.default_rng(seed)


# ── DHIS2 UID generator ────────────────────────────────────────────────────────
# Real DHIS2 UIDs: 11 chars, first char is a letter, rest alphanumeric.

def dhis2_uid(rng: np.random.Generator) -> str:
    first = rng.choice(list(string.ascii_letters))
    rest  = "".join(rng.choice(list(string.ascii_letters + string.digits), size=10).tolist())
    return first + rest


def uid_pool(n: int, rng: np.random.Generator) -> list[str]:
    """Generate n unique DHIS2 UIDs."""
    seen = set()
    result = []
    while len(result) < n:
        uid = dhis2_uid(rng)
        if uid not in seen:
            seen.add(uid)
            result.append(uid)
    return result


# ── Domain constants ───────────────────────────────────────────────────────────

HEALTH_AREAS = ["Malaria", "HIV", "Tuberculosis", "Reproductive Health", "Child Health"]

# Per health area: list of (indicator name, valueType)
INDICATORS: dict[str, list[tuple[str, str]]] = {
    "Malaria": [
        ("Malaria confirmed cases", "INTEGER_ZERO_OR_POSITIVE"),
        ("Malaria deaths", "INTEGER_ZERO_OR_POSITIVE"),
        ("ACT doses distributed", "INTEGER_ZERO_OR_POSITIVE"),
        ("RDT tests conducted", "INTEGER_ZERO_OR_POSITIVE"),
        ("LLIN nets distributed", "INTEGER_ZERO_OR_POSITIVE"),
        ("Malaria positivity rate", "PERCENTAGE"),
        ("IRS households covered", "INTEGER_ZERO_OR_POSITIVE"),
        ("Malaria in pregnancy cases", "INTEGER_ZERO_OR_POSITIVE"),
    ],
    "HIV": [
        ("HIV tests conducted", "INTEGER_ZERO_OR_POSITIVE"),
        ("HIV positive results", "INTEGER_ZERO_OR_POSITIVE"),
        ("ART patients enrolled", "INTEGER_ZERO_OR_POSITIVE"),
        ("Viral load tests done", "INTEGER_ZERO_OR_POSITIVE"),
        ("PMTCT clients enrolled", "INTEGER_ZERO_OR_POSITIVE"),
        ("HIV positivity rate", "PERCENTAGE"),
        ("PrEP initiations", "INTEGER_ZERO_OR_POSITIVE"),
        ("HIV self-test kits distributed", "INTEGER_ZERO_OR_POSITIVE"),
    ],
    "Tuberculosis": [
        ("TB presumptive cases", "INTEGER_ZERO_OR_POSITIVE"),
        ("TB bacteriologically confirmed", "INTEGER_ZERO_OR_POSITIVE"),
        ("TB treatment initiated", "INTEGER_ZERO_OR_POSITIVE"),
        ("TB treatment success rate", "PERCENTAGE"),
        ("DR-TB cases detected", "INTEGER_ZERO_OR_POSITIVE"),
        ("TB contacts investigated", "INTEGER_ZERO_OR_POSITIVE"),
        ("TPT initiations", "INTEGER_ZERO_OR_POSITIVE"),
        ("GeneXpert tests conducted", "INTEGER_ZERO_OR_POSITIVE"),
    ],
    "Reproductive Health": [
        ("ANC first visits", "INTEGER_ZERO_OR_POSITIVE"),
        ("ANC fourth visits", "INTEGER_ZERO_OR_POSITIVE"),
        ("Skilled birth attendants", "INTEGER_ZERO_OR_POSITIVE"),
        ("Modern contraceptive users", "INTEGER_ZERO_OR_POSITIVE"),
        ("Family planning new acceptors", "INTEGER_ZERO_OR_POSITIVE"),
        ("Safe abortion services", "INTEGER_ZERO_OR_POSITIVE"),
        ("Post-abortion care clients", "INTEGER_ZERO_OR_POSITIVE"),
        ("Contraceptive prevalence rate", "PERCENTAGE"),
    ],
    "Child Health": [
        ("Under-5 outpatient visits", "INTEGER_ZERO_OR_POSITIVE"),
        ("DPT3 immunisations", "INTEGER_ZERO_OR_POSITIVE"),
        ("Measles immunisations", "INTEGER_ZERO_OR_POSITIVE"),
        ("Severe acute malnutrition cases", "INTEGER_ZERO_OR_POSITIVE"),
        ("ORS distributed", "INTEGER_ZERO_OR_POSITIVE"),
        ("Vitamin A supplements", "INTEGER_ZERO_OR_POSITIVE"),
        ("Under-5 mortality reported", "INTEGER_ZERO_OR_POSITIVE"),
        ("Full immunisation coverage", "PERCENTAGE"),
    ],
}

# Category disaggregations common in DHIS2 HMIS
CATEGORY_COMBOS: list[tuple[str, list[str]]] = [
    ("Age/Sex",         ["<5 Male", "<5 Female", "5-14 Male", "5-14 Female", "15+ Male", "15+ Female"]),
    ("Sex",             ["Male", "Female"]),
    ("Age",             ["<1 year", "1-4 years", "5-14 years", "15-24 years", "25+ years"]),
    ("Default",         ["default"]),
    ("Service type",    ["Inpatient", "Outpatient", "Community"]),
    ("Facility type",   ["Public", "Private", "NGO/FBO"]),
]

PSI_COUNTRIES = [
    "Kenya", "Nigeria", "Ethiopia", "Uganda", "Tanzania",
    "Mozambique", "Malawi", "Zambia", "Zimbabwe", "Senegal",
    "Ghana", "Mali", "Burkina Faso", "DRC", "Madagascar",
]

# Typical number of districts per country and facilities per district
COUNTRY_REGIONS   = (2, 4)    # regions per country
COUNTRY_DISTRICTS = (3, 6)    # districts per region
DISTRICT_FACILITIES = (5, 15) # facilities per district


# ── Period helpers ─────────────────────────────────────────────────────────────

def period_str(year: int, month: int) -> str:
    return f"{year}{month:02d}"

def period_end_date(year: int, month: int) -> date:
    if month == 12:
        return date(year + 1, 1, 1) - timedelta(days=1)
    return date(year, month + 1, 1) - timedelta(days=1)

def dhis2_timestamp(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000")

def random_datetime_near(base: date, offset_days_range: tuple, rng: np.random.Generator) -> datetime:
    lo, hi = offset_days_range
    offset = int(rng.integers(lo, hi + 1))
    dt = datetime(base.year, base.month, base.day) + timedelta(days=offset)
    dt = dt.replace(
        hour=int(rng.integers(7, 18)),
        minute=int(rng.integers(0, 60)),
        second=int(rng.integers(0, 60)),
    )
    return dt


# ══════════════════════════════════════════════════════════════════════════════
# BUILD metadata.json
# ══════════════════════════════════════════════════════════════════════════════

def build_metadata(rng: np.random.Generator) -> tuple[dict, dict, dict]:
    """
    Returns (metadata_doc, de_uid_map, coc_uid_map).
      de_uid_map  : { uid → (name, valueType, health_area) }
      coc_uid_map : { uid → name }
    """
    print("  Building metadata.json …")

    # ── Data elements ──────────────────────────────────────────────────────
    de_uid_map: dict[str, tuple[str, str, str]] = {}
    data_elements = []

    for health_area, indicators in INDICATORS.items():
        for name, value_type in indicators:
            uid = dhis2_uid(rng)
            de_uid_map[uid] = (name, value_type, health_area)

            # Pick a representative category combo
            cc_name, cc_options = random.choice(CATEGORY_COMBOS)
            cc_uid = dhis2_uid(rng)

            data_elements.append({
                "id":          uid,
                "name":        name,
                "shortName":   name[:50],
                "code":        f"PSI_{health_area[:3].upper()}_{len(data_elements)+1:03d}",
                "valueType":   value_type,
                "domainType":  "AGGREGATE",
                "aggregationType": "SUM" if "rate" not in name.lower() and "coverage" not in name.lower() else "AVERAGE",
                "zeroIsSignificant": True,
                "categoryCombo": {
                    "id":   cc_uid,
                    "name": cc_name,
                },
                "dataElementGroups": [{"id": dhis2_uid(rng), "name": health_area}],
                "created":     dhis2_timestamp(datetime(2020, 1, 1)),
                "lastUpdated": dhis2_timestamp(datetime(2023, 6, 1)),
            })

    # ── Category option combos ─────────────────────────────────────────────
    coc_uid_map: dict[str, str] = {}
    category_option_combos = []

    for _, options in CATEGORY_COMBOS:
        for option in options:
            uid  = dhis2_uid(rng)
            coc_uid_map[uid] = option
            category_option_combos.append({
                "id":          uid,
                "name":        option,
                "created":     dhis2_timestamp(datetime(2020, 1, 1)),
                "lastUpdated": dhis2_timestamp(datetime(2023, 6, 1)),
            })

    # Add "default" COC (always present in DHIS2)
    default_coc_uid = dhis2_uid(rng)
    coc_uid_map[default_coc_uid] = "default"
    category_option_combos.append({
        "id":          default_coc_uid,
        "name":        "default",
        "created":     dhis2_timestamp(datetime(2019, 1, 1)),
        "lastUpdated": dhis2_timestamp(datetime(2023, 6, 1)),
    })

    metadata_doc = {
        "date":                 dhis2_timestamp(datetime.utcnow()),
        "version":              "2.38",
        "dataElements":         data_elements,
        "categoryOptionCombos": category_option_combos,
    }

    print(f"    dataElements:         {len(data_elements)}")
    print(f"    categoryOptionCombos: {len(category_option_combos)}")
    return metadata_doc, de_uid_map, coc_uid_map


# ══════════════════════════════════════════════════════════════════════════════
# BUILD org_units.json
# ══════════════════════════════════════════════════════════════════════════════

def build_org_units(
    countries: list[str],
    rng: np.random.Generator,
) -> tuple[dict, list[dict]]:
    """
    Build a 4-level org unit hierarchy:
        Level 1 — National (one per country)
        Level 2 — Region / Province
        Level 3 — District
        Level 4 — Facility

    Returns (org_units_doc, flat_ou_list) where flat_ou_list contains
    dicts with {uid, name, level, parent_uid, country_name, path}.
    """
    print("  Building org_units.json …")

    org_units   = []
    flat_ou_list = []

    for country in countries:
        # Level 1 — National
        nat_uid  = dhis2_uid(rng)
        nat_path = f"/{nat_uid}"
        org_units.append({
            "id":        nat_uid,
            "name":      country,
            "shortName": country,
            "code":      f"{country[:3].upper()}_NAT",
            "level":     1,
            "path":      nat_path,
            "parent":    None,
            "groups":    [{"id": dhis2_uid(rng), "name": "National"}],
            "created":     dhis2_timestamp(datetime(2015, 1, 1)),
            "lastUpdated": dhis2_timestamp(datetime(2023, 1, 1)),
        })
        flat_ou_list.append({
            "uid": nat_uid, "name": country, "level": 1,
            "parent_uid": None, "country_name": country, "path": nat_path,
        })

        n_regions = int(rng.integers(*COUNTRY_REGIONS))
        for r in range(n_regions):
            # Level 2 — Region
            region_name = f"{FAKE.city()} Region"
            reg_uid     = dhis2_uid(rng)
            reg_path    = f"{nat_path}/{reg_uid}"
            org_units.append({
                "id":        reg_uid,
                "name":      region_name,
                "shortName": region_name[:50],
                "level":     2,
                "path":      reg_path,
                "parent":    {"id": nat_uid, "name": country},
                "groups":    [{"id": dhis2_uid(rng), "name": "Region"}],
                "created":     dhis2_timestamp(datetime(2015, 1, 1)),
                "lastUpdated": dhis2_timestamp(datetime(2023, 1, 1)),
            })
            flat_ou_list.append({
                "uid": reg_uid, "name": region_name, "level": 2,
                "parent_uid": nat_uid, "country_name": country, "path": reg_path,
            })

            n_districts = int(rng.integers(*COUNTRY_DISTRICTS))
            for d in range(n_districts):
                # Level 3 — District
                district_name = f"{FAKE.city()} District"
                dist_uid      = dhis2_uid(rng)
                dist_path     = f"{reg_path}/{dist_uid}"
                org_units.append({
                    "id":        dist_uid,
                    "name":      district_name,
                    "shortName": district_name[:50],
                    "level":     3,
                    "path":      dist_path,
                    "parent":    {"id": reg_uid, "name": region_name},
                    "groups":    [{"id": dhis2_uid(rng), "name": "District"}],
                    "created":     dhis2_timestamp(datetime(2016, 1, 1)),
                    "lastUpdated": dhis2_timestamp(datetime(2023, 1, 1)),
                })
                flat_ou_list.append({
                    "uid": dist_uid, "name": district_name, "level": 3,
                    "parent_uid": reg_uid, "country_name": country, "path": dist_path,
                })

                n_facilities = int(rng.integers(*DISTRICT_FACILITIES))
                for f_idx in range(n_facilities):
                    # Level 4 — Facility
                    fac_type = rng.choice(["Health Centre", "Dispensary", "Hospital", "Clinic", "Health Post"])
                    fac_name = f"{FAKE.last_name()} {fac_type}"
                    fac_uid  = dhis2_uid(rng)
                    fac_path = f"{dist_path}/{fac_uid}"
                    org_units.append({
                        "id":        fac_uid,
                        "name":      fac_name,
                        "shortName": fac_name[:50],
                        "level":     4,
                        "path":      fac_path,
                        "parent":    {"id": dist_uid, "name": district_name},
                        "groups":    [
                            {"id": dhis2_uid(rng), "name": "Facility"},
                            {"id": dhis2_uid(rng), "name": fac_type},
                        ],
                        "created":     dhis2_timestamp(datetime(2016, 6, 1)),
                        "lastUpdated": dhis2_timestamp(datetime(2023, 6, 1)),
                    })
                    flat_ou_list.append({
                        "uid": fac_uid, "name": fac_name, "level": 4,
                        "parent_uid": dist_uid, "country_name": country, "path": fac_path,
                    })

    ou_doc = {
        "date":               dhis2_timestamp(datetime.utcnow()),
        "version":            "2.38",
        "organisationUnits":  org_units,
    }

    n_facilities = sum(1 for o in flat_ou_list if o["level"] == 4)
    print(f"    Countries:  {len(countries)}")
    print(f"    Org units:  {len(org_units)}  (facilities: {n_facilities})")
    return ou_doc, flat_ou_list


# ══════════════════════════════════════════════════════════════════════════════
# BUILD programs.json
# ══════════════════════════════════════════════════════════════════════════════

def build_programs(
    countries: list[str],
    de_uid_map: dict[str, tuple[str, str, str]],
    rng: np.random.Generator,
) -> tuple[dict, list[dict]]:
    """
    One program record per (health_area × country) combination.
    Each program lists the UID subset of data elements it owns.
    Returns (programs_doc, flat_programs_list).
    """
    print("  Building programs.json …")

    # Reverse lookup: health_area → list of UIDs
    area_to_uids: dict[str, list[str]] = {}
    for uid, (_, _, health_area) in de_uid_map.items():
        area_to_uids.setdefault(health_area, []).append(uid)

    programs       = []
    flat_prog_list = []

    for country in countries:
        # Not every country runs every health area (realistic)
        n_areas      = int(rng.integers(2, len(HEALTH_AREAS) + 1))
        active_areas = rng.choice(HEALTH_AREAS, size=n_areas, replace=False).tolist()

        for health_area in active_areas:
            prog_uid = dhis2_uid(rng)
            uids     = area_to_uids.get(health_area, [])

            # Some programs report a subset of indicators (~80–100%)
            n_de     = max(1, int(len(uids) * rng.uniform(0.8, 1.0)))
            prog_des = rng.choice(uids, size=n_de, replace=False).tolist()

            prog = {
                "id":                 prog_uid,
                "name":               f"PSI {country} {health_area} Program",
                "shortName":          f"{country[:3].upper()}_{health_area[:3].upper()}",
                "healthArea":         health_area,
                "country":            country,
                "reportingFrequency": "MONTHLY",
                "dataElements":       prog_des,
                "created":            dhis2_timestamp(datetime(2019, 1, 1)),
                "lastUpdated":        dhis2_timestamp(datetime(2023, 9, 1)),
            }
            programs.append(prog)
            flat_prog_list.append(prog)

    prog_doc = {
        "date":     dhis2_timestamp(datetime.utcnow()),
        "version":  "2.38",
        "programs": programs,
    }

    print(f"    Programs: {len(programs)}  (across {len(countries)} countries × up to {len(HEALTH_AREAS)} health areas)")
    return prog_doc, flat_prog_list


# ══════════════════════════════════════════════════════════════════════════════
# BUILD data_values.json   (the main ELT input — most complexity here)
# ══════════════════════════════════════════════════════════════════════════════

def realistic_value(
    value_type: str,
    indicator_name: str,
    rng: np.random.Generator,
) -> str:
    """
    Return a realistic string value for a given DHIS2 valueType.
    Values are intentionally strings — DHIS2 always serialises them this way.
    """
    name_lower = indicator_name.lower()

    if value_type == "PERCENTAGE":
        # Rates: 0–100, most between 30–85 for health indicators
        val = float(rng.beta(3, 2) * 85)
        return f"{val:.1f}"

    if value_type in ("INTEGER_ZERO_OR_POSITIVE", "INTEGER", "INTEGER_POSITIVE"):
        if "death" in name_lower or "mortality" in name_lower:
            return str(int(rng.integers(0, 15)))
        if "test" in name_lower or "visit" in name_lower or "case" in name_lower:
            return str(int(rng.negative_binomial(5, 0.3)))
        if "net" in name_lower or "dose" in name_lower or "kit" in name_lower:
            return str(int(rng.integers(10, 500)))
        return str(int(rng.integers(1, 300)))

    if value_type == "NUMBER":
        return f"{rng.uniform(0, 100):.2f}"

    if value_type == "BOOLEAN":
        return rng.choice(["true", "false"])

    if value_type == "TRUE_ONLY":
        return "true"

    return str(int(rng.integers(0, 100)))


def build_data_values(
    flat_ou_list: list[dict],
    flat_prog_list: list[dict],
    de_uid_map: dict[str, tuple[str, str, str]],
    coc_uid_map: dict[str, str],
    periods: int,
    rng: np.random.Generator,
) -> dict:
    """
    Generate dataValues mimicking a DHIS2 dataValueSets API export.

    For each facility in each country, for each period, for each program
    active in that country, emit one row per data element in that program.

    Then inject all quality issues documented in the module docstring.
    """
    print("  Building data_values.json …")

    # Only level-4 facilities report data
    facilities = [ou for ou in flat_ou_list if ou["level"] == 4]

    # Build lookup: country → list of programs
    country_to_progs: dict[str, list[dict]] = {}
    for prog in flat_prog_list:
        country_to_progs.setdefault(prog["country"], []).append(prog)

    # Facility UID → country mapping for quick lookup
    facility_country = {ou["uid"]: ou["country_name"] for ou in facilities}

    # COC UID list for sampling
    coc_uids = list(coc_uid_map.keys())
    de_uids  = list(de_uid_map.keys())

    # Generate periods starting 'periods' months before today
    today = date.today()
    start_year  = today.year - (periods // 12 + 1)
    start_month = today.month
    period_list: list[tuple[int, int]] = []
    y, m = start_year, start_month
    for _ in range(periods):
        m += 1
        if m > 12:
            m = 1
            y += 1
        period_list.append((y, m))

    # ── Generate clean data values ────────────────────────────────────────
    clean_rows: list[dict] = []

    # Facilities that will randomly skip some periods (simulates non-reporting)
    skip_facilities = set(
        rng.choice([f["uid"] for f in facilities], size=max(1, int(len(facilities) * 0.08)), replace=False).tolist()
    )

    for facility in facilities:
        country = facility["country_name"]
        progs   = country_to_progs.get(country, [])
        if not progs:
            continue

        for year, month in period_list:
            # ~8% of facilities skip this period entirely
            if facility["uid"] in skip_facilities and rng.random() < 0.5:
                continue

            per = period_str(year, month)
            p_end = period_end_date(year, month)

            for prog in progs:
                for de_uid in prog["dataElements"]:
                    if de_uid not in de_uid_map:
                        continue

                    de_name, value_type, _ = de_uid_map[de_uid]

                    # Pick a category option combo
                    coc_uid = rng.choice(coc_uids)

                    # Generate a realistic value
                    raw_value = realistic_value(value_type, de_name, rng)

                    # Normal lastUpdated: 1–30 days after period end
                    last_updated_dt = random_datetime_near(p_end, (1, 30), rng)
                    created_dt      = random_datetime_near(p_end, (1, 5),  rng)

                    clean_rows.append({
                        "dataElement":          de_uid,
                        "period":               per,
                        "orgUnit":              facility["uid"],
                        "categoryOptionCombo":  coc_uid,
                        "attributeOptionCombo": dhis2_uid(rng),  # usually "default" OU combo
                        "value":                raw_value,
                        "storedBy":             FAKE.user_name(),
                        "created":              dhis2_timestamp(created_dt),
                        "lastUpdated":          dhis2_timestamp(last_updated_dt),
                        "followup":             "false",
                    })

    total_clean = len(clean_rows)
    print(f"    Clean rows generated: {total_clean:,}")

    # ── Inject quality issues ─────────────────────────────────────────────

    def n_rows(fraction: float) -> int:
        return max(1, int(total_clean * fraction))

    # 1. Late-reported rows (~12%): lastUpdated > 60 days after period end
    late_indices = rng.choice(total_clean, size=n_rows(0.12), replace=False).tolist()
    for i in late_indices:
        row = clean_rows[i]
        year, month = int(row["period"][:4]), int(row["period"][4:])
        p_end = period_end_date(year, month)
        late_dt = random_datetime_near(p_end, (61, 180), rng)
        clean_rows[i]["lastUpdated"] = dhis2_timestamp(late_dt)

    # 2. Explicit zero values (~6%): raw_value = "0"
    zero_indices = rng.choice(total_clean, size=n_rows(0.06), replace=False).tolist()
    for i in zero_indices:
        clean_rows[i]["value"] = "0"

    # 3. NULL / missing raw_value (~3%): value = None → will serialise as null
    null_indices = rng.choice(total_clean, size=n_rows(0.03), replace=False).tolist()
    for i in null_indices:
        clean_rows[i]["value"] = None

    # 4. Unresolvable dataElement UIDs (~5%): ghost UIDs not in metadata
    ghost_de_uids  = uid_pool(20, rng)
    ghost_de_indices = rng.choice(total_clean, size=n_rows(0.05), replace=False).tolist()
    for i in ghost_de_indices:
        clean_rows[i]["dataElement"] = ghost_de_uids[int(rng.integers(0, len(ghost_de_uids)))]

    # 5. Orphaned orgUnit UIDs (~4%): ghost facility UIDs not in org_units.json
    ghost_ou_uids  = uid_pool(15, rng)
    ghost_ou_indices = rng.choice(total_clean, size=n_rows(0.04), replace=False).tolist()
    for i in ghost_ou_indices:
        clean_rows[i]["orgUnit"] = ghost_ou_uids[int(rng.integers(0, len(ghost_ou_uids)))]

    # 6. Orphaned COC UIDs not in metadata (~3%)
    ghost_coc_uids = uid_pool(10, rng)
    ghost_coc_indices = rng.choice(total_clean, size=n_rows(0.03), replace=False).tolist()
    for i in ghost_coc_indices:
        clean_rows[i]["categoryOptionCombo"] = ghost_coc_uids[int(rng.integers(0, len(ghost_coc_uids)))]

    # 7. Inconsistent storedBy casing (realistic — multiple users, typos)
    for row in clean_rows:
        if rng.random() < 0.15:
            row["storedBy"] = row["storedBy"].upper()
        elif rng.random() < 0.10:
            row["storedBy"] = row["storedBy"].capitalize()

    # 8. Duplicate rows (~8%): exact copy of an existing row
    n_dups = n_rows(0.08)
    dup_indices = rng.choice(total_clean, size=n_dups, replace=True).tolist()
    duplicate_rows = [dict(clean_rows[i]) for i in dup_indices]

    # 9. Near-duplicate rows (~2%): same key, different value + updated timestamp
    #    These represent a data correction submitted after the original entry.
    near_dup_indices = rng.choice(total_clean, size=n_rows(0.02), replace=False).tolist()
    near_duplicates  = []
    for i in near_dup_indices:
        row   = dict(clean_rows[i])
        year, month = int(row["period"][:4]), int(row["period"][4:])
        p_end = period_end_date(year, month)
        # New value and a later lastUpdated — pipeline must handle dedup correctly
        de_uid = row["dataElement"]
        if de_uid in de_uid_map:
            _, value_type, _ = de_uid_map[de_uid]
        else:
            value_type = "INTEGER_ZERO_OR_POSITIVE"
        row["value"]       = realistic_value(value_type, "generic", rng)
        row["lastUpdated"] = dhis2_timestamp(random_datetime_near(p_end, (5, 45), rng))
        near_duplicates.append(row)

    # Combine all rows and shuffle
    all_rows = clean_rows + duplicate_rows + near_duplicates
    rng.shuffle(all_rows)

    total_final = len(all_rows)
    print(f"    Total rows after injection: {total_final:,}")
    print(f"      Late-reported:         ~{n_rows(0.12):,}  ({12}%)")
    print(f"      Explicit zeros:        ~{n_rows(0.06):,}  ({6}%)")
    print(f"      NULL values:           ~{n_rows(0.03):,}  ({3}%)")
    print(f"      Ghost dataElement:     ~{n_rows(0.05):,}  ({5}%)")
    print(f"      Ghost orgUnit:         ~{n_rows(0.04):,}  ({4}%)")
    print(f"      Ghost COC:             ~{n_rows(0.03):,}  ({3}%)")
    print(f"      Exact duplicates:      ~{n_dups:,}  ({8}%)")
    print(f"      Near-duplicates:       ~{n_rows(0.02):,}  ({2}%)")

    dv_doc = {
        "responseType":   "DataValueSet",
        "version":        "2.38",
        "exportDate":     dhis2_timestamp(datetime.utcnow()),
        "dataValues":     all_rows,
    }
    return dv_doc


# ══════════════════════════════════════════════════════════════════════════════
# WRITE helpers
# ══════════════════════════════════════════════════════════════════════════════

def write_json(obj: Any, path: str, pretty: bool) -> None:
    indent = 2 if pretty else None
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, default=str)
    size_kb = os.path.getsize(path) / 1024
    print(f"    → {path}  ({size_kb:,.0f} KB)")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    rng  = init_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    # Validate
    if args.countries > len(PSI_COUNTRIES):
        raise ValueError(f"--countries max is {len(PSI_COUNTRIES)}")

    countries = PSI_COUNTRIES[:args.countries]

    print("=" * 65)
    print("  PSI DISC · DHIS2 Synthetic Data Generator")
    print("=" * 65)
    print(f"  seed={args.seed}  countries={args.countries}  periods={args.periods}")
    print(f"  output → {os.path.abspath(args.out)}")
    print()

    # 1. Metadata
    metadata_doc, de_uid_map, coc_uid_map = build_metadata(rng)
    write_json(metadata_doc, f"{args.out}/metadata.json", args.pretty)
    print()

    # 2. Org units
    ou_doc, flat_ou_list = build_org_units(countries, rng)
    write_json(ou_doc, f"{args.out}/org_units.json", args.pretty)
    print()

    # 3. Programs
    prog_doc, flat_prog_list = build_programs(countries, de_uid_map, rng)
    write_json(prog_doc, f"{args.out}/programs.json", args.pretty)
    print()

    # 4. Data values (largest file, generated last)
    dv_doc = build_data_values(
        flat_ou_list,
        flat_prog_list,
        de_uid_map,
        coc_uid_map,
        periods=args.periods,
        rng=rng,
    )
    write_json(dv_doc, f"{args.out}/data_values.json", args.pretty)
    print()

    # ── Summary table ──────────────────────────────────────────────────────
    total_dv = len(dv_doc["dataValues"])
    total_ou = len(ou_doc["organisationUnits"])
    total_de = len(metadata_doc["dataElements"])

    print("=" * 65)
    print("  Generated files")
    print("=" * 65)
    for fname in ["metadata.json", "org_units.json", "programs.json", "data_values.json"]:
        path = f"{args.out}/{fname}"
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  {fname:<22}  {size_mb:>6.2f} MB")

    print()
    print("  Dataset summary")
    print(f"  {'Countries:':<28} {len(countries)}")
    print(f"  {'Org units (total):':<28} {total_ou:,}")
    print(f"  {'Facilities (level 4):':<28} {sum(1 for o in flat_ou_list if o['level']==4):,}")
    print(f"  {'Data elements:':<28} {total_de}")
    print(f"  {'Programs:':<28} {len(prog_doc['programs'])}")
    print(f"  {'Reporting periods:':<28} {args.periods}  (monthly)")
    print(f"  {'Total data values:':<28} {total_dv:,}")
    print()
    print("  Injected quality issues")
    print(f"  {'Late-reported (>60d):':<28} ~12%  of clean rows")
    print(f"  {'Explicit zeros:':<28} ~6%")
    print(f"  {'NULL values:':<28} ~3%")
    print(f"  {'Ghost dataElement UIDs:':<28} ~5%  (unresolvable)")
    print(f"  {'Ghost orgUnit UIDs:':<28} ~4%  (orphaned)")
    print(f"  {'Ghost COC UIDs:':<28} ~3%  (orphaned)")
    print(f"  {'Exact duplicate rows:':<28} ~8%")
    print(f"  {'Near-duplicate rows:':<28} ~2%  (same key, corrected value)")
    print(f"  {'Non-reporting facilities:':<28} ~8%  skip some periods")
    print("=" * 65)
    print("  Done. Run:  python pipeline.py --data-dir ./data --output-dir ./output")
    print("=" * 65)


if __name__ == "__main__":
    main()
