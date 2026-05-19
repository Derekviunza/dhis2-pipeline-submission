# DHIS2 Health Data Pipeline

## Overview

This project implements a local PySpark ELT pipeline for synthetic DHIS2 health service data. It ingests four JSON exports, resolves metadata and org unit UIDs, applies data quality rules, builds a dimensional warehouse, and produces analytics-ready outputs.

## Setup

### Linux/WSL2
```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows
PySpark on Windows requires Hadoop binaries (winutils.exe and hadoop.dll):

1. Create a `bin` folder at the project root
2. Download winutils.exe and hadoop.dll (Hadoop 3.2.x compatible) into the bin folder
3. Create virtual environment:
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The pipeline.py automatically sets HADOOP_HOME and adds the bin folder to PATH.

## Generate Data

```bash
python generate_data.py --countries 5 --periods 12 --seed 42 --out ./data
```

## Run Pipeline

```bash
python pipeline.py --data-dir ./data --output-dir ./output
```

## Project Structure

```
models/
  ingestion.py
  metadata.py
  org_units.py
  quality.py
  dimensional.py
  analytics.py
  aggregation.py
pipeline.py
generate_data.py
```

## Star Schema

```
                     dim_data_element
                            |
                            |
dim_period ---- fact_service_delivery ---- dim_org_unit
                            |
                            |
                       dim_program
```

### Fact Table

fact_service_delivery contains one row per data element, facility, period, and category option combo after deduplication.

### Dimensions

- dim_data_element
- dim_org_unit
- dim_period
- dim_program

## Data Quality Handling

The pipeline explicitly handles:

- Malformed records
- Unresolved data element UIDs
- Unresolved org unit UIDs
- Unresolved category option combos
- Duplicate and near-duplicate rows
- Late reporting
- Explicit zero values
- Null/missing values

## Design Decisions

- Explicit Spark schemas are used instead of schema inference.
- Metadata joins use broadcast joins because metadata is small.
- Unresolved UIDs are written to DQ outputs instead of being silently dropped.
- Latest lastUpdated wins for near-duplicate corrections.
- Zero and null are preserved as different reporting states.
- Fact table is partitioned by health_area and year_month.

## Assumptions

- Level-4 org units are facilities.
- lastUpdated is the correction timestamp used for deduplication.
- Program expected indicators are derived from programs.json.
- Country and health area are used to map data elements to programs.

## Known Limitations

- The low-completeness flag currently identifies countries with three or more low-completeness periods, but the first version does not fully enforce strict consecutive-period streak logic.
- The pipeline writes local outputs only and does not assume cloud storage or Databricks.
- Windows requires Hadoop binaries (winutils.exe and hadoop.dll) in the bin folder for PySpark to function properly.

## Output Folders

- output/warehouse/
- output/analytics/
- output/cross_country/
- output/dq/
- output/quarantine/
- output/logs/

## Approximate Time Spent

Add your actual time here.

## Tasks Completed

- Task 01: JSON ingestion and flattening
- Task 02: Metadata UID resolution
- Task 03: Org unit hierarchy resolution
- Task 04: DQ flags and completeness
- Task 05: Dimensional model
- Task 06: Program analytics
- Task 07: Cross-country aggregation
- Task 08: Pipeline orchestration
