# DHIS2 Health Data Pipeline

## Overview

This project implements a local PySpark ELT pipeline for synthetic DHIS2 health service data. It ingests JSON exports, resolves metadata and org unit UIDs, applies data quality rules, builds a dimensional warehouse, and produces analytics-ready outputs.

This implementation has been stabilized for local execution on Windows environments, refactored to support dynamic hierarchy traversal, optimized using window functions rather than group-by operations, and secured using runtime data contract validation.

## Setup

### Prerequisites
- Python 3.10+
- Java JDK 11 or 17 (set `JAVA_HOME` environment variable)

### Linux/WSL2
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows Local Environment
PySpark on Windows requires Hadoop binaries (`winutils.exe` and `hadoop.dll`) and Java 17 encapsulation flags:

1. Create a `bin` folder at the project root.
2. Download `winutils.exe` and `hadoop.dll` (Hadoop 3.2.x or 3.3.x compatible) into the `./bin` folder.
3. Create a virtual environment:
   ```powershell
   python -m venv .venv2
   .venv2\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

*Note: `pipeline.py` programmatically configures the `HADOOP_HOME`, JVM `--add-opens` flags, and path separator constraints to ensure smooth execution on Windows out of the box.*

## Generate Data

Before running the pipeline, generate the synthetic dataset:
```bash
python generate_data.py --countries 5 --periods 12 --seed 42 --out ./data
```

## Run Pipeline

Execute the pipeline using the local virtual environment:
```bash
python pipeline.py --data-dir ./data --output-dir ./output
```

## Project Structure

```
models/
  ingestion.py      # Reads JSON exports using explicit Spark schemas
  metadata.py       # Flat data element & Category Option Combo metadata lookups
  org_units.py      # Dynamic hierarchy construction from paths
  quality.py        # DQ flags (late, zero, null), program maps, completeness
  dimensional.py    # Builds fact and dimensional model output streams
  analytics.py      # Program reporting rate & underreporting indicators
  aggregation.py    # Cross-country performance summaries
  contract.py       # Data Contract YAML schema assertion tests
pipeline.py         # Main execution coordinator and Windows environment setup
contract.yaml       # Strict schema contract specification for fact table
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
`fact_service_delivery` contains one row per data element, facility, period, and category option combo after deduplication. It is partitioned by `health_area` and `year_month` for query optimization.

### Dimensions
- `dim_data_element`: Unique data elements with metadata (value type, aggregation type, health area).
- `dim_org_unit`: Organizational units with hierarchy (facility, district, region, country).
- `dim_period`: Time dimensions (period, year, month, quarter, start/end dates).
- `dim_program`: Programs with expected data elements per country and health area.

## Key Design Decisions & Rubric Compliance

1. **Windows Local Stabilization**:
   Configured PySpark with Hadoop environment settings and JVM compatibility options (`--add-opens=java.base/java.nio=ALL-UNNAMED`, etc.) directly inside Spark session configuration, preventing the common Windows `InaccessibleObjectException` crashes when running on Java 17.

2. **Dynamic Hierarchy Traversal**:
   Replaced rigid array-indexing logic with a dynamic traversal flow. The hierarchy is flattened by exploding the path strings, joining them back to the raw hierarchy to resolve names and levels, and then dynamically pivoting by `ancestor_level`. This handles any arbitrary organisational unit depth without hardcoded index fallbacks.

3. **Window Function Analytics**:
   Per rubric guidelines, analytics functions (`country_reporting_rate` and `top_underreporting_facilities`) strictly utilize PySpark `Window` functions to compute partition aggregates (like `expected_facilities` and `reported_facilities` counts) rather than standard shuffly `groupBy` operators.

4. **Data Contract Validation (Bonus Challenge)**:
   Implemented a strict runtime schema validator. Prior to writing the `fact_service_delivery` dataset, `models/contract.py` validates the schema column-by-column against `contract.yaml` (asserting data types, column names, and nullability limits). The pipeline halts immediately and raises `DataContractError` upon contract violation to protect warehouse integrity.

## Output Folders

The pipeline produces:
- `output/warehouse/` - Parquet fact and dimension tables.
- `output/analytics/` - CSV analytics tables (MoM change, rolling averages, reporting rates, underreporting facilities).
- `output/cross_country/` - CSV cross-country comparisons (volumes, completeness comparisons, coverage matrix, completeness flags).
- `output/dq/` - Parquet logs for data quality tracking (late reported, missing values, explicit zeros, unresolved UIDs).
- `output/quarantine/` - Parquet quarantined malformed JSON rows.
- `output/logs/` - Text logs detailing step execution.

---
*Created as part of the PSI Associate Data Engineer assessment.*
