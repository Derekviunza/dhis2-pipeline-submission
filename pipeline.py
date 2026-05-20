import argparse
import logging
import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from models.ingestion import (
    load_raw_files,
    flatten_data_values,
    quarantine_malformed_data_values,
    valid_data_values,
)
from models.metadata import (
    flatten_metadata,
    resolve_metadata,
    unresolved_data_elements,
    unresolved_category_option_combos,
)
from models.org_units import (
    flatten_org_units,
    build_facility_hierarchy,
    resolve_org_units,
    unresolved_org_units,
)
from models.quality import (
    add_period_columns,
    cast_values_and_flags,
    deduplicate_latest,
    flatten_programs,
    explode_program_data_elements,
    add_program_mapping,
    compute_completeness,
    attach_completeness,
)
from models.dimensional import (
    build_dim_data_element,
    build_dim_org_unit,
    build_dim_period,
    build_dim_program,
    build_fact_service_delivery,
    write_dimensional_outputs,
)
from models.analytics import (
    month_over_month_change,
    rolling_three_month_average,
    country_reporting_rate,
    top_underreporting_facilities,
    write_analytics_outputs,
)
from models.aggregation import (
    global_volumes_by_quarter,
    completeness_comparison,
    data_element_coverage_matrix,
    low_completeness_flags,
    write_cross_country_outputs,
)


def parse_args():
    parser = argparse.ArgumentParser(description="DHIS2 Health Data Pipeline")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("dhis2-health-data-pipeline")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def setup_logging(output_dir: str) -> None:
    os.makedirs(f"{output_dir}/logs", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"{output_dir}/logs/pipeline.log"),
        ],
    )


def log_count(name: str, df) -> int:
    count = df.count()
    logging.info("%s row count: %s", name, count)
    return count


def main():
    args = parse_args()
    setup_logging(args.output_dir)

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    try:
        logging.info("Task 01: Loading raw files")
        raw = load_raw_files(spark, args.data_dir)

        data_values_flat = flatten_data_values(raw["data_values_raw"])
        malformed = quarantine_malformed_data_values(data_values_flat)
        data_values_valid = valid_data_values(data_values_flat)

        malformed.write.mode("overwrite").parquet(f"{args.output_dir}/quarantine/malformed_data_values")

        total_rows = log_count("flattened_data_values", data_values_flat)
        malformed_rows = log_count("malformed_data_values", malformed)
        log_count("valid_data_values", data_values_valid)

        quarantine_rate = malformed_rows / total_rows if total_rows else 1
        if quarantine_rate > 0.10:
            logging.error("Critical DQ failure: quarantine rate above 10%%")
            sys.exit(2)

        logging.info("Task 02: Resolving metadata UIDs")
        data_elements, category_option_combos = flatten_metadata(raw["metadata_raw"])

        bad_de = unresolved_data_elements(data_values_valid, data_elements)
        bad_coc = unresolved_category_option_combos(data_values_valid, category_option_combos)

        bad_de.write.mode("overwrite").parquet(f"{args.output_dir}/dq/unresolved_data_elements")
        bad_coc.write.mode("overwrite").parquet(f"{args.output_dir}/dq/unresolved_category_option_combos")

        log_count("unresolved_data_elements", bad_de)
        log_count("unresolved_category_option_combos", bad_coc)

        resolved_metadata = resolve_metadata(data_values_valid, data_elements, category_option_combos)

        logging.info("Task 03: Resolving org unit hierarchy")
        org_units = flatten_org_units(raw["org_units_raw"])
        facility_hierarchy = build_facility_hierarchy(org_units)

        bad_ou = unresolved_org_units(resolved_metadata, facility_hierarchy)
        bad_ou.write.mode("overwrite").parquet(f"{args.output_dir}/dq/unresolved_org_units")
        log_count("unresolved_org_units", bad_ou)

        resolved_orgs = resolve_org_units(resolved_metadata, facility_hierarchy)

        logging.info("Task 04: Adding DQ flags and completeness")
        programs = flatten_programs(raw["programs_raw"])
        program_elements = explode_program_data_elements(programs)

        enriched = (
            resolved_orgs
            .transform(add_period_columns)
            .transform(cast_values_and_flags)
            .transform(deduplicate_latest)
        )

        enriched = add_program_mapping(enriched, program_elements)

        completeness = compute_completeness(enriched, program_elements)
        enriched = attach_completeness(enriched, completeness)

        enriched.filter(F.col("is_late_reported")).write.mode("overwrite").parquet(
            f"{args.output_dir}/dq/late_reported"
        )
        enriched.filter(F.col("is_missing_value")).write.mode("overwrite").parquet(
            f"{args.output_dir}/dq/missing_values"
        )
        enriched.filter(F.col("is_explicit_zero")).write.mode("overwrite").parquet(
            f"{args.output_dir}/dq/explicit_zeros"
        )

        log_count("enriched_data_values", enriched)

        logging.info("Task 05: Building dimensional model")
        dim_data_element = build_dim_data_element(data_elements)
        dim_org_unit = build_dim_org_unit(facility_hierarchy)
        dim_period = build_dim_period(enriched)
        dim_program = build_dim_program(programs)
        fact = build_fact_service_delivery(enriched)

        fact_count = log_count("fact_service_delivery", fact)
        if fact_count == 0:
            logging.error("Critical DQ failure: fact table has zero rows")
            sys.exit(3)

        write_dimensional_outputs(
            fact,
            dim_data_element,
            dim_org_unit,
            dim_period,
            dim_program,
            args.output_dir,
        )

        logging.info("Task 06: Building program analytics")
        mom = month_over_month_change(fact)
        rolling = rolling_three_month_average(fact)
        reporting_rate = country_reporting_rate(fact)
        underreporting = top_underreporting_facilities(fact)

        write_analytics_outputs(
            mom,
            rolling,
            reporting_rate,
            underreporting,
            args.output_dir,
        )

        logging.info("Task 07: Building cross-country aggregations")
        volumes = global_volumes_by_quarter(fact)
        completeness_comp = completeness_comparison(fact)
        coverage = data_element_coverage_matrix(fact)
        low_flags = low_completeness_flags(completeness_comp)

        write_cross_country_outputs(
            volumes,
            completeness_comp,
            coverage,
            low_flags,
            args.output_dir,
        )

        logging.info("Pipeline completed successfully")

    finally:
        spark.stop()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        logging.error("Pipeline failed with exception:\n%s", traceback.format_exc())
        sys.exit(1)
