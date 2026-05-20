from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def build_dim_data_element(data_elements: DataFrame) -> DataFrame:
    return data_elements.dropDuplicates(["data_element_uid"])


def build_dim_org_unit(facility_hierarchy: DataFrame) -> DataFrame:
    return facility_hierarchy.dropDuplicates(["org_unit_uid"])


def build_dim_program(programs: DataFrame) -> DataFrame:
    return programs.dropDuplicates(["program_uid"])


def build_dim_period(df: DataFrame) -> DataFrame:
    return (
        df
        .select("period", "year_month", "year", "month", "quarter", "period_start_date", "period_end_date")
        .dropDuplicates(["period"])
    )


def build_fact_service_delivery(df: DataFrame) -> DataFrame:
    return (
        df
        .select(
            "data_element_uid",
            "data_element_name",
            "category_option_combo_uid",
            "category_option_combo_name",
            "program_uid",
            "program_name",
            "health_area",
            "org_unit_uid",
            "facility_name",
            "district_name",
            "region_name",
            "country_name",
            "period",
            "year_month",
            "year",
            "month",
            "quarter",
            "raw_value",
            "numeric_value",
            "boolean_value",
            "stored_by",
            "created_at",
            "last_updated_at",
            "is_explicit_zero",
            "is_missing_value",
            "is_late_reported",
            "reported_indicator_count",
            "expected_indicator_count",
            "completeness_score",
        )
    )


def write_dimensional_outputs(
    fact: DataFrame,
    dim_data_element: DataFrame,
    dim_org_unit: DataFrame,
    dim_period: DataFrame,
    dim_program: DataFrame,
    output_dir: str,
) -> None:
    fact = fact.withColumn("health_area", F.coalesce(F.col("health_area"), F.lit("Unknown")))
    fact.write.mode("overwrite").partitionBy("health_area", "year_month").parquet(
        f"{output_dir}/warehouse/fact_service_delivery"
    )

    dim_data_element.write.mode("overwrite").parquet(f"{output_dir}/warehouse/dim_data_element")
    dim_org_unit.write.mode("overwrite").parquet(f"{output_dir}/warehouse/dim_org_unit")
    dim_period.write.mode("overwrite").parquet(f"{output_dir}/warehouse/dim_period")
    dim_program.write.mode("overwrite").parquet(f"{output_dir}/warehouse/dim_program")
