from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def add_period_columns(df: DataFrame) -> DataFrame:
    return (
        df
        .withColumn("year", F.substring("period", 1, 4).cast("int"))
        .withColumn("month", F.substring("period", 5, 2).cast("int"))
        .withColumn("year_month", F.col("period"))
        .withColumn(
            "period_start_date",
            F.to_date(F.concat_ws("-", F.col("year"), F.lpad(F.col("month"), 2, "0"), F.lit("01"))),
        )
        .withColumn("period_end_date", F.last_day("period_start_date"))
        .withColumn("quarter", F.concat(F.col("year"), F.lit("Q"), F.quarter("period_start_date")))
    )


def cast_values_and_flags(df: DataFrame) -> DataFrame:
    return (
        df
        .withColumn(
            "numeric_value",
            F.when(F.col("raw_value").isNull(), F.lit(None).cast("double"))
             .when(F.col("value_type").isin("INTEGER_ZERO_OR_POSITIVE", "INTEGER", "INTEGER_POSITIVE"), F.col("raw_value").cast("double"))
             .when(F.col("value_type").isin("NUMBER", "PERCENTAGE"), F.col("raw_value").cast("double"))
             .otherwise(F.lit(None).cast("double")),
        )
        .withColumn(
            "boolean_value",
            F.when(F.col("value_type") == "BOOLEAN", F.col("raw_value").cast("boolean"))
             .otherwise(F.lit(None).cast("boolean")),
        )
        .withColumn("is_explicit_zero", F.col("raw_value") == F.lit("0"))
        .withColumn("is_missing_value", F.col("raw_value").isNull())
        .withColumn(
            "is_late_reported",
            F.datediff(F.col("last_updated_at"), F.col("period_end_date")) > 60,
        )
    )


def deduplicate_latest(df: DataFrame) -> DataFrame:
    key_cols = [
        "data_element_uid",
        "period",
        "org_unit_uid",
        "category_option_combo_uid",
    ]

    w = Window.partitionBy(*key_cols).orderBy(F.col("last_updated_at").desc_nulls_last())

    return (
        df
        .withColumn("dedupe_rank", F.row_number().over(w))
        .filter(F.col("dedupe_rank") == 1)
        .drop("dedupe_rank")
    )


def flatten_programs(programs_raw: DataFrame) -> DataFrame:
    return (
        programs_raw
        .select(F.explode_outer("programs").alias("program"))
        .select(
            F.col("program.id").alias("program_uid"),
            F.col("program.name").alias("program_name"),
            F.col("program.healthArea").alias("program_health_area"),
            F.col("program.country").alias("program_country"),
            F.col("program.reportingFrequency").alias("reporting_frequency"),
            F.col("program.dataElements").alias("program_data_elements"),
        )
    )


def explode_program_data_elements(programs: DataFrame) -> DataFrame:
    return (
        programs
        .select(
            "program_uid",
            "program_name",
            "program_health_area",
            "program_country",
            F.explode_outer("program_data_elements").alias("data_element_uid"),
        )
    )


def add_program_mapping(df: DataFrame, program_elements: DataFrame) -> DataFrame:
    return df.join(
        program_elements,
        (
            (df.data_element_uid == program_elements.data_element_uid)
            & (df.country_name == program_elements.program_country)
            & (df.health_area == program_elements.program_health_area)
        ),
        "left",
    ).drop(program_elements.data_element_uid)


def compute_completeness(df: DataFrame, program_elements: DataFrame) -> DataFrame:
    expected = (
        program_elements
        .groupBy(
            F.col("program_country").alias("country_name"),
            F.col("program_health_area").alias("health_area"),
        )
        .agg(F.countDistinct("data_element_uid").alias("expected_indicator_count"))
    )

    reported = (
        df
        .filter(F.col("data_element_name").isNotNull())
        .groupBy("country_name", "health_area", "org_unit_uid", "period")
        .agg(F.countDistinct("data_element_uid").alias("reported_indicator_count"))
    )

    return (
        reported
        .join(expected, on=["country_name", "health_area"], how="left")
        .withColumn(
            "completeness_score",
            F.when(F.col("expected_indicator_count") > 0,
                   F.col("reported_indicator_count") / F.col("expected_indicator_count"))
             .otherwise(F.lit(None).cast("double")),
        )
    )


def attach_completeness(df: DataFrame, completeness: DataFrame) -> DataFrame:
    return df.join(
        completeness,
        on=["country_name", "health_area", "org_unit_uid", "period"],
        how="left",
    )
