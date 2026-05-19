from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def global_volumes_by_quarter(fact: DataFrame) -> DataFrame:
    return (
        fact
        .groupBy("health_area", "quarter")
        .agg(F.sum("numeric_value").alias("total_service_volume"))
        .orderBy("health_area", "quarter")
    )


def completeness_comparison(fact: DataFrame) -> DataFrame:
    return (
        fact
        .groupBy("country_name", "health_area", "period")
        .agg(F.avg("completeness_score").alias("avg_completeness_score"))
        .orderBy("country_name", "health_area", "period")
    )


def data_element_coverage_matrix(fact: DataFrame) -> DataFrame:
    coverage = (
        fact
        .select("country_name", "data_element_name")
        .dropDuplicates()
        .withColumn("reported", F.lit(1))
    )

    return (
        coverage
        .groupBy("data_element_name")
        .pivot("country_name")
        .agg(F.max("reported"))
        .fillna(0)
    )


def low_completeness_flags(completeness: DataFrame) -> DataFrame:
    base = (
        completeness
        .withColumn("is_low", F.col("avg_completeness_score") < 0.80)
    )

    return (
        base
        .groupBy("country_name", "health_area")
        .agg(
            F.sum(F.col("is_low").cast("int")).alias("low_completeness_periods"),
            F.avg("avg_completeness_score").alias("overall_avg_completeness"),
        )
        .filter(F.col("low_completeness_periods") >= 3)
    )


def write_cross_country_outputs(
    volumes: DataFrame,
    completeness: DataFrame,
    coverage: DataFrame,
    low_flags: DataFrame,
    output_dir: str,
) -> None:
    volumes.write.mode("overwrite").csv(f"{output_dir}/cross_country/global_volumes_by_quarter", header=True)
    completeness.write.mode("overwrite").csv(f"{output_dir}/cross_country/completeness_comparison", header=True)
    coverage.write.mode("overwrite").csv(f"{output_dir}/cross_country/data_element_coverage_matrix", header=True)
    low_flags.write.mode("overwrite").csv(f"{output_dir}/cross_country/low_completeness_flags", header=True)
