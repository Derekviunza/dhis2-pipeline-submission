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
    # Add month_index to track period order
    base = (
        completeness
        .withColumn("is_low", F.col("avg_completeness_score") < 0.80)
        .withColumn("month_index", F.row_number().over(
            Window.partitionBy("country_name", "health_area").orderBy("period")
        ))
    )

    # Calculate streaks of consecutive low completeness periods
    # A streak starts when is_low is True and the previous period was not low
    w = Window.partitionBy("country_name", "health_area").orderBy("month_index")

    streak_base = (
        base
        .withColumn("prev_is_low", F.lag("is_low").over(w))
        .withColumn("streak_start", F.when(~F.col("prev_is_low") | F.col("prev_is_low").isNull(), F.col("month_index")).otherwise(None))
        .fillna(0, subset=["streak_start"])
    )

    # Forward fill the streak_start to identify which streak each row belongs to
    w_streak = Window.partitionBy("country_name", "health_area").orderBy("month_index").rowsBetween(Window.unboundedPreceding, 0)
    streak_base = streak_base.withColumn("streak_id", F.last("streak_start", ignorenulls=True).over(w_streak))

    # Count consecutive low completeness periods per streak
    streak_counts = (
        streak_base
        .filter(F.col("is_low"))
        .groupBy("country_name", "health_area", "streak_id")
        .agg(F.count("*").alias("consecutive_low_periods"))
        .filter(F.col("consecutive_low_periods") >= 3)
        .select("country_name", "health_area")
    )

    return streak_counts


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
