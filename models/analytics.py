from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def month_over_month_change(fact: DataFrame) -> DataFrame:
    monthly = (
        fact
        .groupBy("health_area", "district_name", "data_element_name", "year_month")
        .agg(F.sum("numeric_value").alias("monthly_value"))
    )

    w = Window.partitionBy("health_area", "district_name", "data_element_name").orderBy("year_month")

    return (
        monthly
        .withColumn("previous_month_value", F.lag("monthly_value").over(w))
        .withColumn(
            "mom_percent_change",
            F.when(F.col("previous_month_value") > 0,
                   ((F.col("monthly_value") - F.col("previous_month_value")) / F.col("previous_month_value")) * 100)
             .otherwise(F.lit(None).cast("double")),
        )
    )


def rolling_three_month_average(fact: DataFrame) -> DataFrame:
    monthly = (
        fact
        .groupBy("country_name", "facility_name", "org_unit_uid", "data_element_name", "year_month")
        .agg(F.sum("numeric_value").alias("monthly_value"))
    )

    w = (
        Window
        .partitionBy("org_unit_uid", "data_element_name")
        .orderBy("year_month")
        .rowsBetween(-2, 0)
    )

    return monthly.withColumn(
        "rolling_3_month_avg",
        F.avg("monthly_value").over(w),
    )


def country_reporting_rate(fact: DataFrame) -> DataFrame:
    expected = fact.groupBy("country_name").agg(F.countDistinct("org_unit_uid").alias("expected_facilities"))
    reported = fact.filter(F.col("raw_value").isNotNull()).groupBy("country_name", "period") \
        .agg(F.countDistinct("org_unit_uid").alias("reported_facilities"))

    return (
        reported.join(expected, "country_name")
        .withColumn(
            "reporting_rate",
            F.when(F.col("expected_facilities") > 0, F.col("reported_facilities") / F.col("expected_facilities"))
             .otherwise(F.lit(None).cast("double"))
        )
        .select("country_name", "period", "expected_facilities", "reported_facilities", "reporting_rate")
    )


def top_underreporting_facilities(fact: DataFrame) -> DataFrame:
    w_period = Window.partitionBy("health_area", "country_name", "facility_name", "org_unit_uid", "period")
    
    # Sum numeric values over the period
    period_total = F.sum(F.coalesce("numeric_value", F.lit(0))).over(w_period)
    
    # Flag if a facility had zero data for a given period
    has_zero_data = (period_total == 0).cast("int")

    # To calculate total zero periods per facility using window functions:
    # First, we must ensure we only count each period once per facility by taking distinct period rows.
    distinct_periods = (
        fact
        .withColumn("has_zero_data", has_zero_data)
        .select("health_area", "country_name", "facility_name", "org_unit_uid", "period", "has_zero_data")
        .dropDuplicates()
    )

    w_facility = Window.partitionBy("health_area", "country_name", "facility_name", "org_unit_uid")
    zero_data_periods = F.sum("has_zero_data").over(w_facility)

    w_rank = Window.partitionBy("health_area").orderBy(F.col("zero_data_periods").desc())

    return (
        distinct_periods
        .withColumn("zero_data_periods", zero_data_periods)
        .select("health_area", "country_name", "facility_name", "org_unit_uid", "zero_data_periods")
        .dropDuplicates()
        .withColumn("rank", F.rank().over(w_rank))
        .filter(F.col("rank") <= 5)
    )


def detect_anomalies(fact: DataFrame) -> DataFrame:
    """
    Flags facilities where any indicator value is more than 3 standard deviations
    from that facility's own 12-month rolling mean.
    """
    w_12m = (
        Window
        .partitionBy("org_unit_uid", "data_element_uid", "category_option_combo_uid")
        .orderBy("year_month")
        .rowsBetween(-11, 0)
    )

    df_stats = (
        fact
        .withColumn("rolling_mean", F.avg("numeric_value").over(w_12m))
        .withColumn("rolling_stddev", F.stddev("numeric_value").over(w_12m))
    )

    anomalies = (
        df_stats
        .filter(
            F.col("rolling_stddev").isNotNull()
            & (F.col("rolling_stddev") > 0)
            & (F.abs(F.col("numeric_value") - F.col("rolling_mean")) > 3 * F.col("rolling_stddev"))
        )
        .withColumn("z_score", (F.col("numeric_value") - F.col("rolling_mean")) / F.col("rolling_stddev"))
        .select(
            "country_name",
            "region_name",
            "district_name",
            "facility_name",
            "org_unit_uid",
            "data_element_name",
            "data_element_uid",
            "category_option_combo_name",
            "period",
            "numeric_value",
            "rolling_mean",
            "rolling_stddev",
            "z_score"
        )
    )
    return anomalies


def write_analytics_outputs(
    mom: DataFrame,
    rolling: DataFrame,
    reporting_rate: DataFrame,
    underreporting: DataFrame,
    anomalies: DataFrame,
    output_dir: str,
) -> None:
    mom.write.mode("overwrite").csv(f"{output_dir}/analytics/month_over_month_change", header=True)
    rolling.write.mode("overwrite").csv(f"{output_dir}/analytics/rolling_3_month_average", header=True)
    reporting_rate.write.mode("overwrite").csv(f"{output_dir}/analytics/country_reporting_rate", header=True)
    underreporting.write.mode("overwrite").csv(f"{output_dir}/analytics/top_underreporting_facilities", header=True)
    anomalies.write.mode("overwrite").csv(f"{output_dir}/analytics/anomalies", header=True)
