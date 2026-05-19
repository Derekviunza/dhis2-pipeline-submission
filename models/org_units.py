from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import broadcast


def flatten_org_units(org_units_raw: DataFrame) -> DataFrame:
    return (
        org_units_raw
        .select(F.explode_outer("organisationUnits").alias("ou"))
        .select(
            F.col("ou.id").alias("org_unit_uid"),
            F.col("ou.name").alias("org_unit_name"),
            F.col("ou.level").alias("org_unit_level"),
            F.col("ou.path").alias("org_unit_path"),
            F.col("ou.parent.id").alias("parent_org_unit_uid"),
            F.concat_ws(",", F.expr("transform(ou.groups, x -> x.name)")).alias("org_unit_groups"),
        )
    )


def build_facility_hierarchy(org_units: DataFrame) -> DataFrame:
    # Get just the level 4 facilities to start
    facilities = (
        org_units
        .filter(F.col("org_unit_level") == 4)
        .select(
            "org_unit_uid",
            "org_unit_name",
            "org_unit_level",
            "org_unit_groups",
            F.split(F.regexp_replace("org_unit_path", "^/", ""), "/").alias("path_parts")
        )
    )

    # Explode the path to map each facility to ALL its ancestors dynamically
    exploded = facilities.select(
        "org_unit_uid",
        "org_unit_name",
        "org_unit_level",
        "org_unit_groups",
        F.explode("path_parts").alias("ancestor_uid")
    )

    # Prepare a broadcastable lookup for all ancestors
    ancestors = org_units.select(
        F.col("org_unit_uid").alias("ancestor_uid"),
        F.col("org_unit_name").alias("ancestor_name"),
        F.col("org_unit_level").alias("ancestor_level")
    )

    # Join the exploded paths with the ancestor lookup to resolve levels
    hierarchy = exploded.join(broadcast(ancestors), on="ancestor_uid", how="inner")

    # Pivot by ancestor_level to dynamically flatten the hierarchy regardless of depth
    pivoted = (
        hierarchy
        .groupBy("org_unit_uid", "org_unit_name", "org_unit_level", "org_unit_groups")
        .pivot("ancestor_level")
        .agg(F.first("ancestor_name"))
    )

    # Safely rename standard hierarchy levels, falling back to null if a level didn't exist
    for level, col_name in [("1", "country_name"), ("2", "region_name"), ("3", "district_name")]:
        if level in pivoted.columns:
            pivoted = pivoted.withColumnRenamed(level, col_name)
        else:
            pivoted = pivoted.withColumn(col_name, F.lit(None).cast("string"))

    return (
        pivoted
        .withColumnRenamed("org_unit_name", "facility_name")
        .withColumnRenamed("org_unit_level", "facility_level")
        .withColumnRenamed("org_unit_groups", "facility_groups")
    )


def resolve_org_units(data_values: DataFrame, facility_hierarchy: DataFrame) -> DataFrame:
    return data_values.join(
        broadcast(facility_hierarchy),
        data_values.org_unit_uid == facility_hierarchy.org_unit_uid,
        "left",
    ).drop(facility_hierarchy.org_unit_uid)


def unresolved_org_units(data_values: DataFrame, facility_hierarchy: DataFrame) -> DataFrame:
    return data_values.join(
        broadcast(facility_hierarchy.select("org_unit_uid")),
        on="org_unit_uid",
        how="left_anti",
    )
