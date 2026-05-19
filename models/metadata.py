from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import broadcast


def flatten_metadata(metadata_raw: DataFrame) -> tuple[DataFrame, DataFrame]:
    data_elements = (
        metadata_raw
        .select(F.explode_outer("dataElements").alias("de"))
        .select(
            F.col("de.id").alias("data_element_uid"),
            F.col("de.name").alias("data_element_name"),
            F.col("de.valueType").alias("value_type"),
            F.col("de.domainType").alias("domain_type"),
            F.col("de.aggregationType").alias("aggregation_type"),
            F.col("de.zeroIsSignificant").alias("zero_is_significant"),
            F.col("de.categoryCombo.id").alias("category_combo_uid"),
            F.col("de.categoryCombo.name").alias("category_combo_name"),
            F.col("de.dataElementGroups")[0]["name"].alias("health_area"),
        )
    )

    category_option_combos = (
        metadata_raw
        .select(F.explode_outer("categoryOptionCombos").alias("coc"))
        .select(
            F.col("coc.id").alias("category_option_combo_uid"),
            F.col("coc.name").alias("category_option_combo_name"),
        )
    )

    return data_elements, category_option_combos


def resolve_metadata(
    data_values: DataFrame,
    data_elements: DataFrame,
    category_option_combos: DataFrame,
) -> DataFrame:
    return (
        data_values
        .join(broadcast(data_elements), on="data_element_uid", how="left")
        .join(broadcast(category_option_combos), on="category_option_combo_uid", how="left")
    )


def unresolved_data_elements(data_values: DataFrame, data_elements: DataFrame) -> DataFrame:
    return data_values.join(
        broadcast(data_elements.select("data_element_uid")),
        on="data_element_uid",
        how="left_anti",
    )


def unresolved_category_option_combos(data_values: DataFrame, category_option_combos: DataFrame) -> DataFrame:
    return data_values.join(
        broadcast(category_option_combos.select("category_option_combo_uid")),
        on="category_option_combo_uid",
        how="left_anti",
    )
