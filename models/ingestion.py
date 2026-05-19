from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


def data_values_schema() -> T.StructType:
    return T.StructType([
        T.StructField("responseType", T.StringType(), True),
        T.StructField("version", T.StringType(), True),
        T.StructField("exportDate", T.StringType(), True),
        T.StructField("dataValues", T.ArrayType(T.StructType([
            T.StructField("dataElement", T.StringType(), True),
            T.StructField("period", T.StringType(), True),
            T.StructField("orgUnit", T.StringType(), True),
            T.StructField("categoryOptionCombo", T.StringType(), True),
            T.StructField("attributeOptionCombo", T.StringType(), True),
            T.StructField("value", T.StringType(), True),
            T.StructField("storedBy", T.StringType(), True),
            T.StructField("created", T.StringType(), True),
            T.StructField("lastUpdated", T.StringType(), True),
            T.StructField("followup", T.StringType(), True),
        ])), True),
    ])


def metadata_schema() -> T.StructType:
    return T.StructType([
        T.StructField("date", T.StringType(), True),
        T.StructField("version", T.StringType(), True),
        T.StructField("dataElements", T.ArrayType(T.StructType([
            T.StructField("id", T.StringType(), True),
            T.StructField("name", T.StringType(), True),
            T.StructField("shortName", T.StringType(), True),
            T.StructField("code", T.StringType(), True),
            T.StructField("valueType", T.StringType(), True),
            T.StructField("domainType", T.StringType(), True),
            T.StructField("aggregationType", T.StringType(), True),
            T.StructField("zeroIsSignificant", T.BooleanType(), True),
            T.StructField("categoryCombo", T.StructType([
                T.StructField("id", T.StringType(), True),
                T.StructField("name", T.StringType(), True),
            ]), True),
            T.StructField("dataElementGroups", T.ArrayType(T.StructType([
                T.StructField("id", T.StringType(), True),
                T.StructField("name", T.StringType(), True),
            ])), True),
            T.StructField("created", T.StringType(), True),
            T.StructField("lastUpdated", T.StringType(), True),
        ])), True),
        T.StructField("categoryOptionCombos", T.ArrayType(T.StructType([
            T.StructField("id", T.StringType(), True),
            T.StructField("name", T.StringType(), True),
            T.StructField("created", T.StringType(), True),
            T.StructField("lastUpdated", T.StringType(), True),
        ])), True),
    ])


def org_units_schema() -> T.StructType:
    return T.StructType([
        T.StructField("date", T.StringType(), True),
        T.StructField("version", T.StringType(), True),
        T.StructField("organisationUnits", T.ArrayType(T.StructType([
            T.StructField("id", T.StringType(), True),
            T.StructField("name", T.StringType(), True),
            T.StructField("shortName", T.StringType(), True),
            T.StructField("code", T.StringType(), True),
            T.StructField("level", T.IntegerType(), True),
            T.StructField("path", T.StringType(), True),
            T.StructField("parent", T.StructType([
                T.StructField("id", T.StringType(), True),
                T.StructField("name", T.StringType(), True),
            ]), True),
            T.StructField("groups", T.ArrayType(T.StructType([
                T.StructField("id", T.StringType(), True),
                T.StructField("name", T.StringType(), True),
            ])), True),
            T.StructField("created", T.StringType(), True),
            T.StructField("lastUpdated", T.StringType(), True),
        ])), True),
    ])


def programs_schema() -> T.StructType:
    return T.StructType([
        T.StructField("date", T.StringType(), True),
        T.StructField("version", T.StringType(), True),
        T.StructField("programs", T.ArrayType(T.StructType([
            T.StructField("id", T.StringType(), True),
            T.StructField("name", T.StringType(), True),
            T.StructField("shortName", T.StringType(), True),
            T.StructField("healthArea", T.StringType(), True),
            T.StructField("country", T.StringType(), True),
            T.StructField("reportingFrequency", T.StringType(), True),
            T.StructField("dataElements", T.ArrayType(T.StringType()), True),
            T.StructField("created", T.StringType(), True),
            T.StructField("lastUpdated", T.StringType(), True),
        ])), True),
    ])


def load_raw_files(spark: SparkSession, data_dir: str) -> dict[str, DataFrame]:
    metadata_raw = spark.read.schema(metadata_schema()).json(f"{data_dir}/metadata.json")
    org_units_raw = spark.read.schema(org_units_schema()).json(f"{data_dir}/org_units.json")
    programs_raw = spark.read.schema(programs_schema()).json(f"{data_dir}/programs.json")
    data_values_raw = spark.read.schema(data_values_schema()).json(f"{data_dir}/data_values.json")

    return {
        "metadata_raw": metadata_raw,
        "org_units_raw": org_units_raw,
        "programs_raw": programs_raw,
        "data_values_raw": data_values_raw,
    }


def flatten_data_values(data_values_raw: DataFrame) -> DataFrame:
    return (
        data_values_raw
        .select(F.explode_outer("dataValues").alias("dv"))
        .select(
            F.col("dv.dataElement").alias("data_element_uid"),
            F.col("dv.period").alias("period"),
            F.col("dv.orgUnit").alias("org_unit_uid"),
            F.col("dv.categoryOptionCombo").alias("category_option_combo_uid"),
            F.col("dv.attributeOptionCombo").alias("attribute_option_combo_uid"),
            F.col("dv.value").alias("raw_value"),
            F.lower(F.col("dv.storedBy")).alias("stored_by"),
            F.to_timestamp("dv.created").alias("created_at"),
            F.to_timestamp("dv.lastUpdated").alias("last_updated_at"),
            F.col("dv.followup").alias("followup"),
        )
    )


def quarantine_malformed_data_values(data_values: DataFrame) -> DataFrame:
    return data_values.filter(
        F.col("data_element_uid").isNull()
        | F.col("period").isNull()
        | F.col("org_unit_uid").isNull()
        | ~F.col("period").rlike(r"^\d{6}$")
    )


def valid_data_values(data_values: DataFrame) -> DataFrame:
    return data_values.filter(
        F.col("data_element_uid").isNotNull()
        & F.col("period").isNotNull()
        & F.col("org_unit_uid").isNotNull()
        & F.col("period").rlike(r"^\d{6}$")
    )
