import os
import sys
# Inject project root path into sys.path to enable importing local packages
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import yaml
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from models.contract import validate_contract, DataContractError

@pytest.fixture(scope="session")
def spark():
    # Configure Windows-specific Spark paths to avoid UnsatisfiedLinkError/InaccessibleObjectException
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    bin_dir = os.path.join(project_root, "bin")
    os.makedirs(bin_dir, exist_ok=True)
    
    os.environ["HADOOP_HOME"] = project_root
    path_env = os.environ.get("PATH", "")
    if bin_dir not in path_env:
        os.environ["PATH"] = f"{bin_dir}{os.path.pathsep}{path_env}"

    hadoop_home_fwd = project_root.replace("\\", "/")
    tmp_dir = os.path.join(project_root, "tmp")
    tmp_dir_fwd = tmp_dir.replace("\\", "/")

    java17_flags = (
        "--add-opens=java.base/java.lang=ALL-UNNAMED "
        "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED "
        "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED "
        "--add-opens=java.base/java.io=ALL-UNNAMED "
        "--add-opens=java.base/java.nio=ALL-UNNAMED "
        "--add-opens=java.base/java.util=ALL-UNNAMED "
        "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED "
        "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED "
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
        "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED "
        "--add-opens=java.base/sun.security.action=ALL-UNNAMED "
        "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED "
        "--add-opens=java.security.jgss/sun.security.krb5=ALL-UNNAMED"
    )

    session = (
        SparkSession.builder
        .appName("contract-unit-tests")
        .master("local[1]")
        .config("spark.driver.memory", "1g")
        .config("spark.local.dir", tmp_dir)
        .config(
            "spark.driver.extraJavaOptions",
            f"-Dhadoop.home.dir={hadoop_home_fwd} -Djava.io.tmpdir={tmp_dir_fwd} {java17_flags}"
        )
        .getOrCreate()
    )
    yield session
    session.stop()

@pytest.fixture
def dummy_contract_path(tmp_path):
    contract_data = {
        "schema": {
            "columns": [
                {"name": "health_area", "type": "string", "nullable": False},
                {"name": "numeric_value", "type": "integer", "nullable": True},
                {"name": "facility_name", "type": "string", "nullable": True}
            ]
        }
    }
    path = tmp_path / "test_contract.yaml"
    with open(path, "w") as f:
        yaml.dump(contract_data, f)
    return str(path)

def test_contract_validation_success(spark, dummy_contract_path):
    df = (
        spark.range(2)
        .withColumn("health_area", F.when(F.col("id") == 0, "HIV").otherwise("Malaria"))
        .withColumn("numeric_value", F.when(F.col("id") == 0, 42).otherwise(None).cast("int"))
        .withColumn("facility_name", F.when(F.col("id") == 0, "Clinic A").otherwise("Clinic B"))
        .select("health_area", "numeric_value", "facility_name")
    )
    
    # Should not raise any error
    validate_contract(df, dummy_contract_path)

def test_contract_validation_missing_col(spark, dummy_contract_path):
    df = (
        spark.range(1)
        .withColumn("health_area", F.lit("HIV").cast("string"))
        .withColumn("numeric_value", F.lit(42).cast("int"))
        .select("health_area", "numeric_value")
    )
    
    with pytest.raises(DataContractError) as exc_info:
        validate_contract(df, dummy_contract_path)
    assert "Missing required column: facility_name" in str(exc_info.value)

def test_contract_validation_type_mismatch(spark, dummy_contract_path):
    df = (
        spark.range(1)
        .withColumn("health_area", F.lit("HIV").cast("string"))
        .withColumn("numeric_value", F.lit("42").cast("string")) # should be integer
        .withColumn("facility_name", F.lit("Clinic A").cast("string"))
        .select("health_area", "numeric_value", "facility_name")
    )
    
    with pytest.raises(DataContractError) as exc_info:
        validate_contract(df, dummy_contract_path)
    assert "Column type mismatch for numeric_value" in str(exc_info.value)

def test_contract_validation_non_nullable_violation(spark, dummy_contract_path):
    df = (
        spark.range(1)
        .withColumn("health_area", F.lit(None).cast("string")) # health_area is null but contract says nullable: False
        .withColumn("numeric_value", F.lit(42).cast("int"))
        .withColumn("facility_name", F.lit("Clinic A").cast("string"))
        .select("health_area", "numeric_value", "facility_name")
    )
    
    with pytest.raises(DataContractError) as exc_info:
        validate_contract(df, dummy_contract_path)
    assert "health_area is marked nullable: false, but contains 1 null records" in str(exc_info.value)
