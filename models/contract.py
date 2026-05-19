import yaml
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

class DataContractError(Exception):
    pass

def validate_contract(df: DataFrame, contract_path: str) -> None:
    """
    Validates a PySpark DataFrame against a YAML contract schema.
    Raises DataContractError if constraints are violated.
    """
    with open(contract_path, 'r') as f:
        contract = yaml.safe_load(f)
        
    schema_cols = contract.get("schema", {}).get("columns", [])
    if not schema_cols:
        raise DataContractError("Contract contains no columns definition.")
        
    # Check column existence and types
    df_types = dict(df.dtypes)
    for col in schema_cols:
        col_name = col["name"]
        expected_type = col["type"]
        
        if col_name not in df_types:
            raise DataContractError(f"Missing required column: {col_name}")
            
        actual_type = df_types[col_name]
        
        # Pyspark dtype mappings
        type_mapping = {
            "integer": "int",
            "long": "bigint",
            "double": "double",
            "string": "string",
            "boolean": "boolean",
            "timestamp": "timestamp"
        }
        
        expected_mapped = type_mapping.get(expected_type, expected_type)
        if actual_type != expected_mapped:
            raise DataContractError(f"Column type mismatch for {col_name}. Expected {expected_mapped}, got {actual_type}.")
            
        # Check nullable constraint
        is_nullable = col.get("nullable", True)
        if not is_nullable:
            null_count = df.filter(F.col(col_name).isNull()).count()
            if null_count > 0:
                raise DataContractError(f"Column {col_name} is marked nullable: false, but contains {null_count} null records.")
                
    print("Data contract validation successful.")
