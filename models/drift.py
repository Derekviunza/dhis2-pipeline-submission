import json
from typing import Dict, List, Any


def detect_metadata_drift(current_metadata_path: str, reference_metadata_path: str, output_path: str) -> None:
    """
    Compares current metadata.json to a reference snapshot and reports drift.
    Detects data elements that have been added, removed, or renamed between exports.
    """
    with open(current_metadata_path, 'r') as f:
        current = json.load(f)
    
    with open(reference_metadata_path, 'r') as f:
        reference = json.load(f)
    
    current_des = {de["id"]: de for de in current.get("dataElements", [])}
    reference_des = {de["id"]: de for de in reference.get("dataElements", [])}
    
    current_ids = set(current_des.keys())
    reference_ids = set(reference_des.keys())
    
    added_ids = current_ids - reference_ids
    removed_ids = reference_ids - current_ids
    common_ids = current_ids & reference_ids
    
    renamed = []
    for de_id in common_ids:
        if current_des[de_id].get("name") != reference_des[de_id].get("name"):
            renamed.append({
                "id": de_id,
                "old_name": reference_des[de_id].get("name"),
                "new_name": current_des[de_id].get("name")
            })
    
    drift_report = {
        "added_data_elements": [
            {"id": de_id, "name": current_des[de_id].get("name")} for de_id in sorted(added_ids)
        ],
        "removed_data_elements": [
            {"id": de_id, "name": reference_des[de_id].get("name")} for de_id in sorted(removed_ids)
        ],
        "renamed_data_elements": renamed,
        "summary": {
            "total_added": len(added_ids),
            "total_removed": len(removed_ids),
            "total_renamed": len(renamed)
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(drift_report, f, indent=2)
