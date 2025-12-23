import pandas as pd
from typing import Dict, Any

def table_to_df(data_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Converts a dictionary representation of a table into pandas DataFrame.
    """
    records = []
    for row in data_dict["data"]:
        record = {}
        for i, (header, _) in enumerate(data_dict["header"]):
            text = row[i][0].strip()
            record[header] = text
        records.append(record)
    return pd.DataFrame(records)