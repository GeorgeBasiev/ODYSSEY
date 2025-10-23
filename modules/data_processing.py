import pandas as pd
import spacy
from typing import Dict, List, Any


def table_to_df(data_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Converts a dictionary representation of a table into a pandas DataFrame.
    Each cell contains both the text and an optional link.
    """
    records = []
    for row in data_dict["data"]:
        record = {}
        for i, (header, _) in enumerate(data_dict["header"]):
            text = row[i][0].strip()
            link = row[i][1][0] if row[i][1] else None
            record[header] = text
            record[f"{header}_link"] = link
        records.append(record)

    return pd.DataFrame(records)


def extract_entities_by_doc(
    passages: Dict[str, str],
    model: str = "en_core_web_trf",
    keep_duplicates: bool = True,
) -> Dict[str, List[str]]:
    """
    Extracts named entities from a dictionary of passages using spaCy.
    """
    nlp = spacy.load(model)

    out = {}
    for doc_id, text in passages.items():
        doc = nlp(text or "")
        ents = []
        for e in doc.ents:
            ents.append(e.text.strip())
        if not keep_duplicates:
            seen, uniq = set(), []
            for x in ents:
                if x not in seen:
                    seen.add(x)
                    uniq.append(x)
            ents = uniq
        out[doc_id] = ents
    return out