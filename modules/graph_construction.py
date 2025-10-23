import pandas as pd
import networkx as nx
import itertools
from typing import Dict, List, Any, Optional


def build_hybrid_graph(
    table_df: pd.DataFrame,
    doc_entities: Dict[str, List[str]],
    headers: List[str],
) -> nx.Graph:
    """
    Builds a hybrid graph connecting table headers, cells, and document entities.
    """
    G = nx.Graph()

    # Add header nodes
    for col in headers:
        G.add_node(("header", col), kind="header", col=col)

    n_rows = len(table_df)
    link_suffix = "_link"
    for r in range(n_rows):
        for col in headers:
            raw = table_df.loc[r, col] if col in table_df.columns else ""
            text = "" if pd.isna(raw) else str(raw)

            link_col = f"{col}{link_suffix}"
            link_val = table_df.loc[r, link_col] if link_col in table_df.columns else None
            link = None if (link_val is None or (isinstance(link_val, float) and pd.isna(link_val)) or str(link_val).strip()=="") else str(link_val).strip()

            cell = ("cell", r, col)
            G.add_node(cell, kind="cell", row=r, col=col, text=text, link=link)
            G.add_edge(cell, ("header", col), kind="cell-header")

    # Add row edges connecting cells within the same row
    for r in range(n_rows):
        row_cells = [("cell", r, col) for col in headers]
        for u, v in itertools.combinations(row_cells, 2):
            if G.has_node(u) and G.has_node(v):
                G.add_edge(u, v, kind="row")

    def add_ent(ent_text: str) -> Optional[tuple]:
        ent_text = str(ent_text).strip()
        if not ent_text:
            return None
        node = ("ent", ent_text)
        if not G.has_node(node):
            G.add_node(node, kind="ent", text=ent_text)
        return node

    # Add edges between cells and their document entities
    for node, attr in list(G.nodes(data=True)):
        if attr.get("kind") != "cell":
            continue
        link = attr.get("link")
        if not link:
            continue

        ents = doc_entities.get(link, [])
        if not ents:
            continue

        for ent in ents:
            ent_node = add_ent(ent)
            if ent_node:
                G.add_edge(node, ent_node, kind="cell-ent")

    return G