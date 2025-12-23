import torch
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict, deque
import pandas as pd
import networkx as nx
import re
from typing import Dict, List, Optional, Tuple, Union
import ast

def parse_column_mapping(mapping: Union[str, List[str]]) -> List[Tuple[str, str]]:
    """
    Parse column mapping that could be a single column or multiple columns.
    Returns list of (table_name, column_name) tuples.
    """
    if mapping == "Others":
        return []
    
    if isinstance(mapping, list):
        mapping_str = " ".join(mapping)
    else:
        mapping_str = str(mapping)
    
    
    mapping_str = mapping_str.replace('"', '').replace("'", "").strip()
    
    
    table_column_pairs = []
    
    
    if mapping_str.startswith("[") and mapping_str.endswith("]"):
        try:
            items = ast.literal_eval(mapping_str)
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, str) and "." in item:
                        table, column = item.split(".", 1)
                        table_column_pairs.append((table.strip(), column.strip()))
            return table_column_pairs
        except:
            pass
    
    
    parts = re.split(r'[,\s]+', mapping_str)
    for part in parts:
        part = part.strip()
        if "." in part:
            table, column = part.split(".", 1)
            table_column_pairs.append((table.strip(), column.strip()))
    
    return table_column_pairs

def semantic_match(query: str, candidates: List[Tuple], threshold: float = 0.8) -> List[Tuple]:
    """
    Semantic matching with table.context support.
    candidates: list of (text, table_name, column_name, node)
    """
    if not candidates:
        return []
    
    model = get_instructor_model()
    instruction = "Represent the question entity for matching with table cells"
    
    candidate_texts = [c[0] for c in candidates]
    
    with torch.no_grad():
        query_emb = model.encode([[instruction, query]], convert_to_tensor=True)
        cand_embs = model.encode([[instruction, t] for t in candidate_texts], 
                                convert_to_tensor=True)
        similarities = util.cos_sim(query_emb, cand_embs)[0]
    
    matches = []
    for i, sim in enumerate(similarities):
        if sim >= threshold:
            matches.append((candidates[i], float(sim)))
    
    return matches

def prune_and_traverse_multi_table(
    G: nx.Graph,
    entity_header_mapping: Dict[str, Union[str, List[str]]],
    threshold: float = 0.8
) -> Dict[str, List[str]]:
    """
    Multi-table BFS traversal starting from matched entities.
    Supports multiple columns per entity.
    """
    
    cell_candidates = []
    
    for node, attr in G.nodes(data=True):
        if attr.get("kind") == "cell":
            text = attr.get("text", "").strip()
            if text:
                table = attr.get("table")
                col = attr.get("col")
                cell_candidates.append((text, table, col, node))
    
    start_nodes = set()
    
    
    for entity, mapping in entity_header_mapping.items():
        entity = str(entity).strip()
        if not entity:
            continue
        
        
        table_column_pairs = parse_column_mapping(mapping)
        
        if not table_column_pairs:  
            
            matches = semantic_match(entity, cell_candidates, threshold)
            if matches:
                best_match = max(matches, key=lambda x: x[1])
                start_nodes.add(best_match[0][3])  
        else:
            
            for table_name, column_name in table_column_pairs:
                
                column_cells = []
                for text, table, col, node in cell_candidates:
                    if table == table_name and col == column_name:
                        column_cells.append((text, table, col, node))
                
                if column_cells:
                    
                    found = False
                    for text, table, col, node in column_cells:
                        if text.lower() == entity.lower():
                            start_nodes.add(node)
                            found = True
                            break
                    
                    
                    if not found:
                        matches = semantic_match(entity, column_cells, threshold)
                        if matches:
                            best_match = max(matches, key=lambda x: x[1])
                            start_nodes.add(best_match[0][3])
    
    
    hop_dict = defaultdict(list)
    if not start_nodes:
        return {"1-hop": [], "2-hop": [], "3-hop": []}
    
    visited = set(start_nodes)
    queue = deque([(node, 0) for node in start_nodes])
    
    while queue:
        node, hop = queue.popleft()
        if hop >= 3:
            continue
        
        
        attr = G.nodes[node]
        if attr.get("kind") == "cell":
            text = attr.get("text", "")
            table = attr.get("table")
            col = attr.get("col")
            if text and table and col:
                hop_dict[f"{hop+1}-hop"].append(f"{text}; {table}.{col}")
        
        
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, hop + 1))
    
    
    for hop in ["1-hop", "2-hop", "3-hop"]:
        if hop not in hop_dict:
            hop_dict[hop] = []
    
    return dict(hop_dict)


def semantic_match(query: str, candidates: List[Tuple], threshold: float = 0.8) -> List[Tuple]:
    """
    Semantic matching with table.context support.
    candidates: list of (text, table_name, column_name, node)
    """
    if not candidates:
        return []
    
    model = get_instructor_model()
    instruction = "Represent the question entity for matching with table cells"
    
    candidate_texts = [c[0] for c in candidates]
    
    with torch.no_grad():
        query_emb = model.encode([[instruction, query]], convert_to_tensor=True)
        cand_embs = model.encode([[instruction, t] for t in candidate_texts], 
                                convert_to_tensor=True)
        similarities = util.cos_sim(query_emb, cand_embs)[0]
    
    matches = []
    for i, sim in enumerate(similarities):
        if sim >= threshold:
            matches.append((candidates[i], float(sim)))
    
    return matches

def prune_and_traverse_multi_table(
    G: nx.Graph,
    entity_header_mapping: Dict[str, str],  
    threshold: float = 0.8
) -> Dict[str, List[str]]:
    """
    Multi-table BFS traversal starting from matched entities.
    """
    
    cell_candidates = []
    cell_nodes = []
    
    for node, attr in G.nodes(data=True):
        if attr.get("kind") == "cell":
            text = attr.get("text", "").strip()
            if text:
                table = attr.get("table")
                col = attr.get("col")
                cell_candidates.append((text, table, col, node))
                cell_nodes.append(node)
    
    start_nodes = set()
    
    
    for entity, mapping in entity_header_mapping.items():
        if mapping == "Others":
            
            matches = semantic_match(entity, cell_candidates, threshold)
            if matches:
                best_match = max(matches, key=lambda x: x[1])
                start_nodes.add(best_match[0][3])  
        else:
            
            table_name, column_name = mapping[0].split(".")
            
            
            column_cells = []
            for text, table, col, node in cell_candidates:
                if table == table_name and col == column_name:
                    column_cells.append((text, table, col, node))
            
            if column_cells:
                
                found = False
                for text, table, col, node in column_cells:
                    if text.lower() == entity.lower():
                        start_nodes.add(node)
                        found = True
                        break
                
                
                if not found:
                    matches = semantic_match(entity, column_cells, threshold)
                    if matches:
                        best_match = max(matches, key=lambda x: x[1])
                        start_nodes.add(best_match[0][3])
    
    
    hop_dict = defaultdict(list)
    if not start_nodes:
        return {"1-hop": [], "2-hop": [], "3-hop": []}
    
    visited = set(start_nodes)
    queue = deque([(node, 0) for node in start_nodes])
    
    while queue:
        node, hop = queue.popleft()
        if hop >= 3:
            continue
        
        
        attr = G.nodes[node]
        if attr.get("kind") == "cell":
            text = attr.get("text", "")
            table = attr.get("table")
            col = attr.get("col")
            if text and table and col:
                hop_dict[f"{hop+1}-hop"].append(f"{text}; {table}.{col}")
        
        
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, hop + 1))
    
    return dict(hop_dict)


def remove_think_blocks(text: str) -> str:
    """
    Removes <think>...</think> blocks from text.
    """
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def collect_links(df: pd.DataFrame) -> List[str]:
    """
    Collects non-null link values from link columns in a DataFrame.
    """
    link_columns = [col for col in df.columns if '_link' in col]

    all_links_series = pd.concat([df[col] for col in link_columns], ignore_index=True)
    all_links_cleaned = all_links_series.dropna()

    return all_links_cleaned.tolist()


def extract_subtable_from_hop(hop_list: List[str], table_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts a subtable from the main table based on mentions in the hop list.
    """
    if hop_list is None:
        return table_df
    
    mentioned_cols = set()
    cell_mentions = []

    for item in hop_list:
        if "; " in item:
            value_part, col_part = item.rsplit("; ", 1)
            if col_part in table_df.columns:
                mentioned_cols.add(col_part)
                cell_mentions.append((value_part.strip(), col_part))

    if not mentioned_cols:
        return pd.DataFrame(columns=table_df.columns)

    cols_to_keep = set(mentioned_cols)
    for col in mentioned_cols:
        link_col = f"{col}_link"
        if link_col in table_df.columns:
            cols_to_keep.add(link_col)

    cols_to_keep = sorted(cols_to_keep, key=lambda x: list(table_df.columns).index(x))  

    mask = pd.Series([False] * len(table_df), index=table_df.index)
    for value, col in cell_mentions:
        col_series = table_df[col].astype(str).fillna("")
        mask |= (col_series == value)

    filtered_df = table_df.loc[mask, cols_to_keep].copy()
    return filtered_df

_instructor_model = None

def get_instructor_model() -> SentenceTransformer:
    """
    Initializes and returns the SentenceTransformer model, ensuring it's only loaded once.
    """
    global _instructor_model
    if _instructor_model is None:
        _instructor_model = SentenceTransformer('hkunlp/instructor-large')
    return _instructor_model


def format_node_for_output(G: nx.Graph, node, headers: List[str]) -> Optional[str]:
    """
    Formats a graph node for output based on its type (cell or entity).
    """
    attr = G.nodes[node]
    kind = attr.get("kind")
    if kind == "cell":
        text = str(attr.get("text", "")).strip()
        col = attr.get("col")
        if col in headers and text:
            return f"{text}; {col}"
    elif kind == "ent":
        text = attr.get("text", "").strip()
        if text:
            return text
    return None