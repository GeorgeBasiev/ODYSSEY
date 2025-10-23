import re
from typing import Dict, List, Optional, Tuple
import torch
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict, deque
import pandas as pd
import networkx as nx


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

    cols_to_keep = sorted(cols_to_keep, key=lambda x: list(table_df.columns).index(x))  # сохранить порядок

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


def semantic_match(query: str, candidates: list, threshold: float = 0.8) -> List[Tuple[str, float]]:
    """
    Performs semantic matching between a query and a list of candidates using SentenceTransformers.
    """
    if not candidates:
        return []

    model = get_instructor_model()
    device = model.device

    instruction = "Represent the question entity for matching with table cells or document entities"

    with torch.no_grad():
        query_emb = model.encode(
            [[instruction, query]],
            convert_to_tensor=True,
            device=device
        )

        candidate_embs = model.encode(
            [[instruction, cand] for cand in candidates],
            convert_to_tensor=True,
            device=device
        )

        similarities = util.cos_sim(query_emb, candidate_embs)[0]

    matches = []
    for i, sim in enumerate(similarities):
        if sim >= threshold:
            matches.append((candidates[i], float(sim)))
    
    del query_emb, candidate_embs, similarities
    torch.cuda.empty_cache()
    
    return matches


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


def prune_and_traverse_hybrid_graph(G: nx.Graph, entity_header_mapping: Dict[str, str], headers: List[str], threshold: float = 0.8) -> Optional[Dict[str, List[str]]]:
    """
    Prunes and traverses the hybrid graph to find relevant information starting from initial nodes.
    """
    entity_total_list = []
    entity_to_node = {}
    for node, attr in G.nodes(data=True):
        if attr.get("kind") == "ent":
            text = attr.get("text", "").strip()
            if text:
                entity_total_list.append(text)
                entity_to_node[text] = node

    def top1_semantic_match(query, candidates, threshold=0.8) -> Optional[Tuple[str, float]]:
        if not candidates:
            return None
        matches = semantic_match(query, candidates, threshold=threshold)
        if not matches:
            return None
        matches = sorted(matches, key=lambda x: x[1], reverse=True)
        cand, score = matches[0]
        return (cand, score) if score >= threshold else None

    start_nodes = set()

    for question_entity, mapped_header in entity_header_mapping.items():
        q = str(question_entity).strip()
        if not q:
            continue

        if mapped_header == "Others":
            tm = top1_semantic_match(q, entity_total_list, threshold)
            if tm:
                ent_text, _ = tm
                node = entity_to_node.get(ent_text)
                if node and G.has_node(node):
                    start_nodes.add(node)

        else:
            if mapped_header not in headers:
                continue

            cell_texts, cell_nodes = [], []
            for node, attr in G.nodes(data=True):
                if attr.get("kind") == "cell" and attr.get("col") == mapped_header:
                    text = str(attr.get("text", "")).strip()
                    if text:
                        cell_texts.append(text)
                        cell_nodes.append(node)

            tm = top1_semantic_match(q, cell_texts, threshold)
            if tm:
                best_text, _ = tm
                try:
                    idx = cell_texts.index(best_text)
                    start_nodes.add(cell_nodes[idx])
                except ValueError:
                    pass
            else:
                all_texts, all_nodes = [], []
                for node, attr in G.nodes(data=True):
                    if attr.get("kind") == "cell":
                        t = str(attr.get("text", "")).strip()
                        if t:
                            all_texts.append(t)
                            all_nodes.append(node)
                tm2 = top1_semantic_match(q, all_texts, threshold)
                if tm2:
                    best_text, _ = tm2
                    try:
                        idx = all_texts.index(best_text)
                        start_nodes.add(all_nodes[idx])
                    except ValueError:
                        pass

    if not start_nodes:
        return None
    print(start_nodes)
    hop_dict = defaultdict(list)
    visited = set(start_nodes)
    queue = deque([(node, 0) for node in start_nodes])

    while queue:
        node, hop = queue.popleft()
        if hop >= 3:
            continue

        next_hop = hop + 1
        for neighbor in G.neighbors(node):
            if neighbor in visited:
                continue

            visited.add(neighbor)
            queue.append((neighbor, next_hop))

            formatted = format_node_for_output(G, neighbor, headers)
            if formatted:
                hop_dict[f"{next_hop}-hop"].append(formatted)

    return {
        "1-hop": hop_dict["1-hop"],
        "2-hop": hop_dict["2-hop"],
        "3-hop": hop_dict["3-hop"]
    }