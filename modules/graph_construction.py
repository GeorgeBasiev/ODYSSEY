from collections import defaultdict, deque
from typing import Dict, List, Union, Set, Tuple, Optional
import networkx as nx
import pandas as pd
import itertools
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import warnings
import re

warnings.filterwarnings("ignore")
from sentence_transformers import SentenceTransformer


class TableGraphSearch:
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
        """
        Инициализация с моделью для эмбедингов.
        Модель загружается при необходимости и не кэшируется.

        Args:
            model_name: Название модели SentenceTransformer
        """
        self.model_name = model_name
        self.model = None  

    def _load_model_if_needed(self) -> Optional[SentenceTransformer]:
        """
        Загружает модель если она еще не загружена.
        Возвращает модель или None если загрузка не удалась.
        """
        if self.model is None:
            try:
                print(f"Загрузка модели {self.model_name}...")
                self.model = SentenceTransformer(self.model_name)
                print(f"Модель {self.model_name} загружена успешно")
            except Exception as e:
                print(f"Не удалось загрузить модель {self.model_name}: {e}")
                print("Будет использовано текстовое сравнение")
                self.model = None
        
        return self.model

    def cleanup(self):
        """
        Очищает модель из памяти.
        """
        if self.model is not None:
            print(f"Очистка модели {self.model_name} из памяти...")
            
            del self.model
            self.model = None
            import gc
            gc.collect()  

    def build_multi_table_graph_filtered(
        self,
        tables_dict: Dict[str, pd.DataFrame],
        relevant_headers: List[str],
        relationships: Dict[str, str] = None,
    ) -> nx.Graph:
        """
        Builds a unified graph from multiple tables, considering only relevant columns.
        Each table is a connected component. Adds edges between related columns.
        Модель загружается только если нужна и очищается после использования.

        Args:
            tables_dict: Dictionary of table names to DataFrames
            relevant_headers: List of headers in format "table.column"
            relationships: Dictionary of column relationships in format {"table1.column1": "table2.column2"}
        """
        G = nx.Graph()

        
        table_columns = defaultdict(list)
        for header in relevant_headers:
            if "." in header:
                table, column = header.split(".", 1)
                table_columns[table].append(column)

        
        cell_value_nodes = {}

        
        header_nodes = {}

        
        text_embeddings = {}

        
        for table_name, columns in table_columns.items():
            if table_name not in tables_dict:
                continue

            df = tables_dict[table_name]

            
            existing_columns = [col for col in columns if col in df.columns]

            if not existing_columns:
                continue

            
            for col in existing_columns:
                header_node = f"header.{table_name}.{col}"
                G.add_node(
                    header_node,
                    kind="header",
                    table=table_name,
                    col=col,
                    full_name=f"{table_name}.{col}",
                )
                header_nodes[f"{table_name}.{col}"] = header_node

            
            n_rows = len(df)

            if n_rows > 1000:
                print(
                    f"Обработка таблицы {table_name}: {n_rows} строк, {len(existing_columns)} столбцов..."
                )

            for r in range(n_rows):
                for col in existing_columns:
                    raw = df.iloc[r][col] if col in df.columns else ""
                    text = "" if pd.isna(raw) else str(raw).strip()

                    cell_node = f"cell.{table_name}.{r}.{col}"
                    G.add_node(
                        cell_node,
                        kind="cell",
                        table=table_name,
                        row=r,
                        col=col,
                        text=text,
                        full_name=f"{table_name}.{col}",
                    )

                    
                    header_node = f"header.{table_name}.{col}"
                    G.add_edge(cell_node, header_node, kind="cell-header")

                    
                    if text:
                        key = (table_name, col, text)
                        if key not in cell_value_nodes:
                            cell_value_nodes[key] = []
                        cell_value_nodes[key].append(cell_node)

                        
                        if cell_node not in text_embeddings:
                            text_embeddings[cell_node] = text

                
                row_cells = [f"cell.{table_name}.{r}.{col}" for col in existing_columns]
                for u, v in itertools.combinations(row_cells, 2):
                    if G.has_node(u) and G.has_node(v):
                        G.add_edge(u, v, kind="row")

            if n_rows > 1000 and n_rows % 1000 == 0:
                print(f"  Обработано {r+1}/{n_rows} строк...")

        
        if text_embeddings:
            model = self._load_model_if_needed()
            if model is not None:
                print("Создание эмбедингов для текстов...")
                texts = list(text_embeddings.values())
                text_nodes = list(text_embeddings.keys())

                
                batch_size = 100
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i : i + batch_size]
                    batch_nodes = text_nodes[i : i + batch_size]

                    batch_embeddings = model.encode(
                        batch_texts, convert_to_numpy=True, show_progress_bar=False
                    )

                    
                    for node, embedding in zip(batch_nodes, batch_embeddings):
                        G.nodes[node]["embedding"] = embedding

        
        if relationships:
            for col1, col2 in relationships.items():
                if col1 in header_nodes and col2 in header_nodes:
                    G.add_edge(
                        header_nodes[col1],
                        header_nodes[col2],
                        kind="column-relationship",
                        relationship=f"{col1} ↔ {col2}",
                    )

                    self._add_cell_relationships(G, cell_value_nodes, col1, col2)

        node_count = G.number_of_nodes()
        edge_count = G.number_of_edges()
        print(f"Граф построен: {node_count} узлов, {edge_count} ребер")

        
        self.cleanup()

        return G

    def _add_cell_relationships(
        self,
        G: nx.Graph,
        cell_value_nodes: Dict[Tuple[str, str, str], List[str]],
        col1: str,
        col2: str,
    ) -> None:
        """Добавляет ребра между ячейками с одинаковыми значениями в связанных столбцах."""
        table1, column1 = col1.split(".")
        table2, column2 = col2.split(".")

        values_table1 = defaultdict(list)
        values_table2 = defaultdict(list)

        
        for (table, col, value), nodes in cell_value_nodes.items():
            if table == table1 and col == column1:
                values_table1[value].extend(nodes)

        
        for (table, col, value), nodes in cell_value_nodes.items():
            if table == table2 and col == column2:
                values_table2[value].extend(nodes)

        
        added_edges = 0
        for value, nodes1 in values_table1.items():
            if value in values_table2:
                nodes2 = values_table2[value]
                for node1 in nodes1:
                    for node2 in nodes2:
                        if not G.has_edge(node1, node2):
                            G.add_edge(
                                node1,
                                node2,
                                kind="value-match",
                                value=value,
                                relationship=f"{col1} ↔ {col2}",
                            )
                            added_edges += 1

        if added_edges > 0:
            print(f"  Добавлено {added_edges} связей между {col1} и {col2}")

    def __del__(self):
        """Деструктор для очистки модели при удалении объекта."""
        self.cleanup()

    def preprocess_text(self, text: str) -> str:
        """
        Предобработка текста для улучшения поиска совпадений.

        Args:
            text: Исходный текст

        Returns:
            Предобработанный текст
        """
        if not isinstance(text, str):
            text = str(text)

        
        text = text.lower()

        
        text = " ".join(text.split())

        
        text = re.sub(r"[^\w\s\.\-@#]", " ", text)

        
        text = re.sub(r"\s+", " ", text)

        
        text = text.strip()

        return text

    def normalize_entity_name(self, entity: str) -> str:
        """
        Нормализация названия сущности для поиска.

        Args:
            entity: Исходное название сущности

        Returns:
            Нормализованное название
        """
        entity = self.preprocess_text(entity)

        
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
        }
        words = entity.split()
        filtered_words = [word for word in words if word not in stop_words]

        return " ".join(filtered_words)

    def find_start_nodes_with_embeddings(
        self,
        G: nx.Graph,
        entity_header_mapping: Dict[str, Union[str, List[str]]],
        min_threshold: float = 0.2,
        max_threshold: float = 0.8,
        top_k: int = 15,
    ) -> Set[str]:
        """
        Надежный поиск стартовых узлов с адаптивным порогом и множеством стратегий

        Args:
            G: Граф с данными
            entity_header_mapping: Маппинг сущностей на столбцы
            min_threshold: Минимальный порог схожести
            max_threshold: Максимальный порог схожести
            top_k: Количество наиболее похожих результатов

        Returns:
            Множество стартовых узлов
        """
        all_start_nodes = set()

        print(f"\n{'='*60}")
        print("ПОИСК СТАРТОВЫХ УЗЛОВ (расширенный)")
        print(f"{'='*60}")

        for entity, mapping in entity_header_mapping.items():
            entity_str = str(entity).strip()
            if not entity_str:
                continue

            print(f"\n● Сущность: '{entity_str}'")
            print(f"  Маппинг: {mapping}")

            
            processed_entity = self.normalize_entity_name(entity_str)
            print(f"  Нормализовано: '{processed_entity}'")

            found_nodes = set()

            
            if self.model is not None:
                print("  Стратегия 1: Поиск по эмбедингам...")
                current_threshold = max_threshold
                while current_threshold >= min_threshold and not found_nodes:
                    nodes = self._find_by_embeddings_adaptive(
                        G, processed_entity, mapping, current_threshold, top_k
                    )
                    if nodes:
                        found_nodes.update(nodes)
                        
                        
                        
                    
                    
                    current_threshold -= 0.1

            
            if not found_nodes:
                print("  Стратегия 2: Поиск точных совпадений...")
                exact_nodes = self._find_exact_matches(G, processed_entity, mapping)
                if exact_nodes:
                    found_nodes.update(exact_nodes)
                    print(f"    ✓ Найдено {len(exact_nodes)} точных совпадений")

            
            if not found_nodes:
                print("  Стратегия 3: Поиск частичных совпадений...")
                partial_nodes = self._find_partial_matches(G, processed_entity, mapping)
                if partial_nodes:
                    found_nodes.update(partial_nodes)
                    print(f"    ✓ Найдено {len(partial_nodes)} частичных совпадений")

            
            if not found_nodes:
                print("  Стратегия 4: Поиск по подстрокам...")
                substring_nodes = self._find_substring_matches(
                    G, processed_entity, mapping
                )
                if substring_nodes:
                    found_nodes.update(substring_nodes)
                    print(
                        f"    ✓ Найдено {len(substring_nodes)} совпадений по подстрокам"
                    )

            
            if not found_nodes:
                print("  Стратегия 5: Поиск по токенам...")
                token_nodes = self._find_token_matches(G, processed_entity, mapping)
                if token_nodes:
                    found_nodes.update(token_nodes)
                    print(f"    ✓ Найдено {len(token_nodes)} совпадений по токенам")

            
            if not found_nodes and len(processed_entity) > 3:
                print("  Стратегия 6: Поиск по начальным символам...")
                prefix_nodes = self._find_prefix_matches(G, processed_entity, mapping)
                if prefix_nodes:
                    found_nodes.update(prefix_nodes)
                    print(
                        f"    ✓ Найдено {len(prefix_nodes)} совпадений по начальным символам"
                    )

            all_start_nodes.update(found_nodes)

            if found_nodes:
                print(f"  ИТОГ для '{entity_str}': найдено {len(found_nodes)} узлов")
                
                for i, node in enumerate(list(found_nodes)[:3]):  
                    attr = G.nodes[node]
                    text = attr.get("text", "")
                    table = attr.get("table", "")
                    col = attr.get("col", "")
                    print(f"    Пример {i+1}: '{text}' в {table}.{col}")
            else:
                print(f"  ВНИМАНИЕ: Не найдено ни одного узла для '{entity_str}'")

        print(f"\n{'='*60}")
        print(f"ОБЩИЙ ИТОГ: найдено {len(all_start_nodes)} стартовых узлов")
        print(f"{'='*60}")

        return all_start_nodes

    def _find_by_embeddings_adaptive(
        self,
        G: nx.Graph,
        entity: str,
        mapping: Union[str, List[str]],
        similarity_threshold: float,
        top_k: int,
    ) -> Set[str]:
        """Поиск по эмбедингам с адаптивным порогом"""
        if self.model is None:
            return set()

        found_nodes = set()

        
        cell_nodes = []
        cell_texts = []
        cell_embeddings = []

        for node, attr in G.nodes(data=True):
            if attr.get("kind") == "cell":
                text = attr.get("text", "").strip()
                if text and len(text) > 0 and "embedding" in attr:
                    cell_nodes.append(node)
                    cell_texts.append(text)
                    cell_embeddings.append(attr["embedding"])

        if not cell_embeddings:
            return set()

        cell_embeddings = np.array(cell_embeddings)

        
        query_embedding = self.model.encode([entity], convert_to_numpy=True)[0]

        
        similarities = cosine_similarity([query_embedding], cell_embeddings)[0]

        
        if isinstance(mapping, str) and mapping.lower() == "others":
            
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            for idx in top_indices:
                if similarities[idx] >= similarity_threshold:
                    found_nodes.add(cell_nodes[idx])
        else:
            
            target_columns = [mapping] if isinstance(mapping, str) else mapping

            column_mask = np.zeros(len(cell_nodes), dtype=bool)
            for i, node in enumerate(cell_nodes):
                attr = G.nodes[node]
                table = attr.get("table", "")
                col = attr.get("col", "")

                for target in target_columns:
                    if "." in target:
                        target_table, target_col = target.split(".", 1)
                        if table == target_table and col == target_col:
                            column_mask[i] = True
                            break
                    else:
                        
                        if col == target:
                            column_mask[i] = True
                            break

            if column_mask.any():
                filtered_similarities = similarities[column_mask]
                filtered_indices = np.where(column_mask)[0]

                if len(filtered_similarities) > 0:
                    top_filtered_idx = np.argsort(filtered_similarities)[-top_k:][::-1]

                    for idx in top_filtered_idx:
                        if filtered_similarities[idx] >= similarity_threshold:
                            original_idx = filtered_indices[idx]
                            found_nodes.add(cell_nodes[original_idx])

        return found_nodes

    def _find_exact_matches(
        self, G: nx.Graph, entity: str, mapping: Union[str, List[str]]
    ) -> Set[str]:
        """Поиск точных совпадений после нормализации"""
        found_nodes = set()

        
        entity_normalized = self.preprocess_text(entity)

        
        target_columns = self._get_target_columns(mapping)

        for node, attr in G.nodes(data=True):
            if attr.get("kind") == "cell":
                
                if not self._check_column_match(attr, target_columns):
                    continue

                
                text = attr.get("text", "").strip()
                if not text:
                    continue

                text_normalized = self.preprocess_text(text)

                
                if text_normalized == entity_normalized:
                    found_nodes.add(node)

        return found_nodes

    def _find_partial_matches(
        self, G: nx.Graph, entity: str, mapping: Union[str, List[str]]
    ) -> Set[str]:
        """Поиск частичных совпадений"""
        found_nodes = set()

        
        entity_normalized = self.preprocess_text(entity)

        
        entity_words = set(entity_normalized.split())
        if not entity_words:
            return found_nodes

        target_columns = self._get_target_columns(mapping)

        for node, attr in G.nodes(data=True):
            if attr.get("kind") == "cell":
                if not self._check_column_match(attr, target_columns):
                    continue

                text = attr.get("text", "").strip()
                if not text:
                    continue

                text_normalized = self.preprocess_text(text)
                text_words = set(text_normalized.split())

                
                common_words = entity_words & text_words
                if common_words:
                    
                    found_nodes.add(node)

        return found_nodes

    def _find_substring_matches(
        self, G: nx.Graph, entity: str, mapping: Union[str, List[str]]
    ) -> Set[str]:
        """Поиск по подстрокам"""
        found_nodes = set()

        
        entity_normalized = self.preprocess_text(entity)

        
        if len(entity_normalized) < 3:
            return found_nodes

        target_columns = self._get_target_columns(mapping)

        for node, attr in G.nodes(data=True):
            if attr.get("kind") == "cell":
                if not self._check_column_match(attr, target_columns):
                    continue

                text = attr.get("text", "").strip()
                if not text:
                    continue

                text_normalized = self.preprocess_text(text)

                
                if (
                    entity_normalized in text_normalized
                    or text_normalized in entity_normalized
                ):
                    found_nodes.add(node)

        return found_nodes

    def _find_token_matches(
        self, G: nx.Graph, entity: str, mapping: Union[str, List[str]]
    ) -> Set[str]:
        """Поиск по токенам (слово за словом)"""
        found_nodes = set()

        
        entity_normalized = self.preprocess_text(entity)
        entity_tokens = entity_normalized.split()

        if len(entity_tokens) < 2:
            return found_nodes

        target_columns = self._get_target_columns(mapping)

        for node, attr in G.nodes(data=True):
            if attr.get("kind") == "cell":
                if not self._check_column_match(attr, target_columns):
                    continue

                text = attr.get("text", "").strip()
                if not text:
                    continue

                text_normalized = self.preprocess_text(text)
                text_tokens = text_normalized.split()

                
                all_tokens_found = all(
                    token in text_normalized for token in entity_tokens
                )
                if all_tokens_found:
                    found_nodes.add(node)

        return found_nodes

    def _find_prefix_matches(
        self, G: nx.Graph, entity: str, mapping: Union[str, List[str]]
    ) -> Set[str]:
        """Поиск по начальным символам"""
        found_nodes = set()

        
        entity_normalized = self.preprocess_text(entity)

        
        prefix_length = min(4, len(entity_normalized))
        prefix = entity_normalized[:prefix_length]

        target_columns = self._get_target_columns(mapping)

        for node, attr in G.nodes(data=True):
            if attr.get("kind") == "cell":
                if not self._check_column_match(attr, target_columns):
                    continue

                text = attr.get("text", "").strip()
                if not text:
                    continue

                text_normalized = self.preprocess_text(text)

                
                if text_normalized.startswith(prefix) or prefix in text_normalized:
                    found_nodes.add(node)

        return found_nodes

    def _get_target_columns(self, mapping: Union[str, List[str]]) -> List[str]:
        """Преобразует mapping в список целевых столбцов"""
        if mapping == "Others" or (
            isinstance(mapping, str) and mapping.lower() == "others"
        ):
            return []  

        if isinstance(mapping, str):
            return [mapping]
        else:
            return mapping

    def _check_column_match(self, attr: Dict, target_columns: List[str]) -> bool:
        """Проверяет, соответствует ли ячейка целевому столбцу"""
        if not target_columns:  
            return True

        table = attr.get("table", "")
        col = attr.get("col", "")

        for target in target_columns:
            if "." in target:
                target_table, target_col = target.split(".", 1)
                if table == target_table and col == target_col:
                    return True
            else:
                
                if col == target:
                    return True

        return False

    def _jaccard_similarity(self, str1: str, str2: str) -> float:
        """Вычисляет Jaccard схожесть между строками."""
        set1 = set(str1.split())
        set2 = set(str2.split())

        if not set1 and not set2:
            return 1.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def prune_and_traverse_with_embeddings(
        self,
        G: nx.Graph,
        entity_header_mapping: Dict[str, Union[str, List[str]]],
        similarity_threshold: float = 0.7,
        top_k: int = 5,
        max_hops: int = 3,
    ) -> Dict[str, List[str]]:
        """
        Multi-table BFS traversal starting from matched entities using embeddings.

        Args:
            G: Граф с данными
            entity_header_mapping: Маппинг сущностей на столбцы
            similarity_threshold: Порог схожести для поиска стартовых узлов
            top_k: Количество наиболее похожих результатов
            max_hops: Максимальная глубина поиска

        Returns:
            Словарь с результатами по уровням
        """
        
        print(f"Поиск стартовых узлов для {len(entity_header_mapping)} сущностей...")
        start_nodes = self.find_start_nodes_with_embeddings(
            G, entity_header_mapping, similarity_threshold, top_k
        )

        print(f"Найдено {len(start_nodes)} стартовых узлов")

        
        hop_dict = defaultdict(list)
        if not start_nodes:
            return {f"{i+1}-hop": [] for i in range(max_hops)}

        visited = set(start_nodes)
        queue = deque([(node, 0) for node in start_nodes])

        while queue:
            node, hop = queue.popleft()
            if hop >= max_hops:
                continue

            
            attr = G.nodes[node]
            if attr.get("kind") == "cell":
                text = attr.get("text", "")
                table = attr.get("table", "")
                col = attr.get("col", "")
                if text and table and col:
                    hop_dict[f"{hop+1}-hop"].append(f"{text}; {table}.{col}")

            
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, hop + 1))

        
        result = {}
        for hop in [f"{i+1}-hop" for i in range(max_hops)]:
            if hop in hop_dict:
                seen = set()
                unique_list = []
                for item in hop_dict[hop]:
                    if item not in seen:
                        seen.add(item)
                        unique_list.append(item)
                result[hop] = unique_list
            else:
                result[hop] = []

        return result



def build_multi_table_graph_filtered(
    tables_dict: Dict[str, pd.DataFrame],
    relevant_headers: List[str],
    relationships: Dict[str, str] = None,
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",
) -> nx.Graph:
    """Обертка для обратной совместимости."""
    searcher = TableGraphSearch(model_name)
    return searcher.build_multi_table_graph_filtered(
        tables_dict, relevant_headers, relationships
    )


def prune_and_traverse_multi_table_filtered(
    G: nx.Graph,
    entity_header_mapping: Dict[str, Union[str, List[str]]],
    threshold: float = 0.8,
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    similarity_threshold: float = 0.7,
) -> Dict[str, List[str]]:
    """Обертка для обратной совместимости."""
    searcher = TableGraphSearch(model_name)
    return searcher.prune_and_traverse_with_embeddings(
        G, entity_header_mapping, similarity_threshold
    )


def accumulate_hop_information(hop_dict, tables_dict):
    """
    Накопление информации из hops и возврат только финального контекста 3-hop.

    Args:
        hop_dict: Словарь с результатами по hops
        tables_dict: Исходные таблицы

    Returns:
        Финальный контекст 3-hop (строка) и словарь с накопленной статистикой
    """
    
    accumulated_tables = defaultdict(
        lambda: {
            "rows": set(),  
            "found_via": defaultdict(set),  
        }
    )

    
    for hop in ["1-hop", "2-hop", "3-hop"]:
        hop_items = hop_dict.get(hop, [])
        if not hop_items:
            continue

        
        new_rows_info = defaultdict(lambda: {"indices": set(), "via": {}})

        for item in hop_items:
            if ";" in item:
                value, table_col = item.split(";", 1)
                value = value.strip()
                table_col = table_col.strip()

                if "." in table_col:
                    table_name, column_name = table_col.split(".", 1)

                    df = tables_dict.get(table_name)
                    if df is not None and column_name in df.columns:
                        mask = df[column_name].astype(str).str.strip() == value
                        matching_indices = df[mask].index.tolist()

                        for idx in matching_indices:
                            new_rows_info[table_name]["indices"].add(idx)
                            new_rows_info[table_name]["via"][idx] = new_rows_info[
                                table_name
                            ]["via"].get(idx, set())
                            new_rows_info[table_name]["via"][idx].add(column_name)

        
        for table_name, info in new_rows_info.items():
            accumulated_tables[table_name]["rows"].update(info["indices"])

            
            for idx, columns in info["via"].items():
                accumulated_tables[table_name]["found_via"][idx].update(columns)

    
    table_contexts = []
    stats = {"total_tables": 0, "total_rows": 0, "total_cells": 0}

    for table_name, data in accumulated_tables.items():
        if not data["rows"]:
            continue

        df = tables_dict.get(table_name)
        if df is None:
            continue

        
        sorted_rows = sorted(data["rows"])

        
        all_columns = list(df.columns)

        
        sub_df = df.loc[sorted_rows, all_columns]

        
        sources_info = []
        for idx in sorted_rows:
            if idx in data["found_via"] and data["found_via"][idx]:
                sources = sorted(data["found_via"][idx])
                sources_info.append(f"Row {idx}: found via {', '.join(sources)}")

        if sources_info:
            sources_summary = " | " + " | ".join(sources_info)
        else:
            sources_summary = ""

        
        markdown_table = sub_df.to_markdown(index=True)

        
        table_header = (
            f"Table: {table_name} - {len(sorted_rows)} rows{sources_summary}\n"
        )

        table_contexts.append(f"{table_header}{markdown_table}")

        
        stats["total_tables"] += 1
        stats["total_rows"] += len(sorted_rows)
        stats["total_cells"] += len(sorted_rows) * len(all_columns)

    
    context_header = f"## Final Context (3-hop accumulated)\n"
    context_header += f"## Tables found through graph traversal\n\n"

    final_context = (
        context_header + "\n\n".join(table_contexts) if table_contexts else ""
    )

    return final_context, stats



def process_hops(hop_dict, tables_dict, llm, llm_reader_prompt, question):
    """
    Основная функция обработки - отправляет в LLM только финальный 3-hop контекст.

    Args:
        hop_dict: Результаты BFS обхода
        tables_dict: Исходные таблицы
        llm: Языковая модель
        llm_reader_prompt: Функция для создания промпта
        question: Вопрос пользователя

    Returns:
        Ответ LLM и контекст
    """
    
    final_context, stats = accumulate_hop_information(hop_dict, tables_dict)

    if not final_context:
        print("No data found in any hop")
        return "No data found", {}

    
    print(
        f"Final context: {stats['total_tables']} tables, "
        f"{stats['total_rows']} rows, ~{stats['total_cells']} cells"
    )

    
    try:
        answer_response = llm.ask(llm_reader_prompt(final_context, question))
        answer = answer_response.strip()
    except Exception as e:
        print(f"Error calling LLM: {e}")
        answer = "Error processing request"

    
    invalid_responses = [
        "i cannot answer based on the provided tables",
        "none",
        "not found",
        "cannot answer",
        "i don't know",
        "i do not know",
        "information is not available",
        "нет информации",
        "не могу ответить",
    ]

    if answer and not any(inv.lower() in answer.lower() for inv in invalid_responses):
        print(f"\n{'='*50}")
        print(f"FINAL ANSWER: {answer}")
        print(f"{'='*50}")
        return answer, final_context
    else:
        print("\nCould not find a definitive answer.")
        return "Cannot determine from available information", final_context



def process_hops_with_debug(hop_dict, tables_dict, llm, llm_reader_prompt, question):
    """
    Версия с сохранением отладочной информации, но отправкой только 3-hop в LLM.
    """
    
    accumulated_tables = defaultdict(
        lambda: {"rows": set(), "found_via": defaultdict(set)}
    )

    hop_contexts = {}  
    debug_info = {}  

    
    for hop in ["1-hop", "2-hop", "3-hop"]:
        hop_items = hop_dict.get(hop, [])

        new_rows_info = defaultdict(lambda: {"indices": set(), "via": {}})

        if hop_items:
            for item in hop_items:
                if ";" in item:
                    value, table_col = item.split(";", 1)
                    value = value.strip()
                    table_col = table_col.strip()

                    if "." in table_col:
                        table_name, column_name = table_col.split(".", 1)

                        df = tables_dict.get(table_name)
                        if df is not None and column_name in df.columns:
                            mask = df[column_name].astype(str).str.strip() == value
                            matching_indices = df[mask].index.tolist()

                            for idx in matching_indices:
                                new_rows_info[table_name]["indices"].add(idx)
                                new_rows_info[table_name]["via"][idx] = new_rows_info[
                                    table_name
                                ]["via"].get(idx, set())
                                new_rows_info[table_name]["via"][idx].add(column_name)

        
        for table_name, info in new_rows_info.items():
            accumulated_tables[table_name]["rows"].update(info["indices"])

            for idx, columns in info["via"].items():
                accumulated_tables[table_name]["found_via"][idx].update(columns)

        
        table_contexts = []
        hop_stats = {"tables": 0, "rows": 0, "new_rows": 0}

        for table_name, data in accumulated_tables.items():
            if not data["rows"]:
                continue

            df = tables_dict.get(table_name)
            if df is None:
                continue

            sorted_rows = sorted(data["rows"])
            all_columns = list(df.columns)
            sub_df = df.loc[sorted_rows, all_columns]

            
            hop_stats["tables"] += 1
            hop_stats["rows"] += len(sorted_rows)

            
            if new_rows_info.get(table_name):
                new_in_this_hop = len(new_rows_info[table_name]["indices"])
                hop_stats["new_rows"] += new_in_this_hop

            
            sources_info = []
            for idx in sorted_rows:
                if idx in data["found_via"] and data["found_via"][idx]:
                    sources = sorted(data["found_via"][idx])
                    sources_info.append(f"Row {idx}: found via {', '.join(sources)}")

            sources_summary = " | " + " | ".join(sources_info) if sources_info else ""
            markdown_table = sub_df.to_markdown(index=True)
            table_header = (
                f"Table: {table_name} - {len(sorted_rows)} rows{sources_summary}\n"
            )

            table_contexts.append(f"{table_header}{markdown_table}")

        
        if table_contexts:
            hop_contexts[hop] = "\n\n".join(table_contexts)
        else:
            hop_contexts[hop] = ""

        debug_info[hop] = hop_stats
        print(
            f"{hop}: {hop_stats['tables']} tables, {hop_stats['rows']} rows, +{hop_stats['new_rows']} new rows"
        )

    
    final_context = hop_contexts.get(
        "3-hop", hop_contexts.get("2-hop", hop_contexts.get("1-hop", ""))
    )

    if not final_context:
        print("No data found in any hop")
        return "No data found", hop_contexts

    
    

    

    
    answer_response = llm.ask(llm_reader_prompt(final_context, question))
    answer = answer_response.strip()

    
    invalid_responses = [
        "i cannot answer based on the provided tables",
        "none",
        "not found",
        "cannot answer",
        "i don't know",
        "i do not know",
        "information is not available",
        "нет информации",
        "не могу ответить",
    ]

    if answer and not any(inv.lower() in answer.lower() for inv in invalid_responses):
        print(f"\n{'='*50}")
        print(f"FINAL ANSWER: {answer}")
        print(f"{'='*50}")
    else:
        print("\nCould not find a definitive answer.")
        answer = "Cannot determine from available information"

    return answer, hop_contexts
