from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage
import json
import sys
import pandas as pd
from collections import defaultdict
import os
import re
import ast
from typing import Union, List, Dict
from collections import defaultdict, deque
import networkx as nx
from pprint import pprint
from collections import defaultdict

from dotenv import load_dotenv
from modules.data_processing import table_to_df
from modules.graph_construction import *
from modules.prompts import *
from modules.utils import semantic_match

import json
import os
import time
from tqdm import tqdm
from openai import OpenAI


def clean_mapping_value(mapping_value: str) -> Union[str, List[str]]:
    """
    Clean and parse mapping value that could contain multiple columns.
    """
    if not mapping_value or mapping_value.strip() == "":
        return "Others"

    value = mapping_value.strip()

    if value.lower() == "others":
        return "Others"

    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except:
            pass

    parts = [p.strip().strip('"').strip("'") for p in value.split(",")]
    valid_parts = []
    for part in parts:
        if part and "." in part:
            table_col = part.split(".", 1)
            if len(table_col) == 2 and table_col[0] and table_col[1]:
                valid_parts.append(part)

    if valid_parts:
        return valid_parts if len(valid_parts) > 1 else valid_parts[0]

    return "Others"


class DeepSeekClient:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    def ask(self, prompt, **kwargs):
        response = self.client.chat.completions.create(
            model=kwargs.get("model", "deepseek-reasoner"),
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.3),
            extra_body=(
                {"thinking": {"type": "enabled"}} if kwargs.get("think", True) else None
            ),
            response_format=(
                {"type": "json_object"} if kwargs.get("json_output", False) else None
            ),
        )
        return response.choices[0].message.content


def process_multitable(
    tables_path: str, question_path: str, client: DeepSeekClient
) -> str:
    """
    Process multiple CSV tables from a folder and a JSON file containing the question.

    Args:
        folder_path (str): Path to the folder containing CSV files and question.json
        client (DeepSeekClient): The DeepSeek client instance

    Returns:
        str: The result of processing the multi-table QA
    """
    tables_dict = {}
    tables_info = {}

    
    for filename in os.listdir(tables_path):
        table_name = os.path.splitext(filename)[
            0
        ]  
        df = pd.read_csv(os.path.join(tables_path, filename))
        tables_dict[table_name] = df
        tables_info[table_name] = list(df.columns)

    with open(question_path, "r", encoding="utf-8") as f:
        question_data = json.load(f)
        question = question_data.get("question", "")

    
    tables_info_formatted = []
    for key in tables_info:
        for value in tables_info[key]:
            tables_info_formatted.append(f"{key}.{value}")

    entities_prompt = prompt_ent_extr_from_q(question, tables_info_formatted)
    entities_response = client.ask(entities_prompt, model="deepseek-chat", think=False)
    entities_str = entities_response
    print(entities_str)

    entities = []
    if "Entities:" in entities_str:
        entities_part = entities_str.split("Entities:")[1].strip()
        try:
            if entities_part.startswith("[") and entities_part.endswith("]"):
                entities = ast.literal_eval(entities_part)
        except:
            entities = []

    rel_headers_prompt = prompt_relevant_header(
        question, tables_info_formatted, entities
    )
    rel_headers_response = client.ask(
        rel_headers_prompt, model="deepseek-chat", think=False
    )
    rel_headers_str = rel_headers_response
    print(rel_headers_str)

    relevant_headers = []
    for line in rel_headers_str.strip().split("\n"):
        line = line.strip()
        if line:
            matches = re.findall(r"[\w\-]+\.[\w\-]+", line)
            relevant_headers.extend(matches)

    mapping_prompt = prompt_entity_header_mapping(question, entities, relevant_headers)
    mapping_response = client.ask(mapping_prompt, model="deepseek-chat", think=False)
    mapping_str = mapping_response
    mapping_str = mapping_str.strip("```").strip("json")

    entity_header_mapping = ast.literal_eval(mapping_str)
    print(entity_header_mapping)

    
    table_columns = defaultdict(list)

    for header in relevant_headers:
        if "." in header:
            table, column = header.split(".", 1)
            table_columns[table].append(column)

    
    markdown_parts = []

    for table_name, columns in table_columns.items():
        if table_name in tables_dict:
            df = tables_dict[table_name]

            
            existing_columns = [col for col in columns if col in df.columns]

            if existing_columns:
                
                df_selected = df[existing_columns].head(5)

                
                markdown_parts.append(f"### Таблица: {table_name}")
                markdown_parts.append(f"Столбцы: {', '.join(existing_columns)}")
                markdown_parts.append("")

                
                markdown_parts.append(df_selected.to_markdown(index=False))
                markdown_parts.append("")  

    
    exaple_markdown = "\n".join(markdown_parts)

    links_response = client.ask(
        find_links_prompt(exaple_markdown), model="deepseek-chat", think="False"
    )
    links_str = links_response
    links_str = links_str.strip("```").strip("json")
    links_mapping = ast.literal_eval(links_str)
    print(links_mapping)

    multi_table_graph = build_multi_table_graph_filtered(
        tables_dict, relevant_headers, links_mapping
    )

    hop_dict = prune_and_traverse_multi_table_filtered(
        multi_table_graph, entity_header_mapping, threshold=0.8
    )

    answers, contexts = process_hops_with_debug(
        hop_dict, tables_dict, client, llm_reader_prompt, question
    )

    return answers, contexts


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python odyssey_multitable.py <input_json_file>")
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        inputs = json.load(f)

    result = process_multitable(inputs)
    print(f"Answer: {result}")
