from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import json
import sys
import pandas as pd

from modules.data_processing import table_to_df, extract_entities_by_doc
from modules.graph_construction import build_hybrid_graph
from modules.prompts import prompt_ent_extr_from_q, prompt_relevant_header, prompt_entity_header_mapping, llm_reader_prompt
from modules.utils import remove_think_blocks, extract_subtable_from_hop, collect_links, prune_and_traverse_hybrid_graph


def process_single(inputs: dict) -> str:
    table_df = table_to_df(inputs["table"])

    question = inputs["question"]
    table_id = inputs["table"]["title"]
    table_headers = str([i[0] for i in inputs["table"]["header"]])

    llm = ChatOllama(
        model="qwen3:8b", 
        temperature=0,   
    )

    entities = remove_think_blocks(llm.invoke([HumanMessage(content=prompt_ent_extr_from_q(question, table_id, table_headers))]).content)
    relevant_headers = remove_think_blocks(llm.invoke([HumanMessage(content=prompt_relevant_header(question, table_id, table_headers, entities))]).content)
    entity_headers_match = remove_think_blocks(llm.invoke([HumanMessage(content=prompt_entity_header_mapping(question, table_id, entities, relevant_headers))]).content)

    relevant_headers = relevant_headers[relevant_headers.index("[")+1:relevant_headers.index("]")]
    relevant_headers_list = [i.removesuffix(", ").strip().strip("\"") for i in relevant_headers.split(",")]
    subtable_df = table_df[relevant_headers_list + [i + "_link" for i in relevant_headers_list]]

    doc_entities = extract_entities_by_doc(inputs["links"])

    hybrid_graph = build_hybrid_graph(subtable_df, doc_entities, relevant_headers_list)


    entity_headers_match_stripped = [i.split(":") for i in entity_headers_match.split(",\n")]
    for i in range(len(entity_headers_match_stripped)):
        for j in range(2):
            entity_headers_match_stripped[i][j] = entity_headers_match_stripped[i][j].strip("\"").strip().strip("[").strip("]").strip("\"")

    entity_headers_dict = {}
    for i in entity_headers_match_stripped:
        entity_headers_dict[i[0]] = i[1]

    original_subheaders = [i for i in list(table_df.columns) if not i.endswith("link")]

    hopwise_context = prune_and_traverse_hybrid_graph(
        hybrid_graph,
        entity_headers_dict,
        headers=original_subheaders
    )

    hop1_table = extract_subtable_from_hop(hopwise_context["1-hop"], subtable_df)
    hop1_table_md = hop1_table.to_markdown()
    hop1_links = [inputs["links"][i] for i in collect_links(hop1_table)]

    hop_1_invoke = remove_think_blocks(llm.invoke([HumanMessage(content=llm_reader_prompt(hop1_table_md, hop1_links, question))]).content)
    
    answer = ""
    for i in range(2, 4):
        if "None" in hop_1_invoke:
            relevant_from_hop_1 = hop_1_invoke[hop_1_invoke.index("[")+1:hop_1_invoke.index("]")]
            hop2_table = pd.concat([hop1_table, extract_subtable_from_hop(hopwise_context[f"{i}-hop"], subtable_df)], ignore_index=True, join="inner").drop_duplicates()

            hop2_table_md = hop2_table.to_markdown()
            hop2_links = [inputs["links"][i] for i in collect_links(hop2_table)]

            if relevant_from_hop_1 != "":
                relevant_from_hop_1 = relevant_from_hop_1.split("\', ")
            hop_2_invoke = remove_think_blocks(llm.invoke([HumanMessage(content=llm_reader_prompt(hop2_table_md, hop2_links, question))]).content)
            hop_1_invoke = hop_2_invoke
        else:
            answer = hop_1_invoke
            break

    if "None" in hop_2_invoke:
        relevant_from_hop_1 = hop_1_invoke[hop_1_invoke.index("[")+1:hop_1_invoke.index("]")]
        hop2_table = pd.concat([hop1_table, extract_subtable_from_hop(hopwise_context["2-hop"], subtable_df)], ignore_index=True)
        hop2_table_md = hop2_table.to_markdown()
        hop2_links = [inputs["links"][i] for i in collect_links(hop1_table)]

        if relevant_from_hop_1 != "":
            relevant_from_hop_1 = relevant_from_hop_1.split("\', ")
            hop2_links = hop2_links + relevant_from_hop_1

        hop_2_invoke = remove_think_blocks(llm.invoke([HumanMessage(content=llm_reader_prompt(subtable_df, hop2_links, question))]).content)

    answer = hop_2_invoke

    return answer
    


if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        inputs = json.load(f)
    
    print(f"Answer: {process_single(inputs)}")