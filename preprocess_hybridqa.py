import json
import os
import sys


if len(sys.argv) < 2:
    print("Usage: python preprocess_hybridqa.py <input_questions.json> [question_id]")
    sys.exit(1)

input_file = sys.argv[1]
target_question_id = sys.argv[2] if len(sys.argv) == 3 else None
output_folder = "preprocessed_for_hybrid_graphs"

tables_dir = "WikiTables-WithLinks/tables_tok"
links_dir = "WikiTables-WithLinks/request_tok"

os.makedirs(output_folder, exist_ok=True)

with open(input_file, "r", encoding="utf-8") as f:
    questions_data = json.load(f)

if not isinstance(questions_data, list):
    print("Input JSON must be a list of objects.")
    sys.exit(1)

output_items = []

for item in questions_data:
    q_id = item.get("question_id")
    question = item.get("question")
    table_id = item.get("table_id")
    answer = item.get("answer")

    if not all([q_id, question, table_id]):
        print(f"Skipping item with missing fields: {item}")
        continue

    if target_question_id and q_id != target_question_id:
        continue

    new_item = {
        "question_id": q_id,
        "question": question,
        "table_id": table_id,
        "answer": answer
    }

    table_path = os.path.join(tables_dir, f"{table_id}.json")
    if os.path.exists(table_path):
        with open(table_path, "r", encoding="utf-8") as f:
            new_item["table"] = json.load(f)
    else:
        print(f"Warning: table file not found: {table_path}")
        new_item["table"] = None

    links_path = os.path.join(links_dir, f"{table_id}.json")
    if os.path.exists(links_path):
        with open(links_path, "r", encoding="utf-8") as f:
            new_item["links"] = json.load(f)
    else:
        print(f"Warning: links file not found: {links_path}")
        new_item["links"] = None

    output_items.append(new_item)

    if target_question_id:
        break

if not output_items:
    print("No matching items found.")
    sys.exit(1)

if target_question_id:
    output_filename = os.path.join(output_folder, f"{target_question_id}.json")
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(output_items[0], f, indent=2, ensure_ascii=False)
    print(f"Saved single item to {output_filename}")
else:
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_filename = os.path.join(output_folder, f"{base_name}_enriched.json")
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(output_items, f, indent=2, ensure_ascii=False)
    print(f"Saved enriched list to {output_filename}")