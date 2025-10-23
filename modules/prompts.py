def prompt_ent_extr_from_q(question: str, table_id: str, table_headers: str) -> str:
    """
    Generates a prompt for named entity extraction from the question.
    """
    return f"""Agent Introduction: You are an agent who is going to be assisting me in a question answering
    task. For this task, I need to first identify the named entities in the question.
    Task: Identify the named entities in the provided question. These entities will serve as key elements
    for extracting pertinent information from the available sources, which include table name and its
    headers.
    Output format:
    Entities: [‘<entity1>’, ‘<entity2>’, .....]
    Use the below example to better understand the task
    Input:
    Question: What was the nickname of the gold medal winner in the men’s heavyweight Greco-
    Roman wrestling event of the 1932 Summer Olympics?
    Table Name: Sweden at the 1932 Summer Olympics
    Table Headers: ["Medal", "Name", "Sport", "Event"]
    Output:
    Entities: [‘nickname’, ‘medal’, ‘gold’, ‘men’s heavyweight’,
    ‘Greco-Roman Wrestling event’, ‘1932 Summer Olympics’]
    Input:
    Question: {question}
    Table Name: {table_id}
    Table Headers: {table_headers}
    Output:"""


def prompt_relevant_header(question: str, table_id: str, table_headers: str, entities: str) -> str:
    """
    Generates a prompt for identifying relevant table headers based on the question and entities.
    """
    return f"""Agent Introduction: You are an agent who is going to be assisting me in a question answering
    task. I have a table as a source of information. I have already extracted the relevant entities from the
    question. For this task, I need to first identify the column headers that are relevant in the question.
    Task: Identify the relevant column headers from the provided list, based on the extracted entities
    from the question. I will also provide the extracted entities from the question and name of the table.
    Output format:
    Relevant headers: [‘<header-1>’, ‘<header-2>’, ....]
    Use the below example to better understand the task
    Input:
    Question: What was the nickname of the gold medal winner in the men’s heavyweight Greco-
    Roman wrestling event of the 1932 Summer Olympics?
    Table Name: Sweden at the 1932 Summer Olympics
    Table Headers: ["Medal", "Name", "Sport", "Event"]
    Entities extracted from question: ["gold medal", "men’s heavyweight", "Greco-Roman Wrestling",
    "1932 Summer Olympics"]
    Output:
    Relevant headers: ["Medal", "Name", "Sport", "Event"]
    Input:
    Question: {question}
    Table Name: {table_id}
    Table Headers: {table_headers}
    Entities extracted from question: {entities}
    Output:"""


def prompt_entity_header_mapping(question: str, table_id: str, entities: str, relevant_headers: str) -> str:
    """
    Generates a prompt for mapping entities to relevant table headers.
    """
    return f"""Agent Introduction: You are an agent who is going to be assisting me in a question answering
    task. I have a table as a source of information. I have already extracted relevant entities from the
    question and relevant column headers from the table.
    Task: Map the entities extracted from the question with the relevant headers and the table name.
    Output format:
    "<entity1>": ["<mapping1>", "<mapping2>"],
    "<entity2>": ["<mapping1>"]
    For each entity extracted from the question, there should be a corresponding <mapping> to an item
    in the ‘Relevant headers’ column. If none of the headers match the entity, the mapping should be
    labeled as "Others".
    Use the below example to better understand the task
    Input:
    Question: What was the nickname of the gold medal winner in the men’s heavyweight Greco-
    Roman wrestling event of the 1932 Summer Olympics?
    Table Name: Sweden at the 1932 Summer Olympics
    Entities extracted from question: ["gold medal", "men’s heavyweight", "Greco-Roman Wrestling",
    "1932 Summer Olympics"]
    Relevant headers: ["Medal", "Name", "Sport", "Event"]
    Output:
    "gold medal": ["Medal"],
    "men’s heavyweight": ["Event"],
    "Greco-Roman Wrestling": ["Sport"],
    "1932 Summer Olympics": ["Others"]
    Input:
    Question: {question}
    Table Name: {table_id}
    Entities extracted from question: {entities}
    Relevant Headers: {relevant_headers}
    Output:"""


def llm_reader_prompt(table_data: str, passages: list, question: str) -> str:
    """
    Generates a prompt for the final LLM reader step.
    """
    return f"""Agent Introduction: Hello! I’m your Hybrid-QA expert agent, here to assist you in answering
    complex questions by leveraging both table data and passage information. Let’s combine these
    sources to generate accurate and comprehensive answers!
    Task: Your task involves a central question that requires information from both a table and passages.
    Here’s the context you’ll need:
    Table Data: {table_data}
    Passages: {passages}
    Question: {question}
    Final Answer: Provide the final answer in the format below. If the answer cannot be answered
    with the given context, provide None.
    Final Answer Format:
    Final Answer: <your answer>
    If the final answer is "None", provide the names of passages that are relevant to the above questions.
    If no passages are relevant give ‘[]’ as Relevant Passages.
    Relevant Passages Format:
    Relevant Passages: [‘<name-of-passage1>’, ‘<name-of-passage2>’, ......]"""