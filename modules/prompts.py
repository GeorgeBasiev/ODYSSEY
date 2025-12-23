def prompt_ent_extr_from_q(question: str, table_headers: str) -> str:
    """
    Generates a prompt for named entity extraction from the question.
    """
    return f"""Agent Introduction: You are an agent assisting me in a multi-table question answering task.  
Your role is to extract meaningful named and descriptive entities directly from the question.  
These entities will later be used to align with relevant columns across multiple tables (provided in the format "table_name.column_name").

Task: Identify and list all key entities present in the question. Include:
- Proper nouns (e.g., events, years, names, locations),
- Descriptive phrases that represent attributes or conditions (e.g., "gold medal", "nickname", "men’s heavyweight"),
- Temporal references (e.g., "1932", "Summer Olympics"),
- Domain-specific terms relevant to the tables (e.g., "Greco-Roman Wrestling").

Do not filter or omit entities based on whether they appear in the headers—your goal is to capture all potentially useful query elements.

Output format:
Entities: ["<entity1>", "<entity2>", ...]

Use the example below for guidance:

Input:
Question: What was the birthplace of the director of the movie that won Best Picture in 2005?
Table Headers: ["movies.id", "movies.title", "movies.year", "movies.award", "directors.id", "directors.name", "directors.birthplace"]
Output:
Entities: ["birthplace", "director", "movie", "Best Picture", "2005"]

Now process the following:

Input:
Question: {question}
Table Headers: {table_headers}
Output:"""


def prompt_relevant_header(question: str, table_headers: str, entities: str) -> str:
    """
    Generates a prompt for identifying relevant table headers based on the question and entities.
    """
    return f"""Agent Introduction: You are an agent assisting me in a multi-table question answering task.  
I have multiple tables as sources of information and have already extracted relevant entities from the question.  
Your job is to identify which fully qualified column names (in the format table_name.column_name) are relevant for answering the question.

Task: From the provided list of column names in the format "table_name.column_name", select all columns that are semantically or contextually relevant to the question or the extracted entities.

Important Rules:
1. **Always include identifier columns**: If any table contains a column that serves as a unique row identifier—such as "id", "ID", "Id", "row_id", "index", "identifier", "record_id", "entry_id", "key", "serial", "ref", "uid", or any similar name—you MUST include that column (with its table prefix) in the output list, even if it is not directly referenced in the question or extracted entities.
2. **Why this matters**: Identifier columns are essential for correctly joining tables, filtering specific records, or tracing answers back to source rows—critical steps in a multi-table reasoning pipeline.
3. **Include all semantically relevant columns**: Beyond identifiers, select any column that may contain information matching the question’s intent or the extracted entities (e.g., names, events, categories, dates, outcomes, locations, etc.), regardless of which table it belongs to.
4. **CRITICAL - Return only existing columns**: You MUST output ONLY column names that exist in the provided Table Headers list. Do NOT invent, modify, or truncate any table or column names. Return them exactly as they appear in the input Table Headers.

Output format:
Relevant headers: ["table1.columnA", "table2.columnB", ...]

Example for clarity:

Input:
Question: What was the birthplace of the director of the movie that won Best Picture in 2005?
Table Headers: ["movies.id", "movies.title", "movies.year", "movies.award", "directors.id", "directors.name", "directors.birthplace", "movies.director_id"]
Entities extracted from question: ["Best Picture", "2005", "director", "birthplace"]
Output:
Relevant headers: ["movies.id", "movies.award", "movies.year", "movies.director_id", "directors.id", "directors.birthplace"]

Don't add any comments in the output, follow the schema!
Return table and column names strictly same, as it was in the input

Now process the following:

Input:
Question: {question}
Table Headers: {table_headers}
Entities extracted from question: {entities}

IMPORTANT: Only return column names that exist in the Table Headers list above. Do not create new column names.
Output:"""


def prompt_entity_header_mapping(question: str, entities: str, relevant_headers: str) -> str:
    """
    Generates a prompt for mapping entities to relevant table headers.
    """
    return f"""Agent Introduction: You are an agent assisting me in a multi-table question answering task.  
I have already extracted relevant entities from the question and identified the relevant columns across multiple tables in the format "table_name.column_name".  
Your task is to map each extracted entity to one or more of these relevant columns based on semantic or contextual alignment.

Task: For each entity from the question, assign it to the most appropriate relevant column(s) (e.g., "athletes.name", "events.year").  
If none of the relevant columns meaningfully correspond to an entity, map that entity to "Others".

Important Notes:
- Column names are fully qualified (e.g., "movies.title", "directors.birthplace"), so consider both the table context and the column name when mapping.
- An entity may map to multiple columns if justified (e.g., "2005" might match both "movies.year" and "awards.ceremony_year").
- Be precise: only map an entity to a column if the column is likely to contain or represent that kind of information.

Output format:
"<entity1>": ["table1.columnA", "table2.columnB"],
"<entity2>": ["Others"]

Use the example below to understand the expected behavior:

Input:
Question: What was the birthplace of the director of the movie that won Best Picture in 2005?
Entities extracted from question: ["birthplace", "director", "Best Picture", "2005"]
Relevant Headers: ["movies.id", "movies.award", "movies.year", "movies.director_id", "directors.id", "directors.birthplace"]
Output:
"birthplace": ["directors.birthplace"],
"director": ["movies.director_id", "directors.id"],
"Best Picture": ["movies.award"],
"2005": ["movies.year"]

Return output as a valid JSON.
Now process the following:

Input:
Question: {question}
Entities extracted from question: {entities}
Relevant Headers: {relevant_headers}
Output:"""


def find_links_prompt(tables):
    return f"""Analyze the relationships between the columns in these tables. Find pairs of related columns by:
1. Common column names (customerid, employeeid, orderid)
2. Matching data values, semantic relationships between data in column cells, and similarity in data format

Examples of expected relationships:
- orders.customerid is related to customers.customerid
- orders.employeeid is related to employees.employeeid
If the columns clearly contain the same data but have slightly different names

Return the result in JSON format, don't add any additional text
{{
  "sales.customer_id": "clients.client_id",
  "sales.employee_code": "staff.staff_id",
  "sales.invoice_number": "payments.invoice_ref"
}}

Table data:

{tables}"""


def llm_reader_prompt(table_data: str, question: str) -> str:
    """
    Generates a prompt for the final LLM reader step.
    """
    return f"""Agent Introduction: Hello! I’m your MultiTable-QA expert agent, here to assist you in answering
    complex questions by analyzing and integrating information from multiple tables.
    Task: Your task is to answer the given question using the provided set of tables.

    Final Answer: Provide short final answer in the format below. If the answer cannot be determined
    from the given tables, provide None.
    Final Answer Format: (valid JSON)
    {{"answer": answer}}

    
    Here’s the context you’ll need:
    Table Data: {table_data}
    Question: {question}
    """