import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_neo4j import Neo4jGraph
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.components.agents.llms.llm_factory import LLMFactory
from config.settings import NEO4J_CONFIG

llm = LLMFactory.graph_llm()

graph = Neo4jGraph(
    url=NEO4J_CONFIG["uri"],
    username=NEO4J_CONFIG["user"],
    password=NEO4J_CONFIG["password"],
)

# 2. Few-Shot Cypher Generation Strategy
few_shot_examples = [
    {
        "question": "Identify the customer who did the most transactions.",
        "cypher": "MATCH (c:Customer)-[:MADE_TRANSACTION]->(t:Transaction) RETURN c.customer_id AS customer_id, c.customer_name AS customer_name, count(t) AS transaction_count ORDER BY transaction_count DESC LIMIT 1",
    },
    {
        "question": "Find high-risk customers who have made international transactions over 10000.",
        "cypher": "MATCH (c:Customer)-[:MADE_TRANSACTION]->(t:Transaction) WHERE c.customer_risk_rating = 'HIGH' AND t.transaction_amount > 10000 AND t.is_international = true RETURN c.customer_id AS customer_id, t.transaction_id AS transaction_id, t.transaction_amount AS amount, t.origin_country AS origin, t.destination_country AS destination",
    },
    {
        "question": "Check if any customer has sent money to a beneficiary matched with a Sanction Entity.",
        "cypher": "MATCH (c:Customer)-[:MADE_TRANSACTION]->(t:Transaction)-[:TO_BENEFICIARY]->(b:Beneficiary)-[:SANCTION_MATCH]->(s:SanctionEntity) RETURN c.customer_id AS customer_id, c.customer_name AS customer_name, b.receiver_name AS beneficiary_name, s.entity_name AS sanction_entity, s.sanction_category AS reason",
    },
    {
        "question": "Identify devices that are being shared by more than one customer account.",
        "cypher": "MATCH (c:Customer)-[:HAS_DEVICE]->(d:Device) WITH d, count(c) AS customer_count WHERE customer_count > 1 MATCH (c2:Customer)-[:HAS_DEVICE]->(d) RETURN d.device_id AS device_id, d.device_type AS device_type, collect(distinct c2.customer_id) AS shared_by_customers, customer_count",
    },
    {
        "question": "Find transactions processed at a merchant with a high fraud history count.",
        "cypher": "MATCH (t:Transaction)-[:AT_MERCHANT]->(m:Merchant) WHERE m.fraud_transaction_count > 5 RETURN t.transaction_id AS transaction_id, m.merchant_name AS merchant_name, m.merchant_risk_rating AS merchant_risk, t.transaction_amount AS amount",
    },
    {
        "question": "Get the structural profile and total connection count for a specific customer ID C1001.",
        "cypher": "MATCH (c:Customer {{customer_id: 'C1001'}}) RETURN c.customer_name AS name, c.customer_risk_rating AS risk, count{{(c)--()}} AS total_graph_connections",
    },
    {
        "question": "Look for transactions where account takeover is suspected and flag high risk scores.",
        "cypher": "MATCH (c:Customer)-[:MADE_TRANSACTION]->(t:Transaction) WHERE t.account_takeover_suspected = true OR t.overall_risk_score > 0.8 RETURN c.customer_id AS customer_id, t.transaction_id AS transaction_id, t.overall_risk_score AS risk_score, t.recommended_action AS action",
    },
]

example_prompt = PromptTemplate(
    input_variables=["question", "cypher"],
    template="\nQuestion: {question}\nCypher Query: {cypher}\n",
)

prefix = """You are a Neo4j Cypher expert. Given an input question, your job is to write a valid, read-only Neo4j 5.x Cypher query.

Schema:
{schema}

Rules:
- Generate exactly ONE Cypher query statement.
- Never use CREATE, MERGE, DELETE, SET, or CALL (apoc/gds).
- Return ONLY the executable Cypher query string. Do NOT wrap it in backticks, markdown, or add conversational filler.
- Always prefer modern element-count syntax: Use `COUNT {{ (n)--() }}` instead of `SIZE((n)--())`.

Here are examples of correct conversions:"""

cypher_prompt = FewShotPromptTemplate(
    examples=few_shot_examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix="\nQuestion: {question}\nCypher Query:",
    input_variables=["schema", "question"],
)

cypher_generation_chain = cypher_prompt | llm | StrOutputParser()


# 3. Native Agent Tool Export
@tool
def query_graph_database(question: str) -> str:
    """
    Executes a read-only natural language query against the graph database.
    Use this to look up specific entities, trace paths, inspect transaction details,
    count relationships, or fetch sequential steps for an ongoing investigation.
    Input must be a highly specific standalone question.
    """
    time.sleep(0.5)  # Rate-limit defense for Groq

    try:
        current_schema = graph.schema
        generated_cypher = cypher_generation_chain.invoke(
            {"schema": current_schema, "question": question}
        )
        clean_cypher = (
            generated_cypher.replace("```cypher", "").replace("```", "").strip()
        )

        print(f"\n[Generated Cypher]: {clean_cypher}")
        db_result = graph.query(clean_cypher)

        if not db_result:
            return "Investigation Note: No data found in the database for this query configuration."

        return str(db_result)

    except Exception as e:
        return f"Investigation Note: Query failed to execute. Error: {str(e)}."
