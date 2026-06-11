from langchain_core.prompts import ChatPromptTemplate

from ..tools.rag_tools import AgentRetriever
from ..tools.llm_factory import LLMFactory

retriever = AgentRetriever("behavioral_anomalies")

llm = LLMFactory.behavioral_llm()


def behavioral_node(state):

    tx = state["transaction"]

    context = retriever.search(str(tx))

    prompt = ChatPromptTemplate.from_template("""
    Investigate behavioral fraud indicators.

    Transaction:
    {tx}

    Context:
    {context}

    Return:
    score
    reasoning
    confidence
    """)

    chain = prompt | llm

    result = chain.invoke({"tx": tx, "context": context})

    return {"behavioral_result": {"analysis": result.content}}
