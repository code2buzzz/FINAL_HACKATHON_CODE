from ..tools.rag_tools import AgentRetriever
from ..tools.llm_factory import LLMFactory

retriever = AgentRetriever("legal_compliance")

llm = LLMFactory.reasoning_llm()


def compliance_node(state):

    tx = state["transaction"]

    context = retriever.search(str(tx))

    prompt = f"""
    Check sanctions, AML and regulatory concerns.

    {tx}

    Context:
    {context}
    """

    result = llm.invoke(prompt)

    return {"compliance_result": {"analysis": result.content}}
