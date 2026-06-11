from ..tools.graph_tools import investigate_network
from ..tools.rag_tools import AgentRetriever
from ..tools.llm_factory import LLMFactory

retriever = AgentRetriever("network_typologies")

llm = LLMFactory.graph_llm()


def network_node(state):

    tx = state["transaction"]

    graph_result = investigate_network(tx)

    rag_context = retriever.search(str(tx))

    prompt = f"""
    Analyze mule network risk.

    Transaction:
    {tx}

    Graph:
    {graph_result}

    Context:
    {rag_context}

    Return:
    score
    confidence
    reasoning
    """

    result = llm.invoke(prompt)

    return {"network_result": {"analysis": result.content}}
