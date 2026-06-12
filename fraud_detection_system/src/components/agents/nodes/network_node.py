import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

from src.components.agents.llms.llm_factory import LLMFactory
from src.components.agents.rag.rag_retriever import AgentRetriever

# Initialize retriever once
rag_retriever = AgentRetriever("network_typologies")


# Initialize MCP
client = MultiServerMCPClient(
    {
        "network_graph_server": {
            "transport": "stdio",
            "command": "python",
            "args": ["src/mcp_servers/graph_mcp_server.py"],
        }
    }
)

mcp_tools = None


# Get MCP tools
async def get_tools():
    global mcp_tools
    if mcp_tools is None:
        mcp_tools = await client.get_tools()
    return mcp_tools


async def run_network_agent(tx):

    llm = LLMFactory.graph_llm()
    tools = await get_tools()

    # Create agent
    agent = create_agent(llm, tools=tools)

    # Get RAG context
    rag_context = rag_retriever.search(str(tx))

    query = {
        "messages": [
            (
                "user",
                f"""
                Transaction:
                {tx}

                RAG CONTEXT:
                {rag_context}

                Use MCP tools if graph analysis is required.
                """,
            )
        ]
    }

    result = await agent.ainvoke(query)

    return {"network_result": {"analysis": result["messages"][-1].content}}


def network_node(state):
    return asyncio.run(run_network_agent(state["transaction"]))
