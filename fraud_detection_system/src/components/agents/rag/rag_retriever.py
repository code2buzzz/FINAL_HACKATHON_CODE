import os
from langchain_chroma import Chroma
from config.settings import ROOT_DIR
from langchain_huggingface import HuggingFaceEmbeddings


class AgentRetriever:
    """
    Lightweight retrieval interface for LLM tools.
    This is NOT ingestion — only query-time retrieval.
    """

    def __init__(self, collection_name: str):
        self.collection_name = collection_name

        self.embeddings = HuggingFaceEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.persist_directory = os.path.join(ROOT_DIR, "storage", "chroma_storage")

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

    # ----------------------------
    # BASIC SEMANTIC SEARCH
    # ----------------------------
    def search(self, query: str, k: int = 5) -> str:
        """
        Returns formatted context for LLM consumption.
        """

        results = self.vector_store.similarity_search(query, k=k)

        return self._format_results(results)

    # ----------------------------
    # ADVANCED SEARCH (optional but powerful)
    # ----------------------------
    def search_with_score(self, query: str, k: int = 5):
        results = self.vector_store.similarity_search_with_score(query, k=k)

        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),
            }
            for doc, score in results
        ]

    # ----------------------------
    # FORMAT FOR LLM CONTEXT
    # ----------------------------
    def _format_results(self, docs):
        """
        Converts retrieved documents into structured LLM context.
        """

        if not docs:
            return "No relevant behavioral context found."

        formatted = []

        for i, doc in enumerate(docs, 1):
            meta = doc.metadata

            formatted.append(f"""
                [Evidence {i}]
                Content: {doc.page_content}
                Source: {meta.get('source_file', 'unknown')}
                Type: {meta.get('data_type', 'unknown')}
                Category: {meta.get('folder_category', 'unknown')}
                """)

        return "\n".join(formatted)
