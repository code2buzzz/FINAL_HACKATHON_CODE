import os
from langchain_chroma import Chroma
from config.settings import ROOT_DIR
from langchain_huggingface import HuggingFaceEmbeddings


class AgentRetriever:
    """
    Lightweight retrieval interface for LLM tools.
    This is NOT ingestion — only query-time retrieval.
    """

    _instance = None
    _initialized = False

    def __new__(cls):

        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance

    def __init__(self):

        if self.__class__._initialized:
            return

        # your original code unchanged

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.persist_directory = os.path.join(
            ROOT_DIR,
            "storage",
            "chroma_storage",
        )

        self.vector_store = Chroma(
            collection_name="fraud_documents_knowledge_base",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

        self.__class__._initialized = True

    # ----------------------------
    # BASIC SEMANTIC SEARCH (DYNAMIC FILTER)
    # ----------------------------
    def search(self, query: str, category: str, k: int = 5) -> str:
        """
        Returns formatted context for LLM consumption, filtered dynamically by the provided category.
        """
        # The filter keyword is now driven entirely by your runtime function argument
        results = self.vector_store.similarity_search(
            query, k=k, filter={"folder_category": category}
        )

        return self._format_results(results, category)

    # ----------------------------
    # ADVANCED SEARCH (DYNAMIC FILTER)
    # ----------------------------
    def search_with_score(self, query: str, category: str, k: int = 5):
        """
        Returns documents with similarity scores, filtered dynamically by the provided category.
        """
        results = self.vector_store.similarity_search_with_score(
            query, k=k, filter={"folder_category": category}
        )

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
    def _format_results(self, docs, category: str):
        """
        Converts retrieved documents into structured LLM context.
        """
        if not docs:
            return f"No relevant behavioral context found for category '{category}'."

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
