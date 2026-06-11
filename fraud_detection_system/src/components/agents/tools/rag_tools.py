from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


class AgentRetriever:

    def __init__(self, collection_name):

        # self.vdb = Chroma(
        #     collection_name=collection_name, persist_directory="storage/chroma_storage"
        # )
        pass

    def search(self, query):

        # docs = self.vdb.similarity_search(query, k=5)

        # return "\n".join([d.page_content for d in docs])
        pass
