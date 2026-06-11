from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama


from config.settings import LLM_PROVIDER, GROQ_API_KEY

load_dotenv()
api_key = GROQ_API_KEY


class LLMFactory:

    @staticmethod
    def behavioral_llm():

        if LLM_PROVIDER == "groq":
            return ChatGroq(
                model="llama-3.3-70b-versatile", temperature=0, groq_api_key=api_key
            )

        return ChatOllama(model="llama3")

    @staticmethod
    def reasoning_llm():

        if LLM_PROVIDER == "groq":
            return ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=api_key)

        return ChatOllama(model="deepseek-r1")

    @staticmethod
    def graph_llm():

        if LLM_PROVIDER == "groq":
            return ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=api_key)

        return ChatOllama(model="deepseek-r1")
