import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

    # OLLAMA_BASE_URL = os.getenv(
    #     "OLLAMA_BASE_URL",
    #     "http://localhost:11434"
    # )


settings = Settings()
