import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME", "BankDB"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
}

NEO4J_CONFIG = {
    "uri": os.getenv("NEO4J_URI"),
    "user": os.getenv("NEO4J_USERNAME"),
    "password": os.getenv("NEO4J_PASSWORD"),
}


# Counts for synthetic data generation
COUNTS = {
    "customers": 2000,
    "devices": 1500,
    "beneficiaries": 3000,
    "merchants": 100,
    "customer_devices": 3000,
    "customer_beneficiaries": 5000,
    "sanction_list": 300,
    "transactions": 25000,
    "customer_behavior": 15000,
    "transaction_analysis_logs": 25000,
}

REQUIRED_FILES = [
    "customers.csv",
    "devices.csv",
    "beneficiaries.csv",
    "merchants.csv",
    "customer_devices.csv",
    "customer_beneficiaries.csv",
    "sanction_list.csv",
    "transactions.csv",
    "customer_behavior.csv",
    "transaction_analysis_logs.csv",
]

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
SYNTHETIC_DATA_DIR = os.getenv("DATA_DIR", "data/synthetic")
RAG_DATA_ROOT_DIR = os.getenv("RAG_DATA_DIR", "data/rag")
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # Default to Ollama if not set
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
INGESTION_PIPELINE = [
    ("customers", "customers.csv"),
    ("devices", "devices.csv"),
    ("beneficiaries", "beneficiaries.csv"),
    ("merchants", "merchants.csv"),
    ("customer_devices", "customer_devices.csv"),
    ("customer_beneficiaries", "customer_beneficiaries.csv"),
    ("sanction_list", "sanction_list.csv"),
    ("transactions", "transactions.csv"),
    ("customer_behavior", "customer_behavior.csv"),
    ("transaction_analysis_logs", "transaction_analysis_logs.csv"),
]

TABLE_CREATION_SCHEMA_PATH = str(Path(ROOT_DIR) / "data" / "sql" / "create_tables.sql")
BATCH_SIZE = 2000

MODEL_CONFIG = {
    "preprocessor_path": str(
        Path(ROOT_DIR) / "models" / "artifacts" / "production_preprocessor_pipeline.pkl"
    ),
    "classifier_path": str(
        Path(ROOT_DIR) / "models" / "artifacts" / "transformed_fraud_classifier.pkl"
    ),
    "regressor_path": str(
        Path(ROOT_DIR) / "models" / "artifacts" / "transformed_risk_regressor.pkl"
    ),
}


KAFKA_CONFIG = {
    "bootstrap.servers": os.getenv("BOOTSTRAP_SERVERS"),
    "client.id": os.getenv("CLIENT_ID"),
}

MAX_ITERATIONS = 3
