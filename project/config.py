# config.py
class Config:
    # Dataset paths
    RAW_DATA_DIR = "./data/raw/"
    PROCESSED_DATA_DIR = "./data/processed/"
    ANNOTATED_DATA_DIR = "./data/annotated/"
    
    # Model paths
    BASE_MODEL = "gpt-3.5-turbo"  # or any Hugging Face model
    FINE_TUNED_MODEL_DIR = "./models/fine_tuned/"
    
    # RAG settings
    FAISS_INDEX_PATH = "./retrieval/faiss_index/"
    KNOWLEDGE_BASE_DIR = "./retrieval/knowledge_base/"
    
    # Metrics and monitoring
    METRICS_DIR = "./evaluation/metrics/"
    LOG_DIR = "./logs/"

    # Deployment settings
    API_HOST = "0.0.0.0"
    API_PORT = 8000
