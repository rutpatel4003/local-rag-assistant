import os
from pathlib import Path
from loguru import logger
import sys
class Config:
    SEED=42
    ALLOWED_FILE_EXTENSIONS = set(['.pdf', '.md', '.txt'])

    class Model:
            NAME = 'qwen3:4b'
            TEMPERATURE = 0.7

    class Preprocessing: 
        CHUNK_SIZE = 2048
        CHUNK_OVERLAP = 128
        EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
        RERANKER = 'ms-marco-MiniLM-L-12-v2'
        LLM = ' qwen3:4b'
        CONTEXTUALIZE_CHUNKS = True
        N_SEMANTIC_RESULTS = 5
        N_BM25_RESULTS = 5

    class Chatbot: 
        N_CONTEXT_RESULTS = 3

    class Path:
        APP_HOME = Path(os.getenv("APP_HOME", Path(__file__).parent.parent))
        DATA_DIR = APP_HOME/"data"
        


