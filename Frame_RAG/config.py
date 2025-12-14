import os
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = "/Users/giuseppebonsignore/Desktop/FrameRAG_v2"
DATA_DIR = "/Users/giuseppebonsignore/Desktop/FrameRAG_v2/data"
TTL_PATH = DATA_DIR + "/premon-2018a-fn15-inf.ttl"
FRAME_CORPUS = DATA_DIR + "/frame_corpus.txt"
INDEX_DIR = DATA_DIR + "/frame_index"
INDEX_VECTORS = INDEX_DIR + "/vectors.npy"

EMBED_MODEL = os.getenv("EMBED_MODEL")
LLM_MODEL = os.getenv("LLM_MODEL")
TOP_K = int(os.getenv("TOP_K"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")