# NS-DMN Configuration

import os
from pathlib import Path

# --- Base Paths ---
PROJ_ROOT = Path(__file__).resolve().parent
DATA_DIR_NAME = "data"
DATA_DIR = PROJ_ROOT / DATA_DIR_NAME
LOG_DIR = PROJ_ROOT / "logs"
LOG_FILE = LOG_DIR / "ns_dmn.log"

# --- Hardware / Device Limits ---
# Strict "Split-Compute" Enforcement
LLM_LINGUA_DEVICE = "cpu"  # Force Compression to CPU
FAISS_DEVICE = "cpu"       # Force Vector Search to CPU

# --- Memory Constants ---
GRAPH_FILENAME = "graph_data.pkl"
INDEX_FILENAME = "vector_store.index"
ID_MAP_FILENAME = "id_map.json"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# --- Logic Constants ---
ENTROPY_DECAY_RATE = 0.1     # Point deduction per hour
PRUNING_THRESHOLD = 0.0      # Nodes with energy <= 0 are removed
SIMILARITY_THRESHOLD = 75    # rapidfuzz ratio for merging
COMPRESSION_TARGET_TOKENS = 1000

# --- Titans-Inspired Constants ---
SURPRISE_THRESHOLD = 0.92      # Similarity > 0.92 = redundant (skip entity extraction)
REM_MIN_CLIQUE_SIZE = 3        # Minimum clique size for concept creation
REM_MOMENTUM_THRESHOLD = 5     # Momentum threshold for early consolidation

# --- Operational Constants ---
SAVE_INTERVAL_SECONDS = 600  # Dreamer saves every 10 mins
DREAMER_BATCH_SIZE = 5       # Process N items from STM before checking maintenance
BRAIN_INFERENCE_TIMEOUT = 30 # Seconds to wait for Ollama (optional usage)

# --- Ensure Directories Exist ---
LOG_DIR.mkdir(parents=True, exist_ok=True)
