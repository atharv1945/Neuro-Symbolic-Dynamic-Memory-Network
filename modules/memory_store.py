import os
import json
import pickle
import threading
import shutil
import time
import numpy as np
import networkx as nx
import faiss

# Import configuration
from config import (
    DATA_DIR, 
    GRAPH_FILENAME, 
    INDEX_FILENAME, 
    ID_MAP_FILENAME,
    FAISS_DEVICE,
    LOG_DIR,
    EMBEDDING_DIM
)
from modules.utils import logger, atomic_dir_swap

class SharedMemoryManager:
    """
    Thread-safe manager for Split-Compute Memory (Graph + Vector Store).
    Enforces atomic snapshots and explicit ID mapping.
    """
    def __init__(self, embedding_dim=EMBEDDING_DIM): 
        self.lock = threading.RLock()
        self.embedding_dim = embedding_dim
        
        # ID Management
        self.uuid_to_id = {} # str -> int
        self.id_to_uuid = {} # int -> str
        self.next_id = 0
        
        # Data Structures
        self.graph = nx.Graph()
        self.vectors = {} # uuid -> np.ndarray
        
        # FAISS Setup (CPU Force)
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.embedding_dim))
        
        self._load_state()

    def _load_state(self):
        """Loads data from DATA_DIR if it exists."""
        if not DATA_DIR.exists():
            logger.info("No existing memory found. Initializing empty Main Memory.")
            return

        graph_path = DATA_DIR / GRAPH_FILENAME
        index_path = DATA_DIR / INDEX_FILENAME
        map_path = DATA_DIR / ID_MAP_FILENAME

        try:
            with self.lock:
                if graph_path.exists():
                    with open(graph_path, 'rb') as f:
                        self.graph = pickle.load(f)
                
                if map_path.exists():
                    with open(map_path, 'r') as f:
                        saved_map = json.load(f)
                        self.id_to_uuid = {int(k): v for k, v in saved_map.items()}
                        self.uuid_to_id = {v: k for k, v in self.id_to_uuid.items()}
                        if self.id_to_uuid:
                            self.next_id = max(self.id_to_uuid.keys()) + 1
                        else:
                            self.next_id = 0

                if index_path.exists():
                    self.index = faiss.read_index(str(index_path))

                vectors_path = DATA_DIR / "vectors.pkl"
                if vectors_path.exists():
                    with open(vectors_path, 'rb') as f:
                        self.vectors = pickle.load(f)
                
                # Atomic Persistence Check: Verify FAISS and vectors are in sync
                if len(self.uuid_to_id) != len(self.vectors):
                    logger.warning(
                        f"Persistence mismatch detected! UUID map: {len(self.uuid_to_id)}, "
                        f"Vectors: {len(self.vectors)}. This may indicate incomplete write."
                    )
                    
            logger.info(f"Memory Loaded. Graph Nodes: {self.graph.number_of_nodes()}, Vectors: {self.index.ntotal}")
            
        except Exception as e:
            logger.critical(f"Failed to load memory state: {e}. Starting fresh to prevent corruption.")

    def save_snapshot(self):
        """
        Atomically saves the current state to disk.
        Holds LOCK during the entire Write+Swap process to ensure consistency.
        """
        temp_dir_name = f"data_tmp_{os.getpid()}"
        temp_dir = DATA_DIR.parent / temp_dir_name
        
        # Ensure temp dir exists
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with self.lock:
                logger.info("Acquired Lock. Starting Memory Snapshot...")
                
                # 1. Serialize Graph
                with open(temp_dir / GRAPH_FILENAME, 'wb') as f:
                    pickle.dump(self.graph, f)
                
                # 2. Serialize ID Map
                with open(temp_dir / ID_MAP_FILENAME, 'w') as f:
                    json.dump(self.id_to_uuid, f)
                
                # 3. Serialize FAISS
                faiss.write_index(self.index, str(temp_dir / INDEX_FILENAME))

                # 4. Serialize Vectors
                with open(temp_dir / "vectors.pkl", 'wb') as f:
                    pickle.dump(self.vectors, f)
                
                # Atomic Persistence Check: Verify consistency before swap
                if len(self.uuid_to_id) != len(self.vectors):
                    logger.error(
                        f"Pre-swap validation failed! UUID map: {len(self.uuid_to_id)}, "
                        f"Vectors: {len(self.vectors)}. Aborting snapshot to prevent corruption."
                    )
                    return
                
                logger.info("Serialization complete. Swapping directories...")
                
                # 5. Atomic Swap
                if atomic_dir_swap(DATA_DIR, temp_dir):
                    logger.info("Snapshot Saved Successfully.")
                else:
                    logger.error("Snapshot Swap Failed.")
                    
        except Exception as e:
            logger.error(f"Snapshot Failed with Exception: {e}")
        finally:
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass

    def add_memory(self, uuid: str, text: str, vector: np.ndarray, metadata: dict = None):
        """
        Atomically adds a node to Graph and Vector Store.
        """
        # Ensure numpy array
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector)
            
        vector = vector.reshape(1, -1).astype('float32')
        
        with self.lock:
            try:
                # 1. Assign ID
                current_id = self.next_id
                self.next_id += 1
                
                # 2. Add to FAISS
                self.index.add_with_ids(vector, np.array([current_id]).astype('int64'))
                
                # 3. Add to Graph
                if metadata is None: metadata = {}
                metadata['text'] = text
                metadata['energy'] = 10.0 
                metadata['last_access'] = time.time()
                
                self.graph.add_node(uuid, **metadata)
                
                # 4. Update Maps and Vector Storage
                self.id_to_uuid[current_id] = uuid
                self.uuid_to_id[uuid] = current_id
                self.vectors[uuid] = vector.flatten() # Store as flat array
                
                return True
            except Exception as e:
                logger.error(f"Failed to add memory: {e}")
                return False

    def remove_node(self, uuid: str):
        """Removes a node from Graph and Index."""
        with self.lock:
            if uuid not in self.uuid_to_id:
                return False
            
            internal_id = self.uuid_to_id[uuid]
            
            try:
                self.graph.remove_node(uuid)
                self.index.remove_ids(np.array([internal_id]).astype('int64'))
                del self.uuid_to_id[uuid]
                del self.id_to_uuid[internal_id]
                if uuid in self.vectors:
                    del self.vectors[uuid]
                return True
            except Exception as e:
                logger.error(f"Error removing node {uuid}: {e}")
                return False
                
    def query_similarity(self, vector: np.ndarray, top_k: int = 5):
        """
        Returns a list of (uuid, distance) tuples.
        """
        # Ensure numpy array
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector)

        vector = vector.reshape(1, -1).astype('float32')
        with self.lock:
            if self.index.ntotal == 0:
                return []
            
            distances, ids = self.index.search(vector, top_k)
            
            results = []
            for dist, internal_id in zip(distances[0], ids[0]):
                if internal_id != -1 and internal_id in self.id_to_uuid:
                    results.append((self.id_to_uuid[internal_id], float(dist)))
            
            return results

    def get_node_content(self, uuid: str):
        """Thread-safe graph read."""
        with self.lock:
            if self.graph.has_node(uuid):
                data = self.graph.nodes[uuid]
                data['last_access'] = time.time() 
                return data.get('text', "")
            return None
