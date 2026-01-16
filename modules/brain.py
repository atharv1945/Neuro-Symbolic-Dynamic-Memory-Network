import logging
import queue
import time
from typing import List, Tuple
import numpy as np
from rapidfuzz import fuzz

try:
    from llmlingua import PromptCompressor
except ImportError:
    PromptCompressor = None

from config import (
    LLM_LINGUA_DEVICE,
    COMPRESSION_TARGET_TOKENS,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIM
)
from modules.utils import logger
from modules.memory_store import SharedMemoryManager
from modules.reasoner import CognitiveRouter

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

class NeuralBrain:
    def __init__(self, memory_manager: SharedMemoryManager, stm_queue: queue.Queue):
        self.memory = memory_manager
        self.stm_queue = stm_queue
        # Initialize Reasoner
        self.router = CognitiveRouter(model_name="llama3")
        
        if SentenceTransformer:
            logger.info(f"Initializing Embedding Model ({EMBEDDING_MODEL_NAME})...")
            self.encoder = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu')
        else:
            logger.warning("SentenceTransformer not found. Vector search disabled.")
            self.encoder = None

        if PromptCompressor:
            logger.info("Initializing LLMLingua... (This may take a moment)")
            try:
                self.compressor = PromptCompressor(
                    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                    device_map=LLM_LINGUA_DEVICE,
                    model_config={"revision": "main"},
                    use_llmlingua2=True
                )
                logger.info("LLMLingua Initialized.")
            except Exception as e:
                logger.error(f"Failed to load LLMLingua: {e}. Running in Fallback Mode.")
                self.compressor = None
        else:
            logger.warning("LLMLingua not installed. Compression disabled.")
            self.compressor = None

    def check_safety(self, query: str) -> bool:
        forbidden = ["ignore previous instructions", "system prompt", "delete all"]
        start_lower = query.lower()
        for phrase in forbidden:
            if phrase in start_lower:
                logger.warning(f"Safety Triggered: {phrase}")
                return False
        return True

    def retrieve_context(self, query: str, vector_embedding: np.ndarray, strategy: str = 'vector') -> List[str]:
        context_docs = []
        
        # 1. Vector Search (Always done to find entry points)
        hits = self.memory.query_similarity(vector_embedding, top_k=5)
        
        # Collect distinct UUIDs from hits
        hit_uuids = [uuid for uuid, score in hits]
        
        # Basic content from hits
        for uuid in hit_uuids:
             content = self.memory.get_node_content(uuid)
             if content:
                 context_docs.append(f"[Vector Match]: {content}")
                 
        # 2. Graph Traversal
        if strategy in ['graph', 'hybrid']:
            graph_context = self.retrieve_graph_context(hit_uuids)
            context_docs.extend(graph_context)
                
        return list(set(context_docs))

    def retrieve_graph_context(self, entry_uuids: List[str]) -> List[str]:
        """
        Retrieves 1-hop neighbors for the given entry nodes.
        """
        graph_docs = []
        visited = set(entry_uuids)
        
        with self.memory.lock:
            for uuid in entry_uuids:
                if self.memory.graph.has_node(uuid):
                    # Get neighbors
                    neighbors = list(self.memory.graph.neighbors(uuid))
                    for neighbor_id in neighbors:
                        if neighbor_id not in visited:
                            # Get edge data (relation)
                            edge_data = self.memory.graph.get_edge_data(uuid, neighbor_id)
                            relation = edge_data.get('relation', 'related_to')
                            
                            # Get neighbor content
                            neighbor_content = self.memory.get_node_content(neighbor_id)
                            
                            # Format: Node --relation--> Neighbor
                            # We might need the source node text too to make sense
                            # Simplifying: just adding neighbor text with relation note
                            if neighbor_content:
                                graph_docs.append(f"[Graph Link: {relation}] {neighbor_content}")
                            
                            visited.add(neighbor_id)
        return graph_docs

    def compress_context(self, context: List[str], query: str) -> str:
        if not context:
            return ""
        
        full_text = "\n\n".join(context)
        
        if self.compressor:
            try:
                compressed_res = self.compressor.compress_prompt(
                    context=context,
                    question=query,
                    rate=0.5,
                    target_token=COMPRESSION_TARGET_TOKENS
                )
                return compressed_res['compressed_prompt']
            except Exception as e:
                logger.error(f"Compression failed: {e}")
                return full_text[:2000]
        else:
            return full_text[:2000]

    def process_query(self, query: str) -> str:
        if not self.check_safety(query):
            return "Error: Unsafe Query Detected."
            
        # 1. Reasoner Step
        refined_query, route, thought = self.router.analyze_query(query)
        logger.info(f"Reasoner: {dict(thought=thought, route=route)}")
        print(f"[Brain] Inner Monologue: {thought}")
        print(f"[Brain] Routing: {route.upper()} | Refined Query: {refined_query}")
            
        vector_embedding = None
        if self.encoder:
            try:
                vector_embedding = self.encoder.encode(refined_query)
                # Ensure it's numpy
                if not isinstance(vector_embedding, np.ndarray):
                    vector_embedding = np.array(vector_embedding)
            except Exception as e:
                logger.error(f"Embedding generation failed: {e}")
        
        if vector_embedding is None:
             vector_embedding = np.zeros(EMBEDDING_DIM)

        # Pass route strategy to retrieve_context
        raw_docs = self.retrieve_context(refined_query, vector_embedding, strategy=route)
        final_context = self.compress_context(raw_docs, refined_query)
        
        self.stm_queue.put({
            "type": "interaction",
            "query": query,
            "refined_query": refined_query,
            "thought": thought,
            "context_used": final_context,
            "timestamp": time.time()
        })
        
        return final_context

