import logging
import queue
import time
from typing import List, Tuple, Dict, Optional
import datetime
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

    def retrieve_context(self, query: str, vector_embedding: np.ndarray, strategy: str = 'vector', filters: Optional[Dict] = None) -> List[str]:
        context_docs = []
        
        # 1. Vector Search with Over-Fetching (Always done to find entry points)
        # Increase top_k to 50 to ensure we have enough candidates after filtering
        top_k = 50 if filters else 5
        hits = self.memory.query_similarity(vector_embedding, top_k=top_k)
        
        # 2. Apply Metadata Filtering (if filters are provided)
        if filters:
            filtered_hits = self._apply_metadata_filter(hits, filters)
            logger.info(f"[Brain] Filtering: {len(hits)} candidates -> {len(filtered_hits)} results (filters: {filters})")
            hits = filtered_hits[:5]  # Take top 5 filtered results
        
        # Collect distinct UUIDs from hits
        hit_uuids = [uuid for uuid, score in hits]
        
        # Basic content from hits
        for uuid in hit_uuids:
             content = self.memory.get_node_content(uuid)
             if content:
                 context_docs.append(f"[Vector Match]: {content}")
                 
        # 3. Graph Traversal
        if strategy in ['graph', 'hybrid']:
            graph_context = self.retrieve_graph_context(hit_uuids)
            context_docs.extend(graph_context)
                
        return list(set(context_docs))
    
    def _apply_metadata_filter(self, hits: List[Tuple[str, float]], filters: Dict) -> List[Tuple[str, float]]:
        """
        Filters hits based on metadata constraints.
        Uses fuzzy matching for filenames (case-insensitive).
        """
        if not filters:
            return hits
        
        filtered = []
        for uuid, score in hits:
            with self.memory.lock:
                if not self.memory.graph.has_node(uuid):
                    continue
                
                metadata = self.memory.graph.nodes.get(uuid, {})
            
            # Filter by source file (fuzzy match)
            if filters.get('source_file'):
                file_name = metadata.get('file_name', '')
                if not self._fuzzy_match_filename(file_name, filters['source_file']):
                    continue
            
            # Filter by date range
            if filters.get('date_range'):
                ingest_date = metadata.get('ingest_date', 0)
                if not self._match_date_range(ingest_date, filters['date_range']):
                    continue
            
            filtered.append((uuid, score))
        
        return filtered
    
    def _fuzzy_match_filename(self, stored_name: str, query_name: str) -> bool:
        """Case-insensitive fuzzy filename matching using rapidfuzz."""
        if not stored_name or not query_name:
            return False
        # Already imported: from rapidfuzz import fuzz
        score = fuzz.ratio(stored_name.lower(), query_name.lower())
        return score > 70  # 70% similarity threshold
    
    def _match_date_range(self, timestamp: float, date_range: str) -> bool:
        """Checks if timestamp falls within date range (year-based)."""
        if not timestamp or not date_range:
            return False
        try:
            date = datetime.datetime.fromtimestamp(timestamp)
            year_str = str(date.year)
            # Simple contains check (supports '2025', '2024-01', etc.)
            return date_range in year_str or year_str in date_range
        except (ValueError, OSError):
            return False


    def retrieve_graph_context(self, entry_uuids: List[str], max_hops: int = 2) -> List[str]:
        """
        Retrieves multi-hop neighbors with PRIORITIZED CONTEXT SELECTION.
        
        Implements bucketing to ensure entity definitions (primary) are preserved
        even when total results exceed compression capacity.
        
        IMPORTANT: Works with UNDIRECTED graphs (nx.Graph). Uses neighbors() and 
        filters by edge attributes and node types.
        
        Args:
            entry_uuids: Starting nodes (typically from vector search)
            max_hops: Maximum traversal depth (default: 2)
            
        Returns:
            List of formatted context strings (capped at 15 for quality)
        """
        # === PRIORITIZED CONTEXT BUCKETS ===
        primary_context = []    # Direct entity mentions (highest priority)
        secondary_context = []  # Graph hops (lower priority)
        primary_chunk_ids = set()  # Track chunks in primary to avoid duplicates
        
        visited = set(entry_uuids)
        
        # === TASK 2: RETRIEVAL DIAGNOSTICS ===
        logger.info(f"[Brain] Graph Entry Points: {entry_uuids} (total: {len(entry_uuids)})")
        
        # BFS queue: (node_id, current_hop_depth)
        from collections import deque
        queue = deque([(uuid, 0) for uuid in entry_uuids])
        
        with self.memory.lock:
            while queue:
                current_uuid, hop_depth = queue.popleft()
                
                # Don't traverse beyond max_hops
                if hop_depth >= max_hops:
                    continue
                
                if not self.memory.graph.has_node(current_uuid):
                    continue
                
                # Get node metadata
                node_data = self.memory.graph.nodes[current_uuid]
                node_type = node_data.get('type', 'unknown')
                
                # === TASK 1: ENTITY EXPANSION (PRIMARY CONTEXT) ===
                # If this is an entity node, find source chunks that mention it
                # In an undirected graph, we check neighbors and filter by:
                # 1. Neighbor type = 'chunk'
                # 2. Edge relation = 'mentions'
                if node_type == 'entity':
                    for neighbor_id in self.memory.graph.neighbors(current_uuid):
                        if neighbor_id not in visited:
                            # Check if this neighbor is a chunk
                            neighbor_data = self.memory.graph.nodes.get(neighbor_id, {})
                            neighbor_type = neighbor_data.get('type', 'unknown')
                            
                            if neighbor_type == 'chunk':
                                # Get edge data (works in both directions for undirected graph)
                                edge_data = self.memory.graph.get_edge_data(neighbor_id, current_uuid)
                                
                                # Check if this is a 'mentions' edge
                                if edge_data and edge_data.get('relation') == 'mentions':
                                    # Get the source chunk text
                                    chunk_content = self.memory.get_node_content(neighbor_id)
                                    if chunk_content:
                                        entity_text = node_data.get('text', current_uuid)
                                        logger.info(f"[Brain] Entity Expansion: {entity_text} <- Chunk {neighbor_id[:8]}...")
                                        
                                        # Add to PRIMARY context (high priority)
                                        primary_context.append(
                                            f"[Entity Expansion | {entity_text}]: {chunk_content[:500]}"
                                        )
                                        primary_chunk_ids.add(neighbor_id)
                                        visited.add(neighbor_id)
                
                # === TASK 2: MULTI-HOP TRAVERSAL (SECONDARY CONTEXT) ===
                # Get all neighbors for standard graph traversal
                neighbors = list(self.memory.graph.neighbors(current_uuid))
                
                for neighbor_id in neighbors:
                    if neighbor_id not in visited:
                        # Get edge data (relation)
                        try:
                            edge_data = self.memory.graph.get_edge_data(current_uuid, neighbor_id)
                            relation = edge_data.get('relation', 'related_to') if edge_data else 'related_to'
                        except (KeyError, AttributeError):
                            relation = 'related_to'
                        
                        # Get neighbor content
                        neighbor_content = self.memory.get_node_content(neighbor_id)
                        
                        if neighbor_content:
                            # Tag with hop depth for transparency
                            hop_label = f"Hop {hop_depth + 1}" if hop_depth > 0 else "Direct Link"
                            logger.info(f"[Brain] Traversing ({hop_label}): {current_uuid[:8]}... --{relation}--> {neighbor_id[:8]}...")
                            
                            # Add to SECONDARY context (lower priority)
                            # Skip if already in primary context (deduplication)
                            if neighbor_id not in primary_chunk_ids:
                                secondary_context.append(
                                    f"[Graph {hop_label}: {relation}] {neighbor_content[:300]}"
                                )
                        
                        # Add to queue for next hop
                        visited.add(neighbor_id)
                        queue.append((neighbor_id, hop_depth + 1))
        
        # === PRIORITIZED MERGE & CAP ===
        # Primary context first, then secondary (space permitting)
        all_docs = primary_context + secondary_context
        total_retrieved = len(all_docs)
        
        # Hard cap at 15 documents to ensure quality over quantity
        MAX_CONTEXT_DOCS = 15
        final_docs = all_docs[:MAX_CONTEXT_DOCS]
        
        dropped_count = max(0, total_retrieved - MAX_CONTEXT_DOCS)
        
        logger.info(
            f"[Brain] Graph Retrieval Complete: {len(final_docs)} context docs "
            f"(Primary: {len(primary_context)}, Secondary: {len(secondary_context)}, "
            f"Dropped: {dropped_count}, Nodes Visited: {len(visited)})"
        )
        
        if dropped_count > 0:
            logger.info(
                f"[Brain] Prioritization: Kept {len(final_docs)} docs (Dropped {dropped_count} lower-priority docs)"
            )
        
        return final_docs


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
            
        # 1. Reasoner Step (Now returns 4 values including filters)
        refined_query, route, thought, filters = self.router.analyze_query(query)
        logger.info(f"Reasoner: {dict(thought=thought, route=route, filters=filters)}")
        print(f"[Brain] Inner Monologue: {thought}")
        print(f"[Brain] Routing: {route.upper()} | Refined Query: {refined_query}")
        if filters:
            print(f"[Brain] Filters Applied: {filters}")
            
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

        # Pass route strategy and filters to retrieve_context
        raw_docs = self.retrieve_context(refined_query, vector_embedding, strategy=route, filters=filters)
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

