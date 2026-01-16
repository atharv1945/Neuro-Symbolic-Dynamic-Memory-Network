import time
import queue
import threading
import uuid as uuid_lib
import random
import numpy as np
import networkx as nx

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from config import (
    ENTROPY_DECAY_RATE,
    PRUNING_THRESHOLD,
    SAVE_INTERVAL_SECONDS,
    DREAMER_BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
    SIMILARITY_THRESHOLD
)
from modules.utils import logger
from modules.memory_store import SharedMemoryManager


# --- REM Cycle Configuration ---
REM_LATENT_SIMILARITY_THRESHOLD = 0.85  # Cosine similarity threshold for latent links
REM_MAX_PAIRS_PER_CYCLE = 15            # Max latent pairs to process per cycle (10-20 range)
REM_MIN_CLIQUE_SIZE = 3                 # Minimum clique size for abstraction
REM_CHILD_ENERGY_DECAY = 0.7            # Multiplier to reduce child node energy (soft prune)


class MemoryDreamer(threading.Thread):
    """
    Memory Consolidation Thread with Active REM Cycle.
    
    Responsibilities:
    1. Entropy Decay: Normal memory maintenance and pruning
    2. Latent Association: Discover implicit connections during idle time
    3. Concept Abstraction: Create super-nodes from highly connected clusters
    """
    
    def __init__(self, memory_manager: SharedMemoryManager, stm_queue: queue.Queue):
        super().__init__(name="DreamerThread", daemon=True)
        self.memory = memory_manager
        self.stm_queue = stm_queue
        self.stop_event = threading.Event()
        self.last_save_time = time.time()
        self.last_rem_time = time.time()
        
        if SentenceTransformer:
            # Use config value for consistency with Brain
            self.encoder = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu') 
        else:
            self.encoder = None
            logger.warning("SentenceTransformer not found. Dreamer cannot vectorize new memories.")

    def run(self):
        logger.info("Dreamer Thread Started.")
        processed_count = 0
        
        while not self.stop_event.is_set():
            try:
                item = self.stm_queue.get(timeout=1.0) 
                
                if item == "POISON_PILL":
                    logger.info("Poison Pill received. Saving state and exiting.")
                    self.memory.save_snapshot()
                    break
                
                self.consolidate_memory(item)
                processed_count += 1
                
                if processed_count >= DREAMER_BATCH_SIZE:
                    self.run_maintenance()
                    processed_count = 0

            except queue.Empty:
                # --- IDLE TIME: Run REM Cycle ---
                self.run_rem_cycle()
                
                if time.time() - self.last_save_time > SAVE_INTERVAL_SECONDS:
                    logger.info("Auto-Saving Memory Snapshot...")
                    self.memory.save_snapshot()
                    self.last_save_time = time.time()
                continue
            except Exception as e:
                logger.error(f"Dreamer Error: {e}")
        
        logger.info("Dreamer Thread Exited.")

    def consolidate_memory(self, item: dict):
        """Convert STM item to long-term memory with vector embedding."""
        if not self.encoder:
            return
            
        try:
            text = f"Query: {item.get('query')} | Context: {item.get('context_used')}"
            mem_uuid = str(uuid_lib.uuid4())
            vector = self.encoder.encode(text)
            
            # memory_store.add_memory will handle numpy conversion if needed
            self.memory.add_memory(mem_uuid, text, vector)
            
        except Exception as e:
            logger.error(f"Consolidation Failed: {e}")

    def run_maintenance(self):
        """Standard entropy decay and pruning pass."""
        nodes_to_check = []
        with self.memory.lock:
            nodes_to_check = list(self.memory.graph.nodes())
            
        if not nodes_to_check:
            return

        nodes_to_prune = []
        sample_size = min(len(nodes_to_check), 50)
        sample_nodes = random.sample(nodes_to_check, sample_size)
        
        current_time = time.time()
        
        for node_uuid in sample_nodes:
            # READ
            with self.memory.lock:
                if not self.memory.graph.has_node(node_uuid): 
                    continue
                attributes = self.memory.graph.nodes[node_uuid]
                energy = attributes.get('energy', 10.0)
                last_access = attributes.get('last_access', current_time)
            
            # LOGIC
            hours_elapsed = (current_time - last_access) / 3600.0
            decay = ENTROPY_DECAY_RATE * hours_elapsed
            new_energy = energy - decay
            
            # WRITE Update or Prune
            with self.memory.lock:
                if self.memory.graph.has_node(node_uuid):
                    if new_energy <= PRUNING_THRESHOLD:
                         nodes_to_prune.append(node_uuid)
                    else:
                        self.memory.graph.nodes[node_uuid]['energy'] = new_energy
        
        # Prune
        for node_uuid in nodes_to_prune:
            self.memory.remove_node(node_uuid)
            logger.info(f"Pruned Node {node_uuid} due to low energy.")

    # =========================================================================
    # REM CYCLE: Active Memory Consolidation
    # =========================================================================
    
    def run_rem_cycle(self):
        """
        Execute the REM (Rapid Eye Movement) consolidation cycle.
        
        This runs during idle periods when the STM queue is empty.
        Performs two key operations:
        1. Latent Association Discovery ("Dreaming")
        2. Concept Abstraction ("Consolidation")
        """
        try:
            # Feature 1: Discover latent associations
            self._discover_latent_associations()
            
            # Feature 2: Abstract dense clusters into concepts
            self._abstract_concepts()
            
        except Exception as e:
            logger.error(f"[REM] Cycle Error: {e}")

    def _discover_latent_associations(self):
        """
        Feature 1: Latent Association (The "Dreaming" Step)
        
        Scans FAISS index for vector pairs with high similarity (>0.85)
        that are NOT connected in the NetworkX graph, then creates
        latent association edges between them.
        
        Constraint: Limited to REM_MAX_PAIRS_PER_CYCLE pairs to prevent CPU spikes.
        """
        # Get all node UUIDs and their internal IDs
        with self.memory.lock:
            if self.memory.index.ntotal < 2:
                return  # Need at least 2 vectors
            
            all_uuids = list(self.memory.uuid_to_id.keys())
            uuid_to_id = dict(self.memory.uuid_to_id)
            id_to_uuid = dict(self.memory.id_to_uuid)
        
        if len(all_uuids) < 2:
            return
        
        # Sample nodes to check (avoid processing entire index)
        sample_size = min(len(all_uuids), 30)
        sample_uuids = random.sample(all_uuids, sample_size)
        
        discovered_pairs = []
        
        for source_uuid in sample_uuids:
            if len(discovered_pairs) >= REM_MAX_PAIRS_PER_CYCLE:
                break
                
            # Get source vector from FAISS
            with self.memory.lock:
                if source_uuid not in self.memory.uuid_to_id:
                    continue
                source_id = self.memory.uuid_to_id[source_uuid]
                
                # Reconstruct vector from FAISS index
                try:
                    source_vector = self._reconstruct_vector(source_id)
                    if source_vector is None:
                        continue
                except Exception:
                    continue
            
            # Find similar vectors
            with self.memory.lock:
                if self.memory.index.ntotal == 0:
                    continue
                    
                # Search for top-k similar (k=10 to find candidates)
                distances, ids = self.memory.index.search(
                    source_vector.reshape(1, -1).astype('float32'), 
                    min(10, self.memory.index.ntotal)
                )
            
            # Convert L2 distances to cosine similarity (approximate for normalized vectors)
            # For normalized vectors: cosine_sim ≈ 1 - (L2_dist^2 / 2)
            for dist, target_id in zip(distances[0], ids[0]):
                if len(discovered_pairs) >= REM_MAX_PAIRS_PER_CYCLE:
                    break
                    
                if target_id == -1 or target_id == source_id:
                    continue
                    
                # Convert L2 distance to similarity
                # L2 distance for normalized vectors: d = sqrt(2 - 2*cos(θ))
                # So: cos(θ) = 1 - d²/2
                similarity = 1.0 - (dist / 2.0) if dist < 2.0 else 0.0
                
                if similarity < REM_LATENT_SIMILARITY_THRESHOLD:
                    continue
                
                # Get target UUID
                with self.memory.lock:
                    if target_id not in self.memory.id_to_uuid:
                        continue
                    target_uuid = self.memory.id_to_uuid[target_id]
                    
                    # Check if edge already exists
                    if self.memory.graph.has_edge(source_uuid, target_uuid):
                        continue
                    
                    # Check both nodes still exist
                    if not (self.memory.graph.has_node(source_uuid) and 
                            self.memory.graph.has_node(target_uuid)):
                        continue
                
                # Found a latent pair!
                discovered_pairs.append((source_uuid, target_uuid, similarity))
        
        # Create latent association edges
        for source_uuid, target_uuid, similarity in discovered_pairs:
            with self.memory.lock:
                # Double-check edge doesn't exist (thread safety)
                if not self.memory.graph.has_edge(source_uuid, target_uuid):
                    if (self.memory.graph.has_node(source_uuid) and 
                        self.memory.graph.has_node(target_uuid)):
                        self.memory.graph.add_edge(
                            source_uuid, 
                            target_uuid,
                            relation="latent_association",
                            weight=0.5,
                            discovered_at=time.time(),
                            similarity=similarity
                        )
                        logger.info(
                            f"[REM] Discovered latent link: {source_uuid[:8]}... <-> "
                            f"{target_uuid[:8]}... (similarity: {similarity:.3f})"
                        )

    def _reconstruct_vector(self, internal_id: int) -> np.ndarray:
        """
        Reconstruct a vector from the FAISS index by its internal ID.
        
        Note: This requires the index to support reconstruction.
        For IndexIDMap wrapping IndexFlatL2, we need to search by the ID.
        """
        # For IndexIDMap, we can't directly reconstruct by external ID
        # Instead, we'll search all vectors and find the one with matching ID
        # This is a workaround - in production, you might want to store vectors separately
        
        try:
            # Create a query that will return this specific vector
            # We can use the id_to_uuid mapping and search
            if hasattr(self.memory.index, 'reconstruct'):
                return self.memory.index.reconstruct(internal_id)
            else:
                # Fallback: Cannot reconstruct, skip this node
                return None
        except Exception:
            return None

    def _abstract_concepts(self):
        """
        Feature 2: Concept Abstraction (The "Consolidation" Step)
        
        Detects cliques (fully connected subgraphs of size 3+) in the graph
        and abstracts them into "Super Nodes" representing concepts.
        
        Actions:
        1. Create a new "Super Node" (ID: concept_{uuid})
        2. Aggregate text of child nodes
        3. Link Super Node to neighbors of child nodes
        4. Soft Prune: Lower energy of child nodes significantly
        """
        # Find cliques (fully connected subgraphs)
        with self.memory.lock:
            if self.memory.graph.number_of_nodes() < REM_MIN_CLIQUE_SIZE:
                return
            
            # Find all cliques of minimum size
            try:
                all_cliques = list(nx.find_cliques(self.memory.graph))
            except Exception as e:
                logger.debug(f"[REM] Could not find cliques: {e}")
                return
        
        # Filter to cliques of required size
        valid_cliques = [c for c in all_cliques if len(c) >= REM_MIN_CLIQUE_SIZE]
        
        if not valid_cliques:
            return
        
        # Process only one clique per cycle to prevent CPU spikes
        # Sort by size (prefer larger cliques) and take the largest
        valid_cliques.sort(key=len, reverse=True)
        clique = valid_cliques[0]
        
        # Skip if any node in clique is already a concept node
        with self.memory.lock:
            for node in clique:
                if str(node).startswith("concept_"):
                    return
                if not self.memory.graph.has_node(node):
                    return
        
        # Create the Super Node
        self._create_super_node(clique)

    def _create_super_node(self, clique_nodes: list):
        """
        Create a super node from a clique of child nodes.
        
        Args:
            clique_nodes: List of node UUIDs in the clique
        """
        concept_uuid = f"concept_{uuid_lib.uuid4()}"
        
        with self.memory.lock:
            # Verify all nodes still exist
            for node in clique_nodes:
                if not self.memory.graph.has_node(node):
                    return
            
            # Step 1: Gather child node texts
            child_texts = []
            child_labels = []
            for node in clique_nodes:
                attrs = self.memory.graph.nodes[node]
                text = attrs.get('text', '')
                if text:
                    child_texts.append(text)
                    child_labels.append(node[:8])  # Short label
            
            if not child_texts:
                return
            
            # Step 2: Aggregate text for super node
            aggregated_text = f"Summary of [{', '.join(child_labels)}...]: " + " | ".join(
                t[:100] if len(t) > 100 else t for t in child_texts[:5]
            )
            
            # Step 3: Compute concept vector (average of child vectors)
            child_vectors = []
            for node in clique_nodes:
                if node in self.memory.uuid_to_id:
                    internal_id = self.memory.uuid_to_id[node]
                    vec = self._reconstruct_vector(internal_id)
                    if vec is not None:
                        child_vectors.append(vec)
            
            if child_vectors:
                concept_vector = np.mean(child_vectors, axis=0)
            else:
                # Cannot create concept without vectors
                return
            
            # Step 4: Find all neighbors of child nodes (excluding the clique itself)
            external_neighbors = set()
            for node in clique_nodes:
                for neighbor in self.memory.graph.neighbors(node):
                    if neighbor not in clique_nodes:
                        external_neighbors.add(neighbor)
            
            # Step 5: Add the Super Node
            try:
                # Add to graph only (not FAISS yet, to avoid issues)
                self.memory.graph.add_node(
                    concept_uuid,
                    text=aggregated_text,
                    energy=10.0,
                    last_access=time.time(),
                    is_concept=True,
                    child_nodes=list(clique_nodes)
                )
                
                # Step 6: Link Super Node to external neighbors
                for neighbor in external_neighbors:
                    if self.memory.graph.has_node(neighbor):
                        self.memory.graph.add_edge(
                            concept_uuid,
                            neighbor,
                            relation="concept_link",
                            weight=0.7
                        )
                
                # Step 7: Link Super Node to its children
                for child in clique_nodes:
                    if self.memory.graph.has_node(child):
                        self.memory.graph.add_edge(
                            concept_uuid,
                            child,
                            relation="abstraction",
                            weight=1.0
                        )
                
                # Step 8: Soft Prune - Reduce energy of child nodes
                for child in clique_nodes:
                    if self.memory.graph.has_node(child):
                        current_energy = self.memory.graph.nodes[child].get('energy', 10.0)
                        new_energy = current_energy * REM_CHILD_ENERGY_DECAY
                        self.memory.graph.nodes[child]['energy'] = new_energy
                
                logger.info(
                    f"[REM] Abstracted cluster into concept: {concept_uuid[:20]}... "
                    f"(consolidated {len(clique_nodes)} nodes)"
                )
                
            except Exception as e:
                logger.error(f"[REM] Failed to create super node: {e}")
                # Cleanup: remove concept node if it was partially added
                if self.memory.graph.has_node(concept_uuid):
                    self.memory.graph.remove_node(concept_uuid)

    def stop(self):
        """Signal the dreamer thread to stop."""
        self.stop_event.set()
