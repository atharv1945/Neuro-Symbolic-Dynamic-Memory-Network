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
    SIMILARITY_THRESHOLD,
    REM_MIN_CLIQUE_SIZE,
    REM_MOMENTUM_THRESHOLD
)
from modules.utils import logger
from modules.memory_store import SharedMemoryManager


# --- REM Cycle Configuration ---
REM_MAX_PAIRS_PER_CYCLE = 20            # Max latent pairs to process per cycle
REM_CHILD_ENERGY_DECAY = 0.7            # Multiplier to reduce child node energy (soft prune)
REM_CYCLE_TIMEOUT_SECONDS = 2.0         # Safety backoff timeout


class MemoryDreamer(threading.Thread):
    """
    Memory Consolidation Thread with Holographic REM Cycle.
    
    Responsibilities:
    1. Entropy Decay: Normal memory maintenance and pruning
    2. Latent Association: Discover implicit connections using persistent vectors
    3. Concept Abstraction: Create searchable super-nodes from clusters (Centroid vectors)
    """
    
    def __init__(self, memory_manager: SharedMemoryManager, stm_queue: queue.Queue):
        super().__init__(name="DreamerThread", daemon=True)
        self.memory = memory_manager
        self.stm_queue = stm_queue
        self.stop_event = threading.Event()
        self.last_save_time = time.time()
        self.last_rem_time = 0  # Track last REM cycle time for frequency control
        
        # Gate threshold from config
        self.similarity_gate = SIMILARITY_THRESHOLD / 100.0
        
        if SentenceTransformer:
            self.encoder = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu') 
        else:
            self.encoder = None
            logger.warning("SentenceTransformer not found. Dreamer cannot vectorize new memories.")

    def run(self):
        logger.info("Dreamer Thread Started.")
        processed_count = 0
        
        while not self.stop_event.is_set():
            try:
                # Check for STM items
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
                    # --- IDLE TIME: Run REM Cycle with Frequency Control ---
                    current_time = time.time()
                    REM_CYCLE_INTERVAL = 300  # 5 minutes between REM cycles
                    
                    if current_time - self.last_rem_time >= REM_CYCLE_INTERVAL:
                        logger.info("[REM] Idle detected. Initiating REM Consolidation Cycle...")
                        self.run_rem_cycle()
                        self.last_rem_time = current_time
                        logger.info("[REM] Cycle complete. Next cycle in 5 minutes.")
                    
                    # Periodic save
                    if current_time - self.last_save_time > SAVE_INTERVAL_SECONDS:
                        logger.info("Auto-Saving Memory Snapshot...")
                        self.memory.save_snapshot()
                        self.last_save_time = current_time

            except Exception as e:
                logger.error(f"Dreamer Error: {e}")
        
        logger.info("Dreamer Thread Exited.")

    def consolidate_memory(self, item: dict):
        """Convert STM item to long-term memory."""
        if not self.encoder:
            return
            
        try:
            text = f"Query: {item.get('query')} | Context: {item.get('context_used')}"
            mem_uuid = str(uuid_lib.uuid4())
            vector = self.encoder.encode(text)
            
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
        sample_nodes = random.sample(nodes_to_check, min(len(nodes_to_check), 50))
        current_time = time.time()
        
        for node_uuid in sample_nodes:
            with self.memory.lock:
                if not self.memory.graph.has_node(node_uuid): continue
                attrs = self.memory.graph.nodes[node_uuid]
                energy = attrs.get('energy', 10.0)
                last_access = attrs.get('last_access', current_time)
            
            hours_elapsed = (current_time - last_access) / 3600.0
            decay = ENTROPY_DECAY_RATE * hours_elapsed
            new_energy = energy - decay
            
            with self.memory.lock:
                if self.memory.graph.has_node(node_uuid):
                    if new_energy <= PRUNING_THRESHOLD:
                         nodes_to_prune.append(node_uuid)
                    else:
                        self.memory.graph.nodes[node_uuid]['energy'] = new_energy
        
        for node_uuid in nodes_to_prune:
            self.memory.remove_node(node_uuid)
            logger.info(f"Pruned Node {node_uuid}")

    # =========================================================================
    # REM CYCLE: Holographic Consolidation
    # =========================================================================
    
    def run_rem_cycle(self):
        """Orchestrate the REM cycle with safety yields and momentum initialization."""
        cycle_start = time.time()
        
        # === TITANS: Initialize momentum for all nodes ===
        with self.memory.lock:
            for node in self.memory.graph.nodes():
                if 'momentum' not in self.memory.graph.nodes[node]:
                    self.memory.graph.nodes[node]['momentum'] = 0
        
        # Phase 1: Latent links
        if self._safety_yield(): return
        self._discover_latent_associations(cycle_start)
        
        # Phase 2: Concept abstraction
        if self._safety_yield(): return
        self._abstract_concepts(cycle_start)

    def _safety_yield(self) -> bool:
        """Yield control back if STM queue has tasks."""
        if not self.stm_queue.empty():
            return True
        return False

    def _discover_latent_associations(self, cycle_start: float):
        """
        Scan persistent vectors to find new semantic associations.
        Excludes concept nodes to prevent recursive hallucinations.
        """
        # Sample most recent nodes for latent discovery
        with self.memory.lock:
            # Filter out concept nodes to prevent loopback hallucinations
            all_uuids = [
                uuid for uuid in self.memory.vectors.keys() 
                if not str(uuid).startswith("concept_")
            ]
        
        if len(all_uuids) < 2:
            return

        sample_uuids = random.sample(all_uuids, min(len(all_uuids), 50))
        discovered = 0
        
        for source_uuid in sample_uuids:
            # Safety checks
            if self._safety_yield(): break
            if time.time() - cycle_start > REM_CYCLE_TIMEOUT_SECONDS: break
            if discovered >= REM_MAX_PAIRS_PER_CYCLE: break
            
            # 1. Get source vector from storage (NO LOCK needed for read-only dict access if we have a local copy/ref)
            # Actually, dictionary might be mutated, but we have a copy of keys.
            # To be safe, we check if it exists in a non-blocking way.
            source_vec = self.memory.vectors.get(source_uuid)
            if source_vec is None: continue
            
            # 2. Query similarity via FAISS (FAISS search handles its own internal locks usually, 
            # but memory_store wraps it in self.lock)
            results = self.memory.query_similarity(source_vec, top_k=5)
            
            for target_uuid, dist in results:
                if target_uuid == source_uuid: continue
                
                # Convert L2 distance to similarity (FAISS IndexFlatL2 returns L2^2)
                # For normalized vectors, sim = 1 - (dist / 2)
                similarity = 1.0 - (dist / 2.0)
                
                if similarity >= self.similarity_gate:
                    # 3. Check graph and add edge (FINE-GRAINED LOCKING)
                    with self.memory.lock:
                        if not self.memory.graph.has_edge(source_uuid, target_uuid):
                            if self.memory.graph.has_node(source_uuid) and self.memory.graph.has_node(target_uuid):
                                self.memory.graph.add_edge(
                                    source_uuid, target_uuid,
                                    relation="latent_association",
                                    weight=0.5,
                                    similarity=similarity
                                )
                                
                                # === TITANS: Increment Momentum ===
                                self.memory.graph.nodes[source_uuid]['momentum'] = self.memory.graph.nodes[source_uuid].get('momentum', 0) + 1
                                self.memory.graph.nodes[target_uuid]['momentum'] = self.memory.graph.nodes[target_uuid].get('momentum', 0) + 1
                                
                                logger.info(f"[REM] Discovered latent link: {source_uuid[:8]} <-> {target_uuid[:8]} (sim: {similarity:.2f})")
                                logger.info(f"[Titans] Momentum: {source_uuid[:8]} -> {self.memory.graph.nodes[source_uuid]['momentum']}")
                                logger.info(f"[Titans] Momentum: {target_uuid[:8]} -> {self.memory.graph.nodes[target_uuid]['momentum']}")
                                discovered += 1

    def _abstract_concepts(self, cycle_start: float):
        """
        Detect dense clusters and synthesize them into Super Nodes.
        """
        if self._safety_yield(): return
        
        # Subgraph of high energy nodes (Active Memory)
        with self.memory.lock:
            high_energy_nodes = [
                n for n, d in self.memory.graph.nodes(data=True) 
                if d.get('energy', 0) > 5.0 and not str(n).startswith("concept_")
            ]
            if len(high_energy_nodes) < 3: return
            subgraph = self.memory.graph.subgraph(high_energy_nodes)
            
            # Find cliques (Computationally expensive)
            try:
                cliques = list(nx.find_cliques(subgraph))
                cliques = [c for c in cliques if len(c) >= REM_MIN_CLIQUE_SIZE]
            except Exception:
                return

        if not cliques:
            return

        # === TITANS: Sort cliques by average momentum ===
        def avg_momentum(clique):
            with self.memory.lock:
                momentums = [self.memory.graph.nodes[n].get('momentum', 0) for n in clique if self.memory.graph.has_node(n)]
            return sum(momentums) / len(momentums) if momentums else 0
        
        cliques_sorted = sorted(cliques, key=avg_momentum, reverse=True)
        
        # Process high-momentum clusters first
        for clique in cliques_sorted:
            clique_size = len(clique)
            avg_mom = avg_momentum(clique)
            
            # === TITANS: Early consolidation for high-momentum clusters ===
            if avg_mom >= REM_MOMENTUM_THRESHOLD:
                logger.info(f"[Titans] High-momentum cluster (size={clique_size}, momentum={avg_mom:.1f}) - consolidating")
                self._create_super_node(clique)
                break  # Process one per cycle
            elif clique_size >= REM_MIN_CLIQUE_SIZE:
                # Standard consolidation for large clusters
                logger.info(f"[REM] Standard cluster (size={clique_size}, momentum={avg_mom:.1f}) - consolidating")
                self._create_super_node(clique)
                break  # Process one per cycle

    def _create_super_node(self, nodes: list):
        """
        Synthesize a searchable Super Node from a clique.
        """
        concept_uuid = f"concept_{uuid_lib.uuid4()}"
        
        # 1. Fetch vectors for centroid calculation
        vectors = []
        texts = []
        for node in nodes:
            vec = self.memory.vectors.get(node)
            if vec is not None:
                vectors.append(vec)
            
            with self.memory.lock:
                if self.memory.graph.has_node(node):
                    texts.append(self.memory.graph.nodes[node].get('text', ''))

        if len(vectors) < REM_MIN_CLIQUE_SIZE:
            return

        # 2. Calculate Energy-Weighted Centroid (Prevents Semantic Drift)
        energies = []
        for node in nodes:
            with self.memory.lock:
                if self.memory.graph.has_node(node):
                    energies.append(self.memory.graph.nodes[node].get('energy', 1.0))
                else:
                    energies.append(1.0)
        
        # Normalize energy weights
        energy_sum = sum(energies)
        if energy_sum > 0:
            weights = np.array([e / energy_sum for e in energies])
            # Weighted average
            centroid = np.average(vectors, axis=0, weights=weights)
        else:
            centroid = np.mean(vectors, axis=0)
        
        summary_text = f"Abstract Concept: {', '.join(texts[:3])}..."

        # === TITANS: Calculate average momentum for concept metadata ===
        with self.memory.lock:
            avg_momentum = sum(self.memory.graph.nodes[n].get('momentum', 0) for n in nodes if self.memory.graph.has_node(n)) / len(nodes)

        # 3. Add to Main Memory (Graph + FAISS) with Titans metadata
        self.memory.add_memory(
            concept_uuid, 
            summary_text, 
            centroid, 
            metadata={
                'is_concept': True,
                'method': 'titans_momentum',
                'avg_momentum': avg_momentum
            }
        )
        
        # 4. Link children and Soft Prune
        with self.memory.lock:
            for node in nodes:
                if self.memory.graph.has_node(node):
                    # Abstraction Link
                    self.memory.graph.add_edge(concept_uuid, node, relation="abstraction", weight=1.0)
                    # Soft Pruning
                    self.memory.graph.nodes[node]['energy'] *= REM_CHILD_ENERGY_DECAY
            
            logger.info(f"[REM] Abstracted cluster into concept: {concept_uuid[:15]} (children: {len(nodes)})")

    def stop(self):
        self.stop_event.set()
