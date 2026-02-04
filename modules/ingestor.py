import os
import time
import uuid
import logging
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch # Needed for PyTorch 2.6 fix

# Performance Optimization: SLM (GLiNER) + PyMuPDF
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from gliner import GLiNER
except ImportError:
    GLiNER = None

from config import EMBEDDING_DIM
from modules.utils import logger
from modules.memory_store import SharedMemoryManager

class WindBellIngestor:
    """
    Phase A: Ingestion Pipeline (Wind-Bell) - High Performance Edition
    Uses PyMuPDF (fitz) for fast parsing and GLiNER for millisecond entity extraction.
    """
    def __init__(self, memory_manager: SharedMemoryManager, encoder=None):
        self.memory = memory_manager
        self.encoder = encoder
        
        # Check Dependencies
        if not fitz:
            logger.critical("PyMuPDF (fitz) not found. PDF ingestion will FAIL. Install with `pip install pymupdf`.")
        
        if not GLiNER:
            logger.critical("GLiNER not found. Entity extraction will FAIL. Install with `pip install gliner`.")
            self.gliner_model = None
        else:
            logger.info("Loading GLiNER Model (urchade/gliner_multi)...")
            try:
                # FIX: PyTorch 2.6+ comprehensive compatibility for legacy models
                # GLiNER uses legacy .tar format that requires deep patching
                import pickle
                import torch.serialization
                
                # Save original functions
                original_torch_load = torch.load
                original_pickle_load = pickle.load
                
                def patched_torch_load(*args, **kwargs):
                    """Patched torch.load that forces legacy compatibility"""
                    kwargs['weights_only'] = False
                    if 'map_location' not in kwargs:
                        kwargs['map_location'] = 'cpu'
                    
                    # Try to use _legacy_load if available
                    try:
                        if hasattr(torch.serialization, '_legacy_load'):
                            return torch.serialization._legacy_load(*args, **kwargs)
                    except:
                        pass
                    
                    return original_torch_load(*args, **kwargs)
                
                # Apply patches globally
                torch.load = patched_torch_load
                pickle.load = original_pickle_load  # Keep pickle unchanged for now
                
                try:
                    # Load model with patches active
                    # Using gliner_multi instead of gliner_small-v2.1 for PyTorch 2.3+ compatibility
                    self.gliner_model = GLiNER.from_pretrained("urchade/gliner_multi")
                    self.labels = ["person", "organization", "location", "technology", "concept", "metric", "method"]
                    logger.info("GLiNER Loaded Successfully.")
                finally:
                    # Restore original functions
                    torch.load = original_torch_load
                    pickle.load = original_pickle_load
            except Exception as e:
                logger.error(f"Failed to load GLiNER: {e}")
                self.gliner_model = None

    def extract_text(self, file_path: str) -> str:
        """Extracts text from a PDF file using PyMuPDF (fitz)."""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return ""
            
        if not fitz:
            # Fallback to text read
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except:
                return ""

        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"PDF Extraction failed for {file_path}: {e}")
            return ""

    def extract_entities(self, text_chunk: str) -> List[Tuple[str, str, str]]:
        """
        Uses GLiNER to extract entities and formats them as triplets.
        Format: (EntityText, "is_type", EntityLabel)
        """
        if not self.gliner_model:
            return []

        try:
            # GLiNER Inference (Blocking but Fast)
            entities = self.gliner_model.predict_entities(text_chunk, self.labels)
            
            triplets = []
            for ent in entities:
                text = ent["text"]
                label = ent["label"]
                # Convert to triplet to satisfy Graph logic
                # (Entity, Relation, Class)
                triplets.append((text, "is_type", label))
                
            return triplets
        except Exception as e:
            logger.error(f"GLiNER Extraction failed: {e}")
            return []

    def link_entities(self, triplets: List[Tuple[str, str, str]], chunk_id: Optional[str] = None) -> int:
        """
        Creates edges in the graph for the provided triplets.
        If chunk_id is provided, links the chunk node to the entity nodes.
        Returns the number of edges added.
        """
        edges_added = 0
        with self.memory.lock:
            for subj, pred, obj in triplets:
                subj_uuid = subj 
                obj_uuid = obj 
                
                # Ensure nodes exist
                if not self.memory.graph.has_node(subj_uuid):
                    self.memory.graph.add_node(subj_uuid, text=subj, type='entity', energy=5.0)
                
                if not self.memory.graph.has_node(obj_uuid):
                    self.memory.graph.add_node(obj_uuid, text=obj, type='concept', energy=5.0)
                
                # Add Edge between Entity and its Class (e.g., "RTX 4090" --is_type--> "Technology")
                self.memory.graph.add_edge(subj_uuid, obj_uuid, relation=pred)
                edges_added += 1
                
                # Link Chunk -> Entities (The Subject of the triplet is the Entity found in text)
                if chunk_id and self.memory.graph.has_node(chunk_id):
                    # We link Chunk -> Entity (subj)
                    # Relation: "mentions"
                    self.memory.graph.add_edge(chunk_id, subj_uuid, relation="mentions")
                    
        return edges_added

    def ingest_document(self, file_path: str) -> None:
        """
        Main orchestrator: Read -> Chunk -> Extract -> Embed -> Save -> Link.
        """
        logger.info(f"Starting ingestion for: {file_path}")
        start_time = time.time()
        
        text = self.extract_text(file_path)
        if not text:
            logger.warning("No text extracted. Aborting.")
            return

        # Simple Chunking
        CHUNK_SIZE = 1000
        chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
        
        logger.info(f"Split into {len(chunks)} chunks. Processing...")

        total_entities = 0
        
        for idx, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            
            # 1. Embed Chunk
            vector = np.zeros(EMBEDDING_DIM)
            if self.encoder:
                try:
                    vector = self.encoder.encode(chunk)
                except Exception as e:
                    logger.error(f"Embedding failed: {e}")
            
            # 2. Add Chunk Node
            success = self.memory.add_memory(
                uuid=chunk_id,
                text=chunk,
                vector=vector,
                metadata={"source": file_path, "type": "chunk", "chunk_idx": idx}
            )
            
            if success:
                # 3. Extract Entities (SLM Step - GLiNER)
                # This should now be very fast
                triplets = self.extract_entities(chunk)
                
                if triplets:
                    # logger.info(f"Chunk {idx}: Found {len(triplets)} entities.")
                    self.link_entities(triplets, chunk_id=chunk_id)
                    total_entities += len(triplets)
            
        duration = time.time() - start_time
        logger.info(f"Ingestion complete for {file_path}. Found {total_entities} entities in {duration:.2f}s")
        
        # === TASK 1: ATOMIC INGESTION PERSISTENCE ===
        # Trigger immediate save to prevent data loss on crash/restart
        logger.info("[Ingestor] Triggering auto-save for persistence...")
        self.memory.save_snapshot()
        logger.info("[Ingestor] Auto-save complete. Data is now persistent.")
