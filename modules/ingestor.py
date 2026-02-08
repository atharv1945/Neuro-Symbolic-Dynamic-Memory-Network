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
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

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

    def extract_text(self, file_path: str) -> Tuple[str, Dict[int, int]]:
        """Extracts text from a PDF file using PyMuPDF (fitz) or pypdf fallback.
        
        Returns:
            Tuple of (full_text, page_map) where page_map maps character offsets to page numbers.
            Example: {0: 1, 1024: 2} means chars 0-1023 are page 1, 1024+ are page 2.
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return "", {0: 1}
            
        if not fitz:
            # Try pypdf fallback for PDF files
            if PdfReader and file_path.lower().endswith('.pdf'):
                try:
                    logger.info(f"Using PyPDF fallback for {file_path}")
                    reader = PdfReader(file_path)
                    text = ""
                    page_map = {}
                    
                    for page_num, page in enumerate(reader.pages, start=1):
                        page_map[len(text)] = page_num
                        text += page.extract_text() + "\n"
                    
                    return text, page_map
                except Exception as e:
                    logger.error(f"PyPDF extraction failed: {e}")
            
            # Final fallback to text read (no page info for non-PDF)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read(), {0: 1}
            except:
                return "", {0: 1}

        try:
            doc = fitz.open(file_path)
            text = ""
            page_map = {}  # Maps character offset to page number
            
            for page_num, page in enumerate(doc, start=1):
                page_map[len(text)] = page_num  # Record start offset of this page
                text += page.get_text() + "\n"
            
            return text, page_map
        except Exception as e:
            logger.error(f"PDF Extraction failed for {file_path}: {e}")
            
            # Try pypdf fallback if fitz fails
            if PdfReader and file_path.lower().endswith('.pdf'):
                try:
                    logger.info(f"Retrying with PyPDF fallback after fitz error")
                    reader = PdfReader(file_path)
                    text = ""
                    page_map = {}
                    
                    for page_num, page in enumerate(reader.pages, start=1):
                        page_map[len(text)] = page_num
                        text += page.extract_text() + "\n"
                    
                    return text, page_map
                except Exception as e2:
                    logger.error(f"PyPDF fallback also failed: {e2}")
            
            return "", {0: 1}

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

    def _get_page_for_offset(self, page_map: Dict[int, int], char_offset: int) -> int:
        """Helper to find page number for a given character offset.
        
        Args:
            page_map: Dict mapping character offsets to page numbers
            char_offset: Character position in the document
            
        Returns:
            Page number for the given offset
        """
        if not page_map:
            return 1
        
        # Find the largest offset that's <= char_offset
        page_num = 1
        for offset in sorted(page_map.keys()):
            if offset <= char_offset:
                page_num = page_map[offset]
            else:
                break
        
        return page_num
    
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
        
        # Extract text and page mapping
        text, page_map = self.extract_text(file_path)
        if not text:
            logger.warning("No text extracted. Aborting.")
            return
        
        # Extract file metadata
        file_name = os.path.basename(file_path)
        ingest_date = time.time()  # Current timestamp

        # Simple Chunking
        CHUNK_SIZE = 1000
        chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
        
        logger.info(f"Split into {len(chunks)} chunks. Processing...")

        total_entities = 0
        
        for idx, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            
            # Calculate page number for this chunk
            char_offset = idx * CHUNK_SIZE
            page_number = self._get_page_for_offset(page_map, char_offset)
            
            # 1. Embed Chunk
            vector = np.zeros(EMBEDDING_DIM)
            if self.encoder:
                try:
                    vector = self.encoder.encode(chunk)
                except Exception as e:
                    logger.error(f"Embedding failed: {e}")
            
            # 2. Add Chunk Node with enhanced metadata
            success = self.memory.add_memory(
                uuid=chunk_id,
                text=chunk,
                vector=vector,
                metadata={
                    "source": file_path,
                    "file_name": file_name,
                    "ingest_date": ingest_date,
                    "page_number": page_number,
                    "type": "chunk",
                    "chunk_idx": idx
                }
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
