"""
Migration Utility: Rebuild vectors dictionary from FAISS index.

This script fixes the UUID map / vectors desynchronization by reconstructing
the vectors from the existing FAISS index.
"""

import numpy as np
import pickle
from pathlib import Path
from config import DATA_DIR, EMBEDDING_DIM
from modules.memory_store import SharedMemoryManager

def migrate_vectors():
    print("=" * 60)
    print("Vector Dictionary Migration Utility")
    print("=" * 60)
    
    # Initialize memory manager (will trigger the warning)
    print("\n[1] Loading existing memory state...")
    memory = SharedMemoryManager(embedding_dim=EMBEDDING_DIM)
    
    print(f"    UUID map entries: {len(memory.uuid_to_id)}")
    print(f"    Vectors dict entries: {len(memory.vectors)}")
    print(f"    FAISS index entries: {memory.index.ntotal}")
    
    if len(memory.uuid_to_id) == len(memory.vectors):
        print("\n✓ Already synchronized! No migration needed.")
        return
    
    print(f"\n[2] Rebuilding vectors dictionary from FAISS...")
    
    # FAISS IndexIDMap doesn't support reconstruction by default
    # We need to search for each ID to get its vector
    reconstructed_count = 0
    failed_count = 0
    
    for uuid, internal_id in list(memory.uuid_to_id.items()):
        if uuid in memory.vectors:
            # Already has vector
            continue
        
        try:
            # For IndexIDMap wrapping IndexFlatL2, we need to extract the vector
            # This is a workaround - we'll search with a dummy query and find the exact ID
            # Better approach: reconstruct from the underlying index
            
            # Access the underlying index
            if hasattr(memory.index, 'index'):
                # IndexIDMap wrapper
                underlying_index = memory.index.index
                if hasattr(underlying_index, 'reconstruct'):
                    # Find the sequential position of this ID
                    # For IndexIDMap, the ID map is stored separately
                    # We need to find which position this internal_id is at
                    
                    # Search all IDs to find position
                    for pos in range(memory.index.ntotal):
                        try:
                            vec = underlying_index.reconstruct(int(pos))
                            # Verify this is the right vector by searching
                            distances, ids = memory.index.search(
                                vec.reshape(1, -1).astype('float32'), k=1
                            )
                            if ids[0][0] == internal_id:
                                memory.vectors[uuid] = vec
                                reconstructed_count += 1
                                break
                        except Exception:
                            continue
            
            if uuid not in memory.vectors:
                failed_count += 1
                print(f"    ⚠ Could not reconstruct vector for {uuid[:8]}...")
                
        except Exception as e:
            failed_count += 1
            print(f"    ✗ Failed to reconstruct {uuid[:8]}: {e}")
    
    print(f"\n[3] Migration Results:")
    print(f"    Reconstructed: {reconstructed_count}")
    print(f"    Failed: {failed_count}")
    print(f"    Final vectors dict size: {len(memory.vectors)}")
    
    if failed_count > 0:
        print("\n⚠ WARNING: Some vectors could not be reconstructed.")
        print("   Recommendation: Clean the data directory and re-ingest documents.")
        response = input("\n   Delete failed nodes from graph? (y/n): ")
        if response.lower() == 'y':
            # Remove nodes that don't have vectors
            nodes_to_remove = [
                uuid for uuid in memory.uuid_to_id.keys() 
                if uuid not in memory.vectors
            ]
            for uuid in nodes_to_remove:
                memory.remove_node(uuid)
            print(f"   Removed {len(nodes_to_remove)} nodes without vectors.")
    
    print(f"\n[4] Saving synchronized state...")
    memory.save_snapshot()
    
    print("\n" + "=" * 60)
    print("Migration Complete!")
    print("=" * 60)
    print("\nYou can now restart main_controller.py")

if __name__ == "__main__":
    migrate_vectors()
