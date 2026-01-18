# Data Migration Guide

## Problem: UUID Map / Vectors Desynchronization

The atomic persistence validation detected that your `uuid_to_id` map and `vectors` dictionary are out of sync.

**Root Cause**: The holographic REM consolidation upgrade added a new `vectors.pkl` file for persistent vector storage. Your existing `data/` directory was created before this change.

## Solutions

### Option 1: Clean Start (Recommended for Development)

Delete the existing data directory and re-ingest your documents:

```bash
# Backup existing data (optional)
mv data data_backup_$(date +%Y%m%d)

# Restart the system - it will initialize fresh
python main_controller.py
```

### Option 2: Attempt Migration (Experimental)

Try to rebuild the vectors dictionary from the existing FAISS index:

```bash
python migrate_vectors.py
```

**Note**: FAISS `IndexIDMap` doesn't fully support vector reconstruction, so this may not recover all vectors. Any nodes without recoverable vectors will need to be deleted.

### Option 3: Manual Cleanup

If you only need to preserve specific high-value data:

```python
from modules.memory_store import SharedMemoryManager
from config import EMBEDDING_DIM

memory = SharedMemoryManager(embedding_dim=EMBEDDING_DIM)

# Remove nodes without vectors
for uuid in list(memory.uuid_to_id.keys()):
    if uuid not in memory.vectors:
        memory.remove_node(uuid)

memory.save_snapshot()
```

## Future Prevention

The atomic persistence validation will now **prevent** this type of corruption:
- Pre-save checks ensure UUID/Vector/FAISS synchronization
- Post-load warnings alert you to data inconsistencies
- System aborts snapshots if desynchronization is detected

## Recommended Action

For your current state, I recommend **Option 1 (Clean Start)** since the system is still in development mode.
