"""
NS-DMN Benchmark Suite - Robust Edition
========================================
Senior SDET Validation Script for Neuro-Symbolic Dynamic Memory Network

Features:
- Graceful dependency handling (PyMuPDF, psutil)
- OOM-safe accuracy testing with fallback metrics
- Accurate token counting
- Metabolism verification

Tests: Ingestion -> Retrieval -> Consolidation
"""

import sys
import os
import time
import queue
import threading
import pickle
import copy
from typing import Dict, List, Tuple
from collections import Counter

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# ROBUST DEPENDENCY HANDLING
# ============================================================================

# Try importing PyMuPDF with clear fallback
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
    print("[Dependency] PyMuPDF (fitz) loaded successfully")
except ImportError:
    fitz = None
    PYMUPDF_AVAILABLE = False
    print("[WARNING] PyMuPDF not found!")
    print("          Install with: pip install pymupdf")
    print("          Falling back to file size estimation for token counting")

# Try importing pypdf as secondary fallback
try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
    if not PYMUPDF_AVAILABLE:
        print("[Dependency] Using PyPDF as fallback")
except ImportError:
    PdfReader = None
    PYPDF_AVAILABLE = False

# Try importing psutil for RAM checking
try:
    import psutil
    PSUTIL_AVAILABLE = True
    print("[Dependency] psutil loaded for RAM monitoring")
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False
    print("[INFO] psutil not available - RAM checks disabled")

# Try importing Ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
    print("[Dependency] Ollama loaded for LLM generation")
except ImportError:
    ollama = None
    OLLAMA_AVAILABLE = False
    print("[WARNING] Ollama not available - accuracy tests will use retrieval-only metrics")

from modules.memory_store import SharedMemoryManager
from modules.brain import NeuralBrain
from modules.dreamer import MemoryDreamer
from modules.ingestor import WindBellIngestor
from modules.utils import logger


# ============================================================================
# GOLDEN DATASET - Embedded Q&A Pairs
# ============================================================================

GOLD_STANDARD = [
    {
        "question": "What is the primary function of the Cerebellum in the NS-DMN architecture?",
        "answer": "The Cerebellum handles logic, retrieval, and graph management on the CPU."
    },
    {
        "question": "What is the Wind-Bell Indexing Strategy?",
        "answer": "A specific data structure to make Knowledge Graph lookups instant using Neo4j's native format."
    },
    {
        "question": "How does the system handle memory consolidation?",
        "answer": "It uses a Dreamer thread to perform entropy decay and merge nodes during REM sleep cycles."
    }
]


# ============================================================================
# HELPER FUNCTIONS - Token Counting & NLP Metrics
# ============================================================================

def estimate_tokens_from_filesize(file_path: str) -> int:
    """
    Fallback token estimation when PDF libraries unavailable.
    Approximation: 1KB ≈ 150 tokens (conservative estimate)
    """
    try:
        file_size_bytes = os.path.getsize(file_path)
        file_size_kb = file_size_bytes / 1024
        estimated_tokens = int(file_size_kb * 150)
        return estimated_tokens
    except Exception as e:
        logger.error(f"File size estimation failed: {e}")
        return 0


def count_tokens_from_pdf(file_path: str) -> int:
    """
    Accurate token counting from PDF using available libraries.
    Priority: PyMuPDF > PyPDF > file size estimation
    """
    # Try PyMuPDF first
    if PYMUPDF_AVAILABLE:
        try:
            doc = fitz.open(file_path)
            full_text = ""
            for page in doc:
                full_text += page.get_text()
            doc.close()
            # Count words as tokens
            tokens = len(full_text.split())
            print(f"  [Token Count] PyMuPDF: {tokens} tokens")
            return tokens
        except Exception as e:
            print(f"  [Token Count] PyMuPDF failed: {e}")
    
    # Try PyPDF fallback
    if PYPDF_AVAILABLE:
        try:
            reader = PdfReader(file_path)
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text()
            tokens = len(full_text.split())
            print(f"  [Token Count] PyPDF: {tokens} tokens")
            return tokens
        except Exception as e:
            print(f"  [Token Count] PyPDF failed: {e}")
    
    # Final fallback: file size estimation
    tokens = estimate_tokens_from_filesize(file_path)
    print(f"  [Token Count] File size estimation: {tokens} tokens (approximate)")
    return tokens


def get_available_ram_gb() -> float:
    """Get available RAM in GB. Returns -1 if psutil unavailable."""
    if not PSUTIL_AVAILABLE:
        return -1
    try:
        mem = psutil.virtual_memory()
        return mem.available / (1024 ** 3)  # Convert to GB
    except Exception as e:
        logger.error(f"RAM check failed: {e}")
        return -1


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip, collapse whitespace."""
    return " ".join(text.lower().strip().split())


def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization."""
    return normalize_text(text).split()


def compute_f1(prediction: str, truth: str) -> float:
    """
    Compute token-level F1 score.
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    pred_tokens = Counter(tokenize(prediction))
    truth_tokens = Counter(tokenize(truth))
    
    # Calculate intersection
    common = pred_tokens & truth_tokens
    num_common = sum(common.values())
    
    if num_common == 0:
        return 0.0
    
    num_pred = sum(pred_tokens.values())
    num_truth = sum(truth_tokens.values())
    
    if num_pred == 0 or num_truth == 0:
        return 0.0
    
    precision = num_common / num_pred
    recall = num_common / num_truth
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1


def compute_exact_match(prediction: str, truth: str) -> float:
    """Compute exact match score (1.0 if normalized strings match, 0.0 otherwise)."""
    return 1.0 if normalize_text(prediction) == normalize_text(truth) else 0.0


# ============================================================================
# BENCHMARK TEST FUNCTIONS
# ============================================================================

def run_ingestion_test(ingestor: WindBellIngestor, test_file: str) -> Dict:
    """
    Test ingestion performance with accurate token counting.
    
    Returns:
        {
            "duration_sec": float,
            "total_tokens": int,
            "total_entities": int,
            "tokens_per_sec": float (TPI),
            "entities_per_sec": float,
            "status": "PASS" | "FAIL"
        }
    """
    print("\n" + "="*70)
    print("TEST 1: INGESTION STRESS TEST")
    print("="*70)
    
    # Accurate token counting BEFORE ingestion
    print("\n[Phase 1] Counting tokens from PDF...")
    total_tokens = count_tokens_from_pdf(test_file)
    
    if total_tokens == 0:
        print("  [WARNING] Token count is 0 - metrics may be inaccurate")
    
    # Count initial entities
    with ingestor.memory.lock:
        initial_node_count = ingestor.memory.graph.number_of_nodes()
    
    print(f"\n[Phase 2] Starting ingestion...")
    print(f"  Initial nodes: {initial_node_count}")
    
    start_time = time.perf_counter()
    ingestor.ingest_document(test_file)
    duration = time.perf_counter() - start_time
    
    # Count final entities
    with ingestor.memory.lock:
        final_node_count = ingestor.memory.graph.number_of_nodes()
    
    entities_found = final_node_count - initial_node_count
    
    # Calculate metrics
    tokens_per_sec = total_tokens / duration if duration > 0 and total_tokens > 0 else 0
    entities_per_sec = entities_found / duration if duration > 0 else 0
    
    # PASS if > 100 tokens/sec (reasonable for PDF processing)
    status = "PASS" if tokens_per_sec > 100 else "FAIL"
    
    print(f"\n[Results]")
    print(f"  Duration: {duration:.2f} sec")
    print(f"  Tokens processed: {total_tokens}")
    print(f"  Entities extracted: {entities_found}")
    print(f"  TPI (Tokens/sec): {tokens_per_sec:.1f}")
    print(f"  Entities/sec: {entities_per_sec:.1f}")
    print(f"  Status: {status}")
    
    return {
        "duration_sec": duration,
        "total_tokens": total_tokens,
        "total_entities": entities_found,
        "tokens_per_sec": tokens_per_sec,
        "entities_per_sec": entities_per_sec,
        "status": status
    }


def run_accuracy_test(brain: NeuralBrain, gold_data: List[Dict]) -> Dict:
    """
    Safety-first accuracy testing with OOM protection.
    
    Falls back to Retrieval F1 (context vs answer) if LLM generation fails.
    
    Returns:
        {
            "mean_f1": float,
            "mean_em": float,
            "mean_retrieval_latency": float,
            "mean_generation_latency": float,
            "individual_scores": List[Dict],
            "mode": "generation" | "retrieval_only",
            "status": "PASS" | "FAIL" | "SKIP"
        }
    """
    print("\n" + "="*70)
    print("TEST 2: ACCURACY EVALUATION (F1 & EM)")
    print("="*70)
    
    # Check RAM availability
    available_ram = get_available_ram_gb()
    if available_ram > 0:
        print(f"\n[RAM Check] Available: {available_ram:.1f} GB")
        if available_ram < 3.0:
            print(f"  [WARNING] Low RAM detected - LLM generation may fail")
    
    if not OLLAMA_AVAILABLE:
        print("\n[WARNING] Ollama not available - using retrieval-only metrics")
        use_llm = False
    else:
        use_llm = True
    
    f1_scores = []
    em_scores = []
    retrieval_latencies = []
    generation_latencies = []
    individual_results = []
    
    llm_failures = 0
    
    for idx, item in enumerate(gold_data):
        question = item["question"]
        ground_truth = item["answer"]
        
        print(f"\n[Q{idx+1}] {question[:60]}...")
        
        try:
            # PHASE 1: Retrieval (always done)
            retrieval_start = time.perf_counter()
            context = brain.process_query(question)
            retrieval_latency = time.perf_counter() - retrieval_start
            retrieval_latencies.append(retrieval_latency)
            
            print(f"  Retrieval: {retrieval_latency:.2f}s")
            
            # PHASE 2: LLM Generation (with OOM protection)
            prediction = None
            generation_latency = 0.0
            
            if use_llm:
                try:
                    messages = [
                        {"role": "system", "content": "You are a helpful Research Assistant. Use the provided Context to answer concisely."},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
                    ]
                    
                    generation_start = time.perf_counter()
                    response = ollama.chat(
                        model="llama3",
                        messages=messages,
                        stream=False
                    )
                    generation_latency = time.perf_counter() - generation_start
                    generation_latencies.append(generation_latency)
                    
                    prediction = response['message']['content']
                    print(f"  Generation: {generation_latency:.2f}s")
                    
                except Exception as e:
                    logger.error(f"LLM Generation failed: {e}")
                    print(f"  [FALLBACK] LLM Generation Skipped (Low RAM or OOM)")
                    llm_failures += 1
                    use_llm = False  # Disable for remaining queries
            
            # Calculate F1/EM
            if prediction:
                # Generation F1 (prediction vs ground truth)
                f1 = compute_f1(prediction, ground_truth)
                em = compute_exact_match(prediction, ground_truth)
            else:
                # Retrieval F1 (context vs ground truth) - FALLBACK
                f1 = compute_f1(context, ground_truth)
                em = 0.0  # EM not applicable for retrieval
                print(f"  [FALLBACK] Using Retrieval F1 instead of Generation F1")
            
            f1_scores.append(f1)
            em_scores.append(em)
            
            individual_results.append({
                "question": question,
                "ground_truth": ground_truth,
                "prediction": prediction[:100] + "..." if prediction and len(prediction) > 100 else (prediction or "N/A"),
                "f1": f1,
                "em": em,
                "retrieval_latency": retrieval_latency,
                "generation_latency": generation_latency
            })
            
            print(f"  F1: {f1:.3f} | EM: {em:.1f}")
            
        except Exception as e:
            logger.error(f"Query failed for Q{idx+1}: {e}")
            f1_scores.append(0.0)
            em_scores.append(0.0)
            individual_results.append({
                "question": question,
                "error": str(e),
                "f1": 0.0,
                "em": 0.0
            })
    
    mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    mean_em = sum(em_scores) / len(em_scores) if em_scores else 0.0
    mean_retrieval_latency = sum(retrieval_latencies) / len(retrieval_latencies) if retrieval_latencies else 0.0
    mean_generation_latency = sum(generation_latencies) / len(generation_latencies) if generation_latencies else 0.0
    
    # Determine mode
    mode = "generation" if llm_failures == 0 and use_llm else "retrieval_only"
    
    # PASS if mean F1 > 0.50
    status = "PASS" if mean_f1 > 0.50 else "FAIL"
    
    print(f"\n[Results]")
    print(f"  Mode: {mode}")
    print(f"  Mean F1: {mean_f1:.3f}")
    print(f"  Mean EM: {mean_em:.3f}")
    print(f"  Mean Retrieval Latency: {mean_retrieval_latency:.2f}s")
    if mean_generation_latency > 0:
        print(f"  Mean Generation Latency: {mean_generation_latency:.2f}s")
    print(f"  Status: {status}")
    
    return {
        "mean_f1": mean_f1,
        "mean_em": mean_em,
        "mean_retrieval_latency": mean_retrieval_latency,
        "mean_generation_latency": mean_generation_latency,
        "individual_scores": individual_results,
        "mode": mode,
        "status": status
    }


def run_consolidation_test(memory_manager: SharedMemoryManager, dreamer: MemoryDreamer) -> Dict:
    """
    Metabolism verification - checks if REM cycle creates concept nodes.
    
    Returns:
        {
            "initial_nodes": int,
            "final_nodes": int,
            "new_concepts": int,
            "consolidation_ratio": float,
            "status": "ACTIVE" | "PASSIVE"
        }
    """
    print("\n" + "="*70)
    print("TEST 3: METABOLISM VERIFICATION (Consolidation)")
    print("="*70)
    
    # Count initial state
    with memory_manager.lock:
        initial_node_count = memory_manager.graph.number_of_nodes()
        
        # Count existing concept nodes (check metadata)
        initial_concepts = sum(
            1 for node, data in memory_manager.graph.nodes(data=True)
            if data.get('is_concept') == True or str(node).startswith("concept_")
        )
    
    print(f"\n[Initial State]")
    print(f"  Total nodes: {initial_node_count}")
    print(f"  Concept nodes: {initial_concepts}")
    
    # Trigger REM cycle
    print(f"\n[Triggering REM Cycle...]")
    try:
        dreamer.run_rem_cycle()
        print(f"  REM cycle completed")
    except Exception as e:
        logger.error(f"REM cycle failed: {e}")
        return {
            "initial_nodes": initial_node_count,
            "final_nodes": initial_node_count,
            "new_concepts": 0,
            "consolidation_ratio": 0.0,
            "status": "ERROR",
            "error": str(e)
        }
    
    # Count final state
    with memory_manager.lock:
        final_node_count = memory_manager.graph.number_of_nodes()
        
        # Count new concept nodes
        final_concepts = sum(
            1 for node, data in memory_manager.graph.nodes(data=True)
            if data.get('is_concept') == True or str(node).startswith("concept_")
        )
    
    new_concepts = final_concepts - initial_concepts
    
    # Calculate consolidation ratio
    consolidation_ratio = (new_concepts / final_node_count * 100) if final_node_count > 0 else 0.0
    
    # ACTIVE if new concepts were created
    status = "ACTIVE" if new_concepts > 0 else "PASSIVE"
    
    print(f"\n[Final State]")
    print(f"  Total nodes: {final_node_count}")
    print(f"  Concept nodes: {final_concepts}")
    print(f"  New concepts: {new_concepts}")
    print(f"  Consolidation ratio: {consolidation_ratio:.2f}%")
    print(f"  Status: {status}")
    
    return {
        "initial_nodes": initial_node_count,
        "final_nodes": final_node_count,
        "new_concepts": new_concepts,
        "consolidation_ratio": consolidation_ratio,
        "status": status
    }


# ============================================================================
# MAIN BENCHMARK ORCHESTRATOR
# ============================================================================

def run_benchmark():
    """
    Main benchmark orchestrator.
    Runs all tests and generates Performance Certificate.
    """
    print("\n" + "="*70)
    print(" NS-DMN BENCHMARK SUITE - ROBUST EDITION")
    print(" Senior SDET Validation - Full Lifecycle Test")
    print("="*70)
    
    # Initialize system components
    print("\n[Init] Loading NS-DMN System Components...")
    
    memory_manager = SharedMemoryManager()
    stm_queue = queue.Queue()
    
    # Start dreamer thread
    dreamer = MemoryDreamer(memory_manager, stm_queue)
    dreamer.start()
    
    brain = NeuralBrain(memory_manager, stm_queue)
    ingestor = WindBellIngestor(memory_manager, brain.encoder)
    
    print("[Init] System Online.\n")
    
    # Collect all metrics
    metrics = {}
    
    # === TEST 1: Ingestion Stress Test ===
    test_file = os.path.join(os.path.dirname(__file__), "test", "testing.pdf")
    
    if not os.path.exists(test_file):
        print(f"[ERROR] Test file not found: {test_file}")
        print("Please ensure test/testing.pdf exists.")
        dreamer.stop()
        dreamer.join(timeout=5)
        return
    
    metrics["ingestion"] = run_ingestion_test(ingestor, test_file)
    
    # === TEST 2: Accuracy Evaluation ===
    metrics["accuracy"] = run_accuracy_test(brain, GOLD_STANDARD)
    
    # === TEST 3: Consolidation Test ===
    metrics["consolidation"] = run_consolidation_test(memory_manager, dreamer)
    
    # Shutdown dreamer
    print("\n[Shutdown] Stopping Dreamer...")
    stm_queue.put("POISON_PILL")
    dreamer.stop()
    dreamer.join(timeout=5)
    
    # ========================================================================
    # GENERATE PERFORMANCE CERTIFICATE
    # ========================================================================
    
    print("\n" + "="*70)
    print(" PERFORMANCE CERTIFICATE - NS-DMN BENCHMARK RESULTS")
    print("="*70)
    
    # Header
    print(f"\n{'Metric':<40} | {'Value':<20} | {'Status':<10}")
    print("-"*70)
    
    # Ingestion Metrics
    ing = metrics["ingestion"]
    print(f"{'Ingestion Speed (TPI)':<40} | {ing['tokens_per_sec']:.1f} tokens/sec | {ing['status']:<10}")
    print(f"{'Entities Extracted':<40} | {ing['total_entities']:<20} | {'INFO':<10}")
    print(f"{'Ingestion Duration':<40} | {ing['duration_sec']:.2f} sec | {'INFO':<10}")
    
    # Accuracy Metrics
    acc = metrics["accuracy"]
    print(f"{'Mean F1 Score':<40} | {acc['mean_f1']:.3f} | {acc['status']:<10}")
    print(f"{'Mean Exact Match':<40} | {acc['mean_em']:.3f} | {'INFO':<10}")
    print(f"{'Accuracy Mode':<40} | {acc.get('mode', 'N/A'):<20} | {'INFO':<10}")
    print(f"{'Mean Retrieval Latency':<40} | {acc['mean_retrieval_latency']:.2f} sec | {'INFO':<10}")
    if acc['mean_generation_latency'] > 0:
        print(f"{'Mean Generation Latency':<40} | {acc['mean_generation_latency']:.2f} sec | {'INFO':<10}")
    
    # Consolidation Metrics
    cons = metrics["consolidation"]
    if "error" not in cons:
        print(f"{'Consolidation Ratio':<40} | {cons['consolidation_ratio']:.2f}% | {cons['status']:<10}")
        print(f"{'New Concept Nodes':<40} | {cons['new_concepts']:<20} | {'INFO':<10}")
    else:
        print(f"{'Consolidation Test':<40} | {'ERROR':<20} | {'FAIL':<10}")
    
    print("-"*70)
    
    # Overall Status
    all_passed = (
        ing["status"] == "PASS" and 
        acc["status"] == "PASS" and
        "error" not in cons
    )
    
    overall_status = "✓ ALL TESTS PASSED" if all_passed else "✗ SOME TESTS FAILED"
    print(f"\n{'OVERALL STATUS':<40} | {overall_status:<30}")
    
    print("\n" + "="*70)
    print(" Benchmark Complete. Review results above.")
    print("="*70)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        run_benchmark()
    except KeyboardInterrupt:
        print("\n\n[System] Benchmark interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] Benchmark failed: {e}")
        logger.exception("Benchmark suite error")
        sys.exit(1)
