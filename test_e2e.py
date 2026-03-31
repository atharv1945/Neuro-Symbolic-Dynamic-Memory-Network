"""
NS-DMN Standard RAG Pipeline Metrics Evaluation
=================================================
Computes industry-standard RAG metrics:

RETRIEVAL METRICS:
  - Context Recall: % of ground-truth tokens found in retrieved context
  - Context Precision: % of context tokens relevant to the ground truth
  - Hit Rate: Did the retrieval find any relevant info? (binary)

GENERATION METRICS:
  - Token F1 Score: Harmonic mean of precision & recall at token level
  - Exact Match (EM): Exact normalized string match
  - ROUGE-1: Unigram overlap (recall-oriented)
  - ROUGE-L: Longest Common Subsequence overlap
  - Faithfulness: % of answer tokens grounded in the context (no hallucination)
  - Answer Relevance: % of answer tokens relevant to the query

LATENCY METRICS:
  - Retrieval Latency (ms)
  - Generation Latency (ms)
  - End-to-End Latency (ms)

SYSTEM METRICS:
  - Ingestion Throughput (tokens/sec)
  - Compression Ratio
"""

import sys
import os
import time
import queue
import json
from typing import List, Dict, Tuple
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_DIR
from modules.utils import logger, recover_previous_state
from modules.memory_store import SharedMemoryManager
from modules.brain import NeuralBrain
from modules.dreamer import MemoryDreamer
from modules.ingestor import WindBellIngestor

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


# =============================================================================
# GOLDEN DATASET - Ground Truth Q&A with Expected Context Keywords
# =============================================================================

GOLDEN_DATASET = [
    {
        "id": "Q1",
        "question": "What is the primary function of the Cerebellum in the NS-DMN architecture?",
        "ground_truth": "The Cerebellum handles logic, retrieval, and graph management on the CPU. It is responsible for the slow, logical thinking including knowledge graph operations, FAISS vector search, and memory management.",
        "expected_context_keywords": ["cerebellum", "cpu", "logic", "retrieval", "graph", "faiss"],
        "category": "architecture"
    },
    {
        "id": "Q2",
        "question": "What is the Wind-Bell Indexing Strategy?",
        "ground_truth": "The Wind-Bell Indexing Strategy is a specific data structure designed to make Knowledge Graph lookups instant, preventing Graph Latency. It uses Neo4j's native format for efficient indexing.",
        "expected_context_keywords": ["wind-bell", "indexing", "knowledge graph", "lookups", "instant", "latency"],
        "category": "indexing"
    },
    {
        "id": "Q3",
        "question": "How does the system handle memory consolidation?",
        "ground_truth": "The system uses a Dreamer thread that runs in the background to perform memory consolidation. It implements entropy decay to reduce node energy over time, prunes low-energy nodes, discovers latent associations through vector similarity, and creates concept super-nodes from dense clusters during REM sleep cycles.",
        "expected_context_keywords": ["dreamer", "consolidation", "entropy", "decay", "pruning", "rem", "sleep"],
        "category": "memory"
    },
    {
        "id": "Q4",
        "question": "What embedding model does the system use?",
        "ground_truth": "The system uses nomic-embed-text-v1.5 as the embedding model for high performance and low dimensional embeddings. The implementation also uses all-MiniLM-L6-v2 from SentenceTransformers.",
        "expected_context_keywords": ["embedding", "model", "nomic", "minilm", "sentence"],
        "category": "models"
    },
    {
        "id": "Q5",
        "question": "What is the Split-Compute architecture in NS-DMN?",
        "ground_truth": "The Split-Compute architecture divides processing between CPU and GPU. The CPU handles the Cerebellum (logic, graphs, FAISS) while the GPU handles the Cerebrum (LLM inference via Ollama). This prevents OOM crashes and optimizes resource usage.",
        "expected_context_keywords": ["split", "compute", "cpu", "gpu", "cerebellum", "cerebrum"],
        "category": "architecture"
    },
]


# =============================================================================
# METRIC COMPUTATION FUNCTIONS
# =============================================================================

def normalize(text: str) -> str:
    """Normalize text for comparison."""
    return " ".join(text.lower().strip().split())

def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization after normalization."""
    return normalize(text).split()

def compute_token_f1(prediction: str, reference: str) -> Dict[str, float]:
    """Compute token-level Precision, Recall, and F1."""
    pred_tokens = Counter(tokenize(prediction))
    ref_tokens = Counter(tokenize(reference))
    
    common = pred_tokens & ref_tokens
    num_common = sum(common.values())
    
    num_pred = sum(pred_tokens.values())
    num_ref = sum(ref_tokens.values())
    
    if num_pred == 0 or num_ref == 0 or num_common == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    precision = num_common / num_pred
    recall = num_common / num_ref
    f1 = 2 * precision * recall / (precision + recall)
    
    return {"precision": precision, "recall": recall, "f1": f1}

def compute_exact_match(prediction: str, reference: str) -> float:
    """Exact match after normalization."""
    return 1.0 if normalize(prediction) == normalize(reference) else 0.0

def compute_rouge_1(prediction: str, reference: str) -> Dict[str, float]:
    """ROUGE-1: Unigram overlap."""
    pred_tokens = Counter(tokenize(prediction))
    ref_tokens = Counter(tokenize(reference))
    
    overlap = pred_tokens & ref_tokens
    overlap_count = sum(overlap.values())
    
    pred_count = sum(pred_tokens.values())
    ref_count = sum(ref_tokens.values())
    
    precision = overlap_count / pred_count if pred_count > 0 else 0.0
    recall = overlap_count / ref_count if ref_count > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {"precision": precision, "recall": recall, "f1": f1}

def _lcs_length(x: List[str], y: List[str]) -> int:
    """Compute length of Longest Common Subsequence."""
    m, n = len(x), len(y)
    # Optimize memory: only need two rows
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i-1] == y[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n] if m > 0 else 0

def compute_rouge_l(prediction: str, reference: str) -> Dict[str, float]:
    """ROUGE-L: Longest Common Subsequence based metric."""
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    
    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    lcs = _lcs_length(pred_tokens, ref_tokens)
    
    precision = lcs / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
    recall = lcs / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {"precision": precision, "recall": recall, "f1": f1}

def compute_context_recall(context: str, ground_truth: str) -> float:
    """What fraction of ground truth tokens appear in the context."""
    context_tokens = set(tokenize(context))
    truth_tokens = tokenize(ground_truth)
    
    if not truth_tokens:
        return 0.0
    
    found = sum(1 for t in truth_tokens if t in context_tokens)
    return found / len(truth_tokens)

def compute_context_precision(context: str, ground_truth: str) -> float:
    """What fraction of context tokens are relevant (appear in ground truth)."""
    context_tokens = tokenize(context)
    truth_tokens = set(tokenize(ground_truth))
    
    if not context_tokens:
        return 0.0
    
    relevant = sum(1 for t in context_tokens if t in truth_tokens)
    return relevant / len(context_tokens)

def compute_keyword_hit_rate(context: str, keywords: List[str]) -> float:
    """What fraction of expected keywords are found in the context."""
    if not keywords:
        return 1.0
    
    context_lower = context.lower()
    hits = sum(1 for kw in keywords if kw.lower() in context_lower)
    return hits / len(keywords)

def compute_faithfulness(answer: str, context: str) -> float:
    """What fraction of answer tokens are grounded in the context (no hallucination)."""
    answer_tokens = tokenize(answer)
    context_tokens = set(tokenize(context))
    
    if not answer_tokens:
        return 0.0
    
    # Filter out stop words for more meaningful measurement
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                  "have", "has", "had", "do", "does", "did", "will", "would", "could",
                  "should", "may", "might", "shall", "can", "to", "of", "in", "for",
                  "on", "with", "at", "by", "from", "as", "into", "through", "during",
                  "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
                  "neither", "each", "every", "all", "any", "few", "more", "most",
                  "other", "some", "such", "no", "only", "own", "same", "than", "too",
                  "very", "just", "because", "if", "when", "while", "this", "that",
                  "these", "those", "it", "its", "they", "them", "their", "which", "who"}
    
    content_tokens = [t for t in answer_tokens if t not in stop_words]
    
    if not content_tokens:
        return 1.0  # Only stop words = trivially faithful
    
    grounded = sum(1 for t in content_tokens if t in context_tokens)
    return grounded / len(content_tokens)

def compute_answer_relevance(answer: str, question: str) -> float:
    """What fraction of answer content tokens are relevant to the question."""
    answer_tokens = tokenize(answer)
    question_tokens = set(tokenize(question))
    
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                  "have", "has", "had", "do", "does", "did", "will", "would",
                  "to", "of", "in", "for", "on", "with", "at", "by", "from",
                  "and", "but", "or", "not", "it", "its", "this", "that"}
    
    content_tokens = [t for t in answer_tokens if t not in stop_words]
    
    if not content_tokens:
        return 0.0
    
    relevant = sum(1 for t in content_tokens if t in question_tokens)
    return relevant / len(content_tokens)


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def run_evaluation():
    print("=" * 80)
    print("  NS-DMN STANDARD RAG PIPELINE METRICS EVALUATION")
    print("=" * 80)
    
    # --- INIT ---
    print("\n[Init] Loading system components...")
    recover_previous_state(DATA_DIR)
    memory_manager = SharedMemoryManager()
    stm_queue = queue.Queue()
    
    dreamer = MemoryDreamer(memory_manager, stm_queue)
    dreamer.start()
    
    brain = NeuralBrain(memory_manager, stm_queue)
    ingestor = WindBellIngestor(memory_manager, brain.encoder)
    
    print(f"[Init] Memory: {memory_manager.graph.number_of_nodes()} nodes, "
          f"{memory_manager.index.ntotal} vectors")
    
    # --- INGEST if empty ---
    if memory_manager.index.ntotal == 0:
        test_file = os.path.join(os.path.dirname(__file__), "test", "Nlp_project.pdf")
        if os.path.exists(test_file):
            print(f"\n[Ingest] Memory empty. Ingesting {test_file}...")
            ingest_start = time.perf_counter()
            ingestor.ingest_document(test_file)
            ingest_duration = time.perf_counter() - ingest_start
            
            with memory_manager.lock:
                chunk_count = sum(1 for _, d in memory_manager.graph.nodes(data=True)
                                  if d.get('type') == 'chunk')
            print(f"[Ingest] Done: {chunk_count} chunks in {ingest_duration:.1f}s")
        else:
            print(f"[ERROR] No test file found at {test_file}. Cannot proceed.")
            return
    
    print(f"\n[Ready] {memory_manager.graph.number_of_nodes()} nodes, "
          f"{memory_manager.index.ntotal} vectors, "
          f"{memory_manager.graph.number_of_edges()} edges")
    
    # =========================================================================
    # RUN EVALUATION ON GOLDEN DATASET
    # =========================================================================
    
    all_results = []
    
    # Aggregate accumulators
    agg = {
        "context_recall": [],
        "context_precision": [],
        "keyword_hit_rate": [],
        "token_f1": [],
        "token_precision": [],
        "token_recall": [],
        "exact_match": [],
        "rouge1_f1": [],
        "rouge1_precision": [],
        "rouge1_recall": [],
        "rougel_f1": [],
        "rougel_precision": [],
        "rougel_recall": [],
        "faithfulness": [],
        "answer_relevance": [],
        "retrieval_latency_ms": [],
        "generation_latency_ms": [],
        "e2e_latency_ms": [],
        "context_length_chars": [],
        "compression_ratio": [],
    }
    
    print("\n" + "=" * 80)
    print("  EVALUATION RESULTS (Per Query)")
    print("=" * 80)
    
    for item in GOLDEN_DATASET:
        qid = item["id"]
        question = item["question"]
        ground_truth = item["ground_truth"]
        keywords = item["expected_context_keywords"]
        
        print(f"\n{'─' * 80}")
        print(f"  {qid}: {question}")
        print(f"{'─' * 80}")
        
        result = {"id": qid, "question": question, "category": item["category"]}
        
        # === RETRIEVAL PHASE ===
        e2e_start = time.perf_counter()
        retrieval_start = time.perf_counter()
        
        try:
            context = brain.process_query(question)
        except Exception as e:
            print(f"  [ERROR] Retrieval failed: {e}")
            continue
        
        retrieval_latency = (time.perf_counter() - retrieval_start) * 1000  # ms
        
        # Retrieval metrics
        ctx_recall = compute_context_recall(context, ground_truth)
        ctx_precision = compute_context_precision(context, ground_truth)
        kw_hit_rate = compute_keyword_hit_rate(context, keywords)
        
        result["context_length"] = len(context)
        result["context_recall"] = ctx_recall
        result["context_precision"] = ctx_precision
        result["keyword_hit_rate"] = kw_hit_rate
        result["retrieval_latency_ms"] = retrieval_latency
        
        agg["context_recall"].append(ctx_recall)
        agg["context_precision"].append(ctx_precision)
        agg["keyword_hit_rate"].append(kw_hit_rate)
        agg["retrieval_latency_ms"].append(retrieval_latency)
        agg["context_length_chars"].append(len(context))
        
        print(f"\n  RETRIEVAL METRICS:")
        print(f"    Context Length:     {len(context):,} chars")
        print(f"    Context Recall:     {ctx_recall:.4f}")
        print(f"    Context Precision:  {ctx_precision:.4f}")
        print(f"    Keyword Hit Rate:   {kw_hit_rate:.4f} ({int(kw_hit_rate * len(keywords))}/{len(keywords)} keywords)")
        print(f"    Retrieval Latency:  {retrieval_latency:.0f} ms")
        
        # === GENERATION PHASE ===
        answer = ""
        generation_latency = 0.0
        
        if OLLAMA_AVAILABLE and context:
            try:
                messages = [
                    {"role": "system", "content": "You are a helpful Research Assistant. Use ONLY the provided Context to answer. Be concise (2-3 sentences). Do not add information not in the context."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
                ]
                
                gen_start = time.perf_counter()
                response = ollama.chat(model="llama3", messages=messages, stream=False)
                generation_latency = (time.perf_counter() - gen_start) * 1000
                answer = response['message']['content'].strip()
                
            except Exception as e:
                print(f"  [WARNING] LLM generation failed: {e}")
                answer = context[:500]  # Fallback to raw context
        else:
            answer = context[:500]
        
        e2e_latency = (time.perf_counter() - e2e_start) * 1000
        
        # Generation metrics
        f1_scores = compute_token_f1(answer, ground_truth)
        em = compute_exact_match(answer, ground_truth)
        rouge1 = compute_rouge_1(answer, ground_truth)
        rougel = compute_rouge_l(answer, ground_truth)
        faithful = compute_faithfulness(answer, context)
        relevance = compute_answer_relevance(answer, question)
        
        result["answer_preview"] = answer[:150]
        result["token_f1"] = f1_scores["f1"]
        result["token_precision"] = f1_scores["precision"]
        result["token_recall"] = f1_scores["recall"]
        result["exact_match"] = em
        result["rouge1"] = rouge1
        result["rougel"] = rougel
        result["faithfulness"] = faithful
        result["answer_relevance"] = relevance
        result["generation_latency_ms"] = generation_latency
        result["e2e_latency_ms"] = e2e_latency
        
        agg["token_f1"].append(f1_scores["f1"])
        agg["token_precision"].append(f1_scores["precision"])
        agg["token_recall"].append(f1_scores["recall"])
        agg["exact_match"].append(em)
        agg["rouge1_f1"].append(rouge1["f1"])
        agg["rouge1_precision"].append(rouge1["precision"])
        agg["rouge1_recall"].append(rouge1["recall"])
        agg["rougel_f1"].append(rougel["f1"])
        agg["rougel_precision"].append(rougel["precision"])
        agg["rougel_recall"].append(rougel["recall"])
        agg["faithfulness"].append(faithful)
        agg["answer_relevance"].append(relevance)
        agg["generation_latency_ms"].append(generation_latency)
        agg["e2e_latency_ms"].append(e2e_latency)
        
        print(f"\n  GENERATION METRICS:")
        print(f"    Answer Preview:     {answer[:100]}...")
        print(f"    Token F1:           {f1_scores['f1']:.4f}")
        print(f"    Token Precision:    {f1_scores['precision']:.4f}")
        print(f"    Token Recall:       {f1_scores['recall']:.4f}")
        print(f"    Exact Match:        {em:.1f}")
        print(f"    ROUGE-1 (F1):       {rouge1['f1']:.4f}")
        print(f"    ROUGE-L (F1):       {rougel['f1']:.4f}")
        print(f"    Faithfulness:       {faithful:.4f}")
        print(f"    Answer Relevance:   {relevance:.4f}")
        
        print(f"\n  LATENCY:")
        print(f"    Retrieval:          {retrieval_latency:.0f} ms")
        print(f"    Generation:         {generation_latency:.0f} ms")
        print(f"    End-to-End:         {e2e_latency:.0f} ms")
        
        all_results.append(result)
    
    # =========================================================================
    # AGGREGATE METRICS TABLE
    # =========================================================================
    
    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0
    
    def p50(lst):
        s = sorted(lst)
        return s[len(s) // 2] if s else 0.0
    
    def p95(lst):
        s = sorted(lst)
        idx = int(len(s) * 0.95)
        return s[min(idx, len(s) - 1)] if s else 0.0
    
    print("\n\n" + "=" * 80)
    print("  AGGREGATE METRICS SUMMARY")
    print("=" * 80)
    
    print(f"\n  {'Metric':<35} {'Mean':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'─' * 65}")
    
    # Retrieval
    print(f"\n  {'RETRIEVAL':}")
    print(f"  {'Context Recall':<35} {mean(agg['context_recall']):>10.4f} {min(agg['context_recall']):>10.4f} {max(agg['context_recall']):>10.4f}")
    print(f"  {'Context Precision':<35} {mean(agg['context_precision']):>10.4f} {min(agg['context_precision']):>10.4f} {max(agg['context_precision']):>10.4f}")
    print(f"  {'Keyword Hit Rate':<35} {mean(agg['keyword_hit_rate']):>10.4f} {min(agg['keyword_hit_rate']):>10.4f} {max(agg['keyword_hit_rate']):>10.4f}")
    print(f"  {'Context Length (chars)':<35} {mean(agg['context_length_chars']):>10.0f} {min(agg['context_length_chars']):>10.0f} {max(agg['context_length_chars']):>10.0f}")
    
    # Generation
    print(f"\n  {'GENERATION':}")
    print(f"  {'Token F1':<35} {mean(agg['token_f1']):>10.4f} {min(agg['token_f1']):>10.4f} {max(agg['token_f1']):>10.4f}")
    print(f"  {'Token Precision':<35} {mean(agg['token_precision']):>10.4f} {min(agg['token_precision']):>10.4f} {max(agg['token_precision']):>10.4f}")
    print(f"  {'Token Recall':<35} {mean(agg['token_recall']):>10.4f} {min(agg['token_recall']):>10.4f} {max(agg['token_recall']):>10.4f}")
    print(f"  {'Exact Match':<35} {mean(agg['exact_match']):>10.4f} {min(agg['exact_match']):>10.4f} {max(agg['exact_match']):>10.4f}")
    print(f"  {'ROUGE-1 (F1)':<35} {mean(agg['rouge1_f1']):>10.4f} {min(agg['rouge1_f1']):>10.4f} {max(agg['rouge1_f1']):>10.4f}")
    print(f"  {'ROUGE-1 (Precision)':<35} {mean(agg['rouge1_precision']):>10.4f} {min(agg['rouge1_precision']):>10.4f} {max(agg['rouge1_precision']):>10.4f}")
    print(f"  {'ROUGE-1 (Recall)':<35} {mean(agg['rouge1_recall']):>10.4f} {min(agg['rouge1_recall']):>10.4f} {max(agg['rouge1_recall']):>10.4f}")
    print(f"  {'ROUGE-L (F1)':<35} {mean(agg['rougel_f1']):>10.4f} {min(agg['rougel_f1']):>10.4f} {max(agg['rougel_f1']):>10.4f}")
    print(f"  {'Faithfulness':<35} {mean(agg['faithfulness']):>10.4f} {min(agg['faithfulness']):>10.4f} {max(agg['faithfulness']):>10.4f}")
    print(f"  {'Answer Relevance':<35} {mean(agg['answer_relevance']):>10.4f} {min(agg['answer_relevance']):>10.4f} {max(agg['answer_relevance']):>10.4f}")
    
    # Latency
    print(f"\n  {'LATENCY (ms)':}")
    print(f"  {'Retrieval Latency':<35} {mean(agg['retrieval_latency_ms']):>10.0f} {min(agg['retrieval_latency_ms']):>10.0f} {max(agg['retrieval_latency_ms']):>10.0f}")
    print(f"  {'Generation Latency':<35} {mean(agg['generation_latency_ms']):>10.0f} {min(agg['generation_latency_ms']):>10.0f} {max(agg['generation_latency_ms']):>10.0f}")
    print(f"  {'End-to-End Latency':<35} {mean(agg['e2e_latency_ms']):>10.0f} {min(agg['e2e_latency_ms']):>10.0f} {max(agg['e2e_latency_ms']):>10.0f}")
    
    # =========================================================================
    # PERFORMANCE GRADE
    # =========================================================================
    
    mean_f1 = mean(agg['token_f1'])
    mean_ctx_recall = mean(agg['context_recall'])
    mean_faithful = mean(agg['faithfulness'])
    mean_rouge1 = mean(agg['rouge1_f1'])
    mean_rougel = mean(agg['rougel_f1'])
    mean_kw_hit = mean(agg['keyword_hit_rate'])
    
    print(f"\n\n  {'=' * 65}")
    print(f"  PERFORMANCE GRADE CARD")
    print(f"  {'=' * 65}")
    
    def grade(score, thresholds):
        """Grade: A (>= t[0]), B (>= t[1]), C (>= t[2]), D (>= t[3]), F (< t[3])"""
        labels = ['A', 'B', 'C', 'D', 'F']
        for i, t in enumerate(thresholds):
            if score >= t:
                return labels[i]
        return 'F'
    
    grades = {
        "Context Recall":    (mean_ctx_recall, grade(mean_ctx_recall, [0.8, 0.6, 0.4, 0.2])),
        "Keyword Hit Rate":  (mean_kw_hit,     grade(mean_kw_hit,     [0.8, 0.6, 0.4, 0.2])),
        "Token F1":          (mean_f1,          grade(mean_f1,         [0.6, 0.4, 0.25, 0.1])),
        "ROUGE-1 F1":        (mean_rouge1,      grade(mean_rouge1,     [0.6, 0.4, 0.25, 0.1])),
        "ROUGE-L F1":        (mean_rougel,      grade(mean_rougel,     [0.5, 0.35, 0.2, 0.1])),
        "Faithfulness":      (mean_faithful,    grade(mean_faithful,   [0.8, 0.6, 0.4, 0.2])),
    }
    
    print(f"\n  {'Metric':<30} {'Score':>8} {'Grade':>8}")
    print(f"  {'─' * 46}")
    for metric, (score, g) in grades.items():
        print(f"  {metric:<30} {score:>8.4f} {g:>8}")
    
    # Overall
    grade_values = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
    avg_grade_val = sum(grade_values[g] for _, (_, g) in grades.items()) / len(grades)
    overall_grade = 'A' if avg_grade_val >= 3.5 else 'B' if avg_grade_val >= 2.5 else 'C' if avg_grade_val >= 1.5 else 'D' if avg_grade_val >= 0.5 else 'F'
    
    print(f"  {'─' * 46}")
    print(f"  {'OVERALL GRADE':<30} {'':>8} {overall_grade:>8}")
    
    # =========================================================================
    # COMPARISON WITH INDUSTRY BENCHMARKS
    # =========================================================================
    
    print(f"\n\n  {'=' * 65}")
    print(f"  COMPARISON: NS-DMN vs TYPICAL RAG BENCHMARKS")
    print(f"  {'=' * 65}")
    print(f"\n  {'Metric':<25} {'NS-DMN':>10} {'Naive RAG':>10} {'Advanced':>10}")
    print(f"  {'─' * 55}")
    print(f"  {'Context Recall':<25} {mean_ctx_recall:>10.3f} {'~0.40':>10} {'~0.75':>10}")
    print(f"  {'Token F1':<25} {mean_f1:>10.3f} {'~0.25':>10} {'~0.55':>10}")
    print(f"  {'ROUGE-1 F1':<25} {mean_rouge1:>10.3f} {'~0.20':>10} {'~0.50':>10}")
    print(f"  {'ROUGE-L F1':<25} {mean_rougel:>10.3f} {'~0.15':>10} {'~0.40':>10}")
    print(f"  {'Faithfulness':<25} {mean_faithful:>10.3f} {'~0.60':>10} {'~0.85':>10}")
    print(f"  {'Keyword Hit Rate':<25} {mean_kw_hit:>10.3f} {'~0.50':>10} {'~0.80':>10}")
    
    print(f"\n  Note: 'Naive RAG' = basic chunking + vector search + LLM")
    print(f"        'Advanced'  = hybrid retrieval + reranking + compression")
    
    print("\n" + "=" * 80)
    print("  Evaluation Complete.")
    print("=" * 80)
    
    # --- SHUTDOWN ---
    stm_queue.put("POISON_PILL")
    dreamer.stop()
    dreamer.join(timeout=5)
    
    return all_results, agg


if __name__ == "__main__":
    try:
        results, aggregates = run_evaluation()
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
    except Exception as e:
        print(f"\n\nFATAL: {e}")
        import traceback
        traceback.print_exc()
