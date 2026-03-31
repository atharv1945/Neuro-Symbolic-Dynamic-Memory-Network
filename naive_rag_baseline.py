"""
Naive RAG Baseline Pipeline
============================
A deliberately simple, unoptimized RAG pipeline for benchmarking against NS-DMN.

Design choices (intentionally naive):
  - Fixed-width character slicing (no sentence awareness)
  - Raw L2 distance via IndexFlatL2 (no L2-normalization / no cosine similarity)
  - No reranking, no compression, no graph memory
  - Blind top-5 concatenation into a basic prompt
"""

import os
import sys
import time
import json
import numpy as np
import faiss
import requests

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers is required. pip install sentence-transformers")
    sys.exit(1)

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


class LocalNaiveRAG:
    """
    A deliberately naive RAG pipeline:
      1. Fixed-width chunking (1000 chars, 100 overlap)
      2. FAISS IndexFlatL2 with raw (un-normalized) embeddings
      3. Top-5 retrieval by L2 distance
      4. Blind concatenation → simple prompt → Ollama LLaMA 3
    """

    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    TOP_K = 5
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    OLLAMA_URL = "http://localhost:11434/api/generate"
    LLM_MODEL = "llama3"

    def __init__(self):
        print("[NaiveRAG] Initializing embedding model...")
        self.encoder = SentenceTransformer(self.EMBEDDING_MODEL, device="cpu")

        # Raw L2 index — intentionally no normalization
        self.index = faiss.IndexFlatL2(self.EMBEDDING_DIM)
        self.chunks: list[str] = []

        print(f"[NaiveRAG] Ready. FAISS index type: IndexFlatL2 (raw L2 distance)")

    # ------------------------------------------------------------------
    # 1. Naive Chunking
    # ------------------------------------------------------------------
    def _naive_chunk(self, text: str) -> list[str]:
        """
        Fixed-width character slicing.  Blindly cuts text every CHUNK_SIZE
        characters with CHUNK_OVERLAP overlap.  No sentence awareness.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.CHUNK_SIZE
            chunks.append(text[start:end])
            start += self.CHUNK_SIZE - self.CHUNK_OVERLAP
        return chunks

    # ------------------------------------------------------------------
    # 2. Text Extraction (PDF helper)
    # ------------------------------------------------------------------
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """Extract raw text from a PDF using PyMuPDF or pypdf."""
        if fitz:
            doc = fitz.open(file_path)
            return "\n".join(page.get_text() for page in doc)
        elif PdfReader:
            reader = PdfReader(file_path)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            raise RuntimeError(
                "No PDF library available. Install pymupdf or pypdf."
            )

    # ------------------------------------------------------------------
    # 3. Ingestion
    # ------------------------------------------------------------------
    def ingest(self, text: str) -> None:
        """Chunk → Embed → Load into FAISS.  No normalization."""
        print(f"[NaiveRAG] Ingesting {len(text):,} characters...")
        new_chunks = self._naive_chunk(text)
        self.chunks.extend(new_chunks)
        print(f"[NaiveRAG] Created {len(new_chunks)} chunks "
              f"({self.CHUNK_SIZE} chars, {self.CHUNK_OVERLAP} overlap)")

        # Batch-encode all chunks (raw vectors — no L2-normalization)
        vectors = self.encoder.encode(new_chunks, show_progress_bar=True)
        vectors = np.array(vectors, dtype=np.float32)

        self.index.add(vectors)
        print(f"[NaiveRAG] FAISS index now contains {self.index.ntotal} vectors")

    def ingest_pdf(self, file_path: str) -> None:
        """Convenience: extract text from PDF then ingest."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        text = self.extract_text_from_pdf(file_path)
        self.ingest(text)

    # ------------------------------------------------------------------
    # 4. Retrieval
    # ------------------------------------------------------------------
    def retrieve(self, query: str) -> list[dict]:
        """
        Embed query (raw, un-normalized) → search FAISS → return top-K
        chunks with their L2 distances.
        """
        query_vec = self.encoder.encode([query])
        query_vec = np.array(query_vec, dtype=np.float32)

        distances, indices = self.index.search(query_vec, self.TOP_K)

        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:
                continue
            results.append({
                "rank": rank + 1,
                "chunk_id": int(idx),
                "l2_distance": float(dist),
                "text": self.chunks[idx],
            })
        return results

    # ------------------------------------------------------------------
    # 5. Generation (Ollama)
    # ------------------------------------------------------------------
    def generate(self, query: str, context_chunks: list[dict]) -> str:
        """
        Blind concatenation of top-K chunks → simple prompt → Ollama.
        No compression.  No reranking.  No streaming.
        """
        # Blind double-newline concatenation
        context_blob = "\n\n".join(c["text"] for c in context_chunks)

        prompt = (
            f"Context:\n{context_blob}\n\n"
            f"Question: {query}\n\n"
            "Answer the question using the context above. Be concise."
        )

        payload = {
            "model": self.LLM_MODEL,
            "prompt": prompt,
            "stream": False,
        }

        try:
            resp = requests.post(self.OLLAMA_URL, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except requests.exceptions.ConnectionError:
            return "[ERROR] Could not connect to Ollama at " + self.OLLAMA_URL
        except Exception as e:
            return f"[ERROR] Generation failed: {e}"

    # ------------------------------------------------------------------
    # 6. End-to-End Query (with latency instrumentation)
    # ------------------------------------------------------------------
    def query(self, question: str) -> dict:
        """Full pipeline: retrieve → generate, with timing."""
        # --- Retrieval ---
        t0 = time.perf_counter()
        hits = self.retrieve(question)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        # --- Generation ---
        t1 = time.perf_counter()
        answer = self.generate(question, hits)
        generation_ms = (time.perf_counter() - t1) * 1000

        e2e_ms = retrieval_ms + generation_ms

        return {
            "question": question,
            "answer": answer,
            "retrieval_latency_ms": round(retrieval_ms, 1),
            "generation_latency_ms": round(generation_ms, 1),
            "e2e_latency_ms": round(e2e_ms, 1),
            "context_length_chars": sum(len(c["text"]) for c in hits),
            "hits": hits,
        }


# ======================================================================
# METRIC COMPUTATION FUNCTIONS (identical to test_e2e.py)
# ======================================================================
from collections import Counter
from typing import List, Dict

def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())

def tokenize(text: str) -> List[str]:
    return normalize(text).split()

def compute_token_f1(prediction: str, reference: str) -> Dict[str, float]:
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
    return 1.0 if normalize(prediction) == normalize(reference) else 0.0

def compute_rouge_1(prediction: str, reference: str) -> Dict[str, float]:
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
    m, n = len(x), len(y)
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
    context_tokens = set(tokenize(context))
    truth_tokens = tokenize(ground_truth)
    if not truth_tokens:
        return 0.0
    found = sum(1 for t in truth_tokens if t in context_tokens)
    return found / len(truth_tokens)

def compute_context_precision(context: str, ground_truth: str) -> float:
    context_tokens = tokenize(context)
    truth_tokens = set(tokenize(ground_truth))
    if not context_tokens:
        return 0.0
    relevant = sum(1 for t in context_tokens if t in truth_tokens)
    return relevant / len(context_tokens)

def compute_keyword_hit_rate(context: str, keywords: list) -> float:
    if not keywords:
        return 1.0
    context_lower = context.lower()
    hits = sum(1 for kw in keywords if kw.lower() in context_lower)
    return hits / len(keywords)

def compute_faithfulness(answer: str, context: str) -> float:
    answer_tokens = tokenize(answer)
    context_tokens = set(tokenize(context))
    if not answer_tokens:
        return 0.0
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
        return 1.0
    grounded = sum(1 for t in content_tokens if t in context_tokens)
    return grounded / len(content_tokens)

def compute_answer_relevance(answer: str, question: str) -> float:
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


# ======================================================================
# GOLDEN DATASET (same as NS-DMN test_e2e.py)
# ======================================================================
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
    {
        "id": "Q6",
        "question": "What BLEU score does the Transformer achieve on the WMT 2014 English-to-German translation task?",
        "ground_truth": "The Transformer model achieves a 28.4 BLEU score on the WMT 2014 English-to-German translation task, improving over the existing best results by over 2 BLEU.",
        "expected_context_keywords": ["bleu", "28.4", "english", "german", "translation", "transformer"],
        "category": "metrics"
    },
    {
        "id": "Q7",
        "question": "What is the primary architecture of the Transformer model proposed in Attention Is All You Need?",
        "ground_truth": "The Transformer relies entirely on an attention mechanism, dispensing with recurrence and convolutions entirely, while still using an encoder-decoder structure.",
        "expected_context_keywords": ["attention", "dispensing", "recurrence", "convolutions", "mechanism"],
        "category": "architecture"
    },
    {
        "id": "Q8",
        "question": "Who is the author of 'Playing and Gaming: Reflections and Classifications'?",
        "ground_truth": "The author is Bo Kampmann Walther, an Associate Professor at the University of Southern Denmark.",
        "expected_context_keywords": ["bo", "kampmann", "walther", "professor", "denmark"],
        "category": "authorship"
    },
    {
        "id": "Q9",
        "question": "According to the gaming article, what is the brief definition of 'Play'?",
        "ground_truth": "Play is defined as an open-ended territory in which make-believe and world-building are crucial factors.",
        "expected_context_keywords": ["open-ended", "territory", "make-believe", "world-building"],
        "category": "definitions"
    },
    {
        "id": "Q10",
        "question": "What journal published the article 'Environmental Sustainability: A Definition for Environmental Professionals'?",
        "ground_truth": "The article was published in the Journal of Environmental Sustainability, Volume 1, Issue 1 in 2011.",
        "expected_context_keywords": ["journal", "environmental", "sustainability", "volume", "2011"],
        "category": "publishing"
    }
]


# ======================================================================
# MAIN — Full Benchmark with RAG Metrics
# ======================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("  NAIVE RAG BASELINE — Full Metrics Benchmark")
    print("=" * 80)

    # --- Locate test documents ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    TEST_DIR = os.path.join(SCRIPT_DIR, "test")

    if not os.path.exists(TEST_DIR):
        print(f"\n[FATAL] Test directory not found at: {TEST_DIR}")
        sys.exit(1)

    # --- Init & Ingest ---
    rag = LocalNaiveRAG()
    for filename in os.listdir(TEST_DIR):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(TEST_DIR, filename)
            rag.ingest_pdf(pdf_path)

    # --- Aggregation accumulators ---
    agg = {
        "context_recall": [], "context_precision": [], "keyword_hit_rate": [],
        "token_f1": [], "token_precision": [], "token_recall": [],
        "exact_match": [],
        "rouge1_f1": [], "rouge1_precision": [], "rouge1_recall": [],
        "rougel_f1": [], "rougel_precision": [], "rougel_recall": [],
        "faithfulness": [], "answer_relevance": [],
        "retrieval_latency_ms": [], "generation_latency_ms": [], "e2e_latency_ms": [],
        "context_length_chars": [],
    }

    print("\n" + "=" * 80)
    print("  PER-QUERY EVALUATION RESULTS")
    print("=" * 80)

    for item in GOLDEN_DATASET:
        qid = item["id"]
        question = item["question"]
        ground_truth = item["ground_truth"]
        keywords = item["expected_context_keywords"]

        print(f"\n{'─' * 80}")
        print(f"  {qid}: {question}")
        print(f"{'─' * 80}")

        # --- Retrieval ---
        e2e_start = time.perf_counter()
        retrieval_start = time.perf_counter()
        hits = rag.retrieve(question)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        # Build context (blind concatenation)
        context = "\n\n".join(h["text"] for h in hits)

        # Retrieval metrics
        ctx_recall = compute_context_recall(context, ground_truth)
        ctx_precision = compute_context_precision(context, ground_truth)
        kw_hit_rate = compute_keyword_hit_rate(context, keywords)

        agg["context_recall"].append(ctx_recall)
        agg["context_precision"].append(ctx_precision)
        agg["keyword_hit_rate"].append(kw_hit_rate)
        agg["retrieval_latency_ms"].append(retrieval_ms)
        agg["context_length_chars"].append(len(context))

        print(f"\n  RETRIEVAL METRICS:")
        print(f"    Context Length:     {len(context):,} chars")
        print(f"    Context Recall:     {ctx_recall:.4f}")
        print(f"    Context Precision:  {ctx_precision:.4f}")
        print(f"    Keyword Hit Rate:   {kw_hit_rate:.4f} ({int(kw_hit_rate * len(keywords))}/{len(keywords)} keywords)")
        print(f"    Retrieval Latency:  {retrieval_ms:.0f} ms")

        # --- Generation ---
        gen_start = time.perf_counter()
        answer = rag.generate(question, hits)
        generation_ms = (time.perf_counter() - gen_start) * 1000
        e2e_ms = (time.perf_counter() - e2e_start) * 1000

        # Generation metrics
        f1_scores = compute_token_f1(answer, ground_truth)
        em = compute_exact_match(answer, ground_truth)
        rouge1 = compute_rouge_1(answer, ground_truth)
        rougel = compute_rouge_l(answer, ground_truth)
        faithful = compute_faithfulness(answer, context)
        relevance = compute_answer_relevance(answer, question)

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
        agg["generation_latency_ms"].append(generation_ms)
        agg["e2e_latency_ms"].append(e2e_ms)

        print(f"\n  GENERATION METRICS:")
        print(f"    Answer Preview:     {answer[:120]}...")
        print(f"    Token F1:           {f1_scores['f1']:.4f}")
        print(f"    Token Precision:    {f1_scores['precision']:.4f}")
        print(f"    Token Recall:       {f1_scores['recall']:.4f}")
        print(f"    Exact Match:        {em:.1f}")
        print(f"    ROUGE-1 (F1):       {rouge1['f1']:.4f}")
        print(f"    ROUGE-L (F1):       {rougel['f1']:.4f}")
        print(f"    Faithfulness:       {faithful:.4f}")
        print(f"    Answer Relevance:   {relevance:.4f}")

        print(f"\n  LATENCY:")
        print(f"    Retrieval:          {retrieval_ms:.0f} ms")
        print(f"    Generation:         {generation_ms:.0f} ms")
        print(f"    End-to-End:         {e2e_ms:.0f} ms")

    # ==================================================================
    # AGGREGATE SUMMARY
    # ==================================================================
    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    print("\n\n" + "=" * 80)
    print("  AGGREGATE METRICS SUMMARY")
    print("=" * 80)

    print(f"\n  {'Metric':<35} {'Mean':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'─' * 65}")

    print(f"\n  {'RETRIEVAL':}")
    print(f"  {'Context Recall':<35} {mean(agg['context_recall']):>10.4f} {min(agg['context_recall']):>10.4f} {max(agg['context_recall']):>10.4f}")
    print(f"  {'Context Precision':<35} {mean(agg['context_precision']):>10.4f} {min(agg['context_precision']):>10.4f} {max(agg['context_precision']):>10.4f}")
    print(f"  {'Keyword Hit Rate':<35} {mean(agg['keyword_hit_rate']):>10.4f} {min(agg['keyword_hit_rate']):>10.4f} {max(agg['keyword_hit_rate']):>10.4f}")
    print(f"  {'Context Length (chars)':<35} {mean(agg['context_length_chars']):>10.0f} {min(agg['context_length_chars']):>10.0f} {max(agg['context_length_chars']):>10.0f}")

    print(f"\n  {'GENERATION':}")
    print(f"  {'Token F1':<35} {mean(agg['token_f1']):>10.4f} {min(agg['token_f1']):>10.4f} {max(agg['token_f1']):>10.4f}")
    print(f"  {'Token Precision':<35} {mean(agg['token_precision']):>10.4f} {min(agg['token_precision']):>10.4f} {max(agg['token_precision']):>10.4f}")
    print(f"  {'Token Recall':<35} {mean(agg['token_recall']):>10.4f} {min(agg['token_recall']):>10.4f} {max(agg['token_recall']):>10.4f}")
    print(f"  {'Exact Match':<35} {mean(agg['exact_match']):>10.4f} {min(agg['exact_match']):>10.4f} {max(agg['exact_match']):>10.4f}")
    print(f"  {'ROUGE-1 (F1)':<35} {mean(agg['rouge1_f1']):>10.4f} {min(agg['rouge1_f1']):>10.4f} {max(agg['rouge1_f1']):>10.4f}")
    print(f"  {'ROUGE-L (F1)':<35} {mean(agg['rougel_f1']):>10.4f} {min(agg['rougel_f1']):>10.4f} {max(agg['rougel_f1']):>10.4f}")
    print(f"  {'Faithfulness':<35} {mean(agg['faithfulness']):>10.4f} {min(agg['faithfulness']):>10.4f} {max(agg['faithfulness']):>10.4f}")
    print(f"  {'Answer Relevance':<35} {mean(agg['answer_relevance']):>10.4f} {min(agg['answer_relevance']):>10.4f} {max(agg['answer_relevance']):>10.4f}")

    print(f"\n  {'LATENCY (ms)':}")
    print(f"  {'Retrieval Latency':<35} {mean(agg['retrieval_latency_ms']):>10.0f} {min(agg['retrieval_latency_ms']):>10.0f} {max(agg['retrieval_latency_ms']):>10.0f}")
    print(f"  {'Generation Latency':<35} {mean(agg['generation_latency_ms']):>10.0f} {min(agg['generation_latency_ms']):>10.0f} {max(agg['generation_latency_ms']):>10.0f}")
    print(f"  {'End-to-End Latency':<35} {mean(agg['e2e_latency_ms']):>10.0f} {min(agg['e2e_latency_ms']):>10.0f} {max(agg['e2e_latency_ms']):>10.0f}")

    # ==================================================================
    # GRADE CARD
    # ==================================================================
    mean_ctx_recall = mean(agg['context_recall'])
    mean_kw_hit = mean(agg['keyword_hit_rate'])
    mean_f1 = mean(agg['token_f1'])
    mean_rouge1 = mean(agg['rouge1_f1'])
    mean_rougel = mean(agg['rougel_f1'])
    mean_faithful = mean(agg['faithfulness'])

    def grade(score, thresholds):
        labels = ['A', 'B', 'C', 'D', 'F']
        for i, t in enumerate(thresholds):
            if score >= t:
                return labels[i]
        return 'F'

    grades = {
        "Context Recall":   (mean_ctx_recall, grade(mean_ctx_recall, [0.8, 0.6, 0.4, 0.2])),
        "Keyword Hit Rate": (mean_kw_hit,     grade(mean_kw_hit,     [0.8, 0.6, 0.4, 0.2])),
        "Token F1":         (mean_f1,          grade(mean_f1,         [0.6, 0.4, 0.25, 0.1])),
        "ROUGE-1 F1":       (mean_rouge1,      grade(mean_rouge1,     [0.6, 0.4, 0.25, 0.1])),
        "ROUGE-L F1":       (mean_rougel,      grade(mean_rougel,     [0.5, 0.35, 0.2, 0.1])),
        "Faithfulness":     (mean_faithful,     grade(mean_faithful,   [0.8, 0.6, 0.4, 0.2])),
    }

    print(f"\n\n  {'=' * 65}")
    print(f"  PERFORMANCE GRADE CARD (Naive RAG Baseline)")
    print(f"  {'=' * 65}")
    print(f"\n  {'Metric':<30} {'Score':>8} {'Grade':>8}")
    print(f"  {'─' * 46}")
    for metric, (score, g) in grades.items():
        print(f"  {metric:<30} {score:>8.4f} {g:>8}")

    grade_values = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
    avg_grade_val = sum(grade_values[g] for _, (_, g) in grades.items()) / len(grades)
    overall = 'A' if avg_grade_val >= 3.5 else 'B' if avg_grade_val >= 2.5 else 'C' if avg_grade_val >= 1.5 else 'D' if avg_grade_val >= 0.5 else 'F'
    print(f"  {'─' * 46}")
    print(f"  {'OVERALL GRADE':<30} {'':>8} {overall:>8}")

    # ==================================================================
    # SYSTEM INFO
    # ==================================================================
    print(f"\n\n  {'=' * 65}")
    print(f"  SYSTEM CONFIGURATION")
    print(f"  {'=' * 65}")
    print(f"  Total chunks:     {rag.index.ntotal}")
    print(f"  Chunk size:       {rag.CHUNK_SIZE} chars, overlap: {rag.CHUNK_OVERLAP} chars")
    print(f"  FAISS index:      IndexFlatL2 (raw L2, no normalization)")
    print(f"  Retrieval:        Top-{rag.TOP_K}, no reranking")
    print(f"  Compression:      None (blind concatenation)")
    print(f"  Prompt:           Basic 'Answer using context' (no strict faithfulness)")

    print("\n" + "=" * 80)
    print("  Naive RAG Benchmark Complete.")
    print("=" * 80)
