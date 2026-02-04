import json
import re
import logging
from typing import Tuple

try:
    import ollama
except ImportError:
    ollama = None

from modules.utils import logger

class CognitiveRouter:
    """
    Phase B: IM-RAG (Inner Monologue Retrieval Augmented Generation)
    Replaces simple keyword routing with reasoning.
    """
    def __init__(self, model_name: str = "llama3"):
        self.model_name = model_name
        if not ollama:
            logger.warning("Ollama not found. Cognitive Router will default to vector search.")

    def analyze_query(self, user_query: str) -> Tuple[str, str, str]:
        """
        Performs step-by-step reasoning on the query.
        Returns: (refined_query, route, thought_trace)
        Route options: 'vector', 'graph', 'hybrid'
        """
        if not ollama:
            return user_query, 'vector', "Ollama unavailable."

        # Structured prompt for JSON output
        prompt = (
            f"You are a Cognitive Router for a RAG system. Your goal is to understand the user's intent.\n"
            f"User Query: \"{user_query}\"\n\n"
            "Task:\n"
            "1. Think silently about what the user really wants, resolving pronouns like 'it' or 'that' if possible based on generic context.\n"
            "2. Reformulate the query into a clear, standalone search string.\n"
            "3. Decide the routing strategy:\n"
            "   - 'hybrid': **PREFERRED** for technical queries, definitions, explanations, or 'What is...' questions. "
            "Hybrid retrieval combines structured entity facts with rich descriptive context from source documents.\n"
            "   - 'graph': ONLY for pure relationship mapping (e.g., 'How does X relate to Y?', 'What connects A and B?'). "
            "Use this when the user explicitly asks about connections or relationships between specific entities.\n"
            "   - 'vector': For opinions, reviews, abstract concepts, sentiment analysis, or general exploratory questions.\n\n"
            "**IMPORTANT**: If the query asks 'What is [X]?', seeks a definition, requests technical specifications, "
            "or asks for a detailed explanation of a concept or system, you MUST choose 'hybrid'.\n\n"
            "Output strictly in valid JSON format:\n"
            "{\n"
            "  \"thought\": \"<your inner monologue>\",\n"
            "  \"reformulated_query\": \"<refined query>\",\n"
            "  \"route\": \"<vector|graph|hybrid>\"\n"
            "}"
        )

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                format='json'  # Force JSON mode
            )
            content = response['message']['content']
            
            # Simple validation & parsing
            data = json.loads(content)
            
            thought = data.get("thought", "No thought provided.")
            refined_query = data.get("reformulated_query", user_query)
            route = data.get("route", "vector").lower()
            
            if route not in ['vector', 'graph', 'hybrid']:
                route = 'vector'
                
            return refined_query, route, thought

        except Exception as e:
            logger.error(f"Reasoning failed: {e}")
            # Fallback
            return user_query, 'vector', f"Reasoning Error: {e}"
