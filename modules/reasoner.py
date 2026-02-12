import json
import re
import logging
from typing import Tuple, Dict

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
            logger.warning("Ollama not found. Cognitive Router will default to vector search with pattern-based filters.")
    
    def _extract_filters_pattern_based(self, query: str) -> dict:
        """Fallback pattern-based filter extraction when LLM is unavailable."""
        filters = {}
        
        # Extract year patterns (2020-2099)
        year_match = re.search(r'\b(20\d{2})\b', query)
        if year_match:
            filters['date_range'] = year_match.group(1)
        
        # Extract filename patterns (.pdf, .txt, etc.)
        file_match = re.search(r'\b(\w+\.(pdf|txt|docx?))\b', query, re.IGNORECASE)
        if file_match:
            filters['source_file'] = file_match.group(1)
        
        return filters

    def analyze_query(self, user_query: str) -> Tuple[str, str, str, dict]:
        """
        Performs step-by-step reasoning on the query.
        Returns: (refined_query, route, thought_trace, filters)
        Route options: 'vector', 'graph', 'hybrid'
        Filters: dict with 'date_range' and 'source_file' keys
        """
        if not ollama:
            filters = self._extract_filters_pattern_based(user_query)
            return user_query, 'vector', "Ollama unavailable. Using pattern-based filters.", filters

        # Structured prompt for JSON output
        prompt = (
            f"You are a Cognitive Router for a RAG system. Your goal is to understand the user's intent.\n"
            f"User Query: \"{user_query}\"\n\n"
            "Task:\n"
            "1. IDENTITY GUARD: When the user asks about 'this system', 'the architecture', or 'system design', they are referring to the NS-DMN project described in Nlp_project.pdf. Do NOT use technical details from other research papers (like 'Attention Is All You Need') to describe the NS-DMN's internal layers.\n"
            "2. Think silently about what the user really wants, resolving pronouns like 'it' or 'that' if possible based on generic context.\n"
            "3. Reformulate the query into a clear, standalone search string.\n"
            "4. Decide the routing strategy:\n"
            "   - 'hybrid': **PREFERRED** for technical queries, definitions, explanations, or 'What is...' questions. "
            "Hybrid retrieval combines structured entity facts with rich descriptive context from source documents.\n"
            "   - 'graph': ONLY for pure relationship mapping (e.g., 'How does X relate to Y?', 'What connects A and B?'). "
            "Use this when the user explicitly asks about connections or relationships between specific entities.\n"
            "   - 'vector': For opinions, reviews, abstract concepts, sentiment analysis, or general exploratory questions.\n\n"
            "**IMPORTANT**: If the query asks 'What is [X]?', seeks a definition, requests technical specifications, "
            "or asks for a detailed explanation of a concept or system, you MUST choose 'hybrid'.\n\n"
            "5. Extract metadata filters from the query (if any):\n"
            "   - date_range: Year or date mentioned (e.g., '2025', '2024-01', null if not mentioned)\n"
            "   - source_file: Filename mentioned (e.g., 'report.pdf', 'testing.pdf', null if not mentioned)\n\n"
            "Output strictly in valid JSON format:\n"
            "{\n"
            "  \"thought\": \"<your inner monologue>\",\n"
            "  \"reformulated_query\": \"<refined query>\",\n"
            "  \"route\": \"<vector|graph|hybrid>\",\n"
            "  \"filters\": {\n"
            "    \"date_range\": \"<year or null>\",\n"
            "    \"source_file\": \"<filename or null>\"\n"
            "  }\n"
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
            filters = data.get("filters", {})
            
            # Validate and clean filters
            if filters:
                # Convert null strings to None
                if filters.get('date_range') in ['null', None, '']:
                    filters['date_range'] = None
                if filters.get('source_file') in ['null', None, '']:
                    filters['source_file'] = None
            
            if route not in ['vector', 'graph', 'hybrid']:
                route = 'vector'
                
            return refined_query, route, thought, filters

        except Exception as e:
            logger.error(f"Reasoning failed: {e}")
            # Fallback with pattern-based filter extraction
            filters = self._extract_filters_pattern_based(user_query)
            return user_query, 'vector', f"Reasoning Error: {e}", filters
