import sys
import queue
import signal
import time
import threading
import numpy as np

try:
    import ollama
except ImportError:
    print("Error: 'ollama' library not found. Please install requirements.")
    sys.exit(1)

from config import DATA_DIR, LOG_FILE
from modules.utils import logger, recover_previous_state, setup_logger
from modules.memory_store import SharedMemoryManager
from modules.brain import NeuralBrain
from modules.dreamer import MemoryDreamer
from modules.ingestor import WindBellIngestor

stop_event = threading.Event()
dreamer_thread = None
stm_queue = None

def signal_handler(sig, frame):
    print("\n\n[System] Shutdown Signal Received using Ctrl+C. Initiating Graceful Exit...")
    logger.info("Shutdown Signal Received.")
    stop_event.set()
    
    if stm_queue:
        stm_queue.put("POISON_PILL")
    
    if dreamer_thread and dreamer_thread.is_alive():
        print("[System] Waiting for Dreamer to save memory...")
        dreamer_thread.join(timeout=10.0)
        
    print("[System] Shutdown Complete. Goodbye.")
    sys.exit(0)

def main():
    global dreamer_thread, stm_queue
    signal.signal(signal.SIGINT, signal_handler)
    
    print("=== Neuro-Symbolic Dynamic Memory Network (NS-DMN) ===")
    print("[Init] Checking file system state...")
    
    recover_previous_state(DATA_DIR)
    
    print("[Init] Loading Shared Memory (Graph + Vector Store)...")
    memory_manager = SharedMemoryManager()
    
    stm_queue = queue.Queue()
    
    print("[Init] Waking up the Dreamer...")
    dreamer_thread = MemoryDreamer(memory_manager, stm_queue)
    dreamer_thread.start()
    
    print("[Init] Initializing Neural Brain (CPU Cortex)...")
    brain = NeuralBrain(memory_manager, stm_queue)
    
    print("[Init] Initializing Ingestion Engine (Wind-Bell)...")
    ingestor = WindBellIngestor(memory_manager, brain.encoder)
    
    print("[Init] Connecting to Ollama (GPU Cortex)...")
    LLM_MODEL = "llama3" 
    
    print("\nSystem Online. Type 'exit' or 'quit' to stop.")
    print("Commands: /ingest <path_to_pdf>")
    
    while not stop_event.is_set():
        try:
            user_input = input("\n>>> User: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ["exit", "quit", "bye"]:
                raise KeyboardInterrupt
            
            # --- Ingestion Command ---
            if user_input.startswith("/ingest"):
                parts = user_input.split(" ", 1)
                if len(parts) < 2:
                    print("[System] Usage: /ingest <path_to_file>")
                    continue
                file_path = parts[1].strip().strip('"')
                print(f"[System] Ingesting: {file_path}")
                ingestor.ingest_document(file_path)
                continue
            
            start_t = time.time()
            final_prompt_context = brain.process_query(user_input)
            
            if final_prompt_context.startswith("Error"):
                print(f"[System] {final_prompt_context}")
                continue
                
            messages = [
                {"role": "system", "content": "You are a helpful Research Assistant with access to a long-term memory graph. Use the provided Context to answer the user."},
                {"role": "user", "content": f"Context:\n{final_prompt_context}\n\nQuestion: {user_input}"}
            ]
            
            print(f"[System] Thinking... (Context Size: {len(final_prompt_context)} chars)")
            
            stream = ollama.chat(
                model=LLM_MODEL,
                messages=messages,
                stream=True
            )
            
            print(">>> Assistant: ", end="", flush=True)
            for chunk in stream:
                content = chunk['message']['content']
                print(content, end="", flush=True)
            print()
            
            logger.info(f"Interaction Complete. Duration: {time.time() - start_t:.2f}s")
            
        except KeyboardInterrupt:
            signal_handler(signal.SIGINT, None)
        except Exception as e:
            logger.error(f"Main Loop Error: {e}")
            print(f"\n[Error] {e}")

if __name__ == "__main__":
    main()
