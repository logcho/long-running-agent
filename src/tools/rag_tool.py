import os
import json
import numpy as np
from src.ollama_client import OllamaClient
import config

class LocalRAG:
    def __init__(self):
        self.kb_dir = config.KNOWLEDGE_BASE_DIR
        self.cache_path = os.path.join(self.kb_dir, ".embeddings_cache.json")
        self.client = OllamaClient()
        self.cache = {}
        self.load_cache()

    def load_cache(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception as e:
                print(f"Failed to load RAG cache: {e}")
                self.cache = {}

    def save_cache(self):
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save RAG cache: {e}")

    def chunk_text(self, text, chunk_size=800, overlap=100):
        chunks = []
        words = text.split()
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1 # +1 for space
            if current_size >= chunk_size:
                chunks.append(" ".join(current_chunk))
                # Retain overlap words
                overlap_words = current_chunk[-overlap//10:] if overlap > 0 else []
                current_chunk = overlap_words
                current_size = sum(len(w) + 1 for w in current_chunk)
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def reindex(self):
        """
        Scans kb_dir, reads all text files, chunks them, and generates embeddings.
        """
        print("Reindexing RAG Knowledge Base...")
        if not os.path.exists(self.kb_dir):
            os.makedirs(self.kb_dir, exist_ok=True)
            
        new_cache = {}
        files = [f for f in os.listdir(self.kb_dir) if f.endswith((".txt", ".md", ".json")) and not f.startswith(".")]
        
        for filename in files:
            path = os.path.join(self.kb_dir, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
                
            chunks = self.chunk_text(content)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{filename}#chunk{i}"
                
                # If chunk is unchanged and already cached, keep it
                if chunk_id in self.cache and self.cache[chunk_id]["text"] == chunk:
                    new_cache[chunk_id] = self.cache[chunk_id]
                else:
                    # Generate embedding
                    emb = self.client.get_embedding(chunk)
                    new_cache[chunk_id] = {
                        "filename": filename,
                        "text": chunk,
                        "embedding": emb
                    }
                    
        self.cache = new_cache
        self.save_cache()
        print(f"Reindexed {len(self.cache)} chunks total.")

    def search(self, query, top_k=3):
        """
        Searches the knowledge base for chunks similar to the query.
        """
        if not self.cache:
            self.reindex()
            if not self.cache:
                return "Knowledge base is empty. Add documents first."

        query_emb = self.client.get_embedding(query)
        
        # If we failed to get embeddings (e.g. model mismatch or no embedding endpoint), fallback to keyword search
        if not query_emb or any(not item.get("embedding") for item in self.cache.values()):
            print("Embeddings unavailable. Falling back to keyword search...")
            return self._keyword_search(query, top_k)

        results = []
        q_vec = np.array(query_emb)
        q_norm = np.linalg.norm(q_vec)
        
        if q_norm == 0:
            return self._keyword_search(query, top_k)

        for chunk_id, info in self.cache.items():
            emb = info.get("embedding")
            if not emb:
                continue
            c_vec = np.array(emb)
            c_norm = np.linalg.norm(c_vec)
            if c_norm == 0:
                continue
                
            similarity = np.dot(q_vec, c_vec) / (q_norm * c_norm)
            results.append((similarity, info))

        # Sort by similarity desc
        results.sort(key=lambda x: x[0], reverse=True)
        
        output = []
        for i, (sim, info) in enumerate(results[:top_k]):
            output.append(f"Result {i+1} (Source: {info['filename']}, Similarity: {sim:.3f}):\n{info['text']}\n")
            
        return "\n---\n".join(output)

    def _keyword_search(self, query, top_k=3):
        """
        Simple keyword frequency search fallback.
        """
        query_words = set(query.lower().split())
        results = []
        for chunk_id, info in self.cache.items():
            text = info["text"].lower()
            score = sum(text.count(word) for word in query_words)
            if score > 0:
                results.append((score, info))
                
        results.sort(key=lambda x: x[0], reverse=True)
        if not results:
            return "No matching documents found in RAG database."
            
        output = []
        for i, (score, info) in enumerate(results[:top_k]):
            output.append(f"Result {i+1} (Source: {info['filename']}, Keyword Score: {score}):\n{info['text']}\n")
            
        return "\n---\n".join(output)

# Initialize global instance
rag_system = LocalRAG()

# Define LangChain/LangGraph Tools
class RAGSearchTool:
    name = "rag_search"
    description = "Search the local knowledge base / RAG store for relevant documents matching the query."
    
    def __call__(self, query: str, top_k: int = 3) -> str:
        return rag_system.search(query, top_k)

class RAGAddDocumentTool:
    name = "rag_add_document"
    description = "Add a new document to the knowledge base, saving it to a file and updating the RAG index. filename should end with .txt or .md."
    
    def __call__(self, filename: str, content: str) -> str:
        if not filename.endswith((".txt", ".md", ".json")):
            filename += ".txt"
        path = os.path.join(config.KNOWLEDGE_BASE_DIR, filename)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            rag_system.reindex()
            return f"Successfully added {filename} to knowledge base and reindexed."
        except Exception as e:
            return f"Failed to add document: {e}"
