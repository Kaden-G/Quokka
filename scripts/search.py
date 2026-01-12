#!/usr/bin/env python3
"""
SOP Search Script
Semantic search over indexed SOP chunks using FAISS.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class SOPSearcher:
    """Semantic search over SOP documents."""

    def __init__(
        self,
        index_dir: str,
        cache_size: int = 100,
        use_reranker: bool = True,
        ollama_model: str = 'llama3.1:8b',
        use_ollama: bool = True
    ):
        self.index_dir = Path(index_dir)

        # Load config
        config_file = self.index_dir / 'config.json'
        with open(config_file, 'r') as f:
            self.config = json.load(f)

        # Load FAISS index
        index_file = self.index_dir / 'faiss.index'
        self.index = faiss.read_index(str(index_file))

        # Load metadata
        metadata_file = self.index_dir / 'metadata.pkl'
        with open(metadata_file, 'rb') as f:
            self.metadata = pickle.load(f)

        # Load embedding model
        model_name = self.config.get('model_name', 'BAAI/bge-small-en-v1.5')
        self.model = SentenceTransformer(model_name)

        # Load cross-encoder for re-ranking (optional but recommended)
        self.use_reranker = use_reranker
        if self.use_reranker:
            print("Loading cross-encoder for re-ranking...")
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        else:
            self.reranker = None

        # Ollama configuration for answer generation
        self.use_ollama = use_ollama and OLLAMA_AVAILABLE
        self.ollama_model = ollama_model
        if self.use_ollama:
            print(f"Ollama integration enabled (model: {ollama_model})")
        elif use_ollama and not OLLAMA_AVAILABLE:
            print("Warning: Ollama requested but not available. Install with: pip install ollama")

        # Initialize query cache for faster repeated queries
        self.query_cache = {}
        self.cache_size = cache_size

        print(f"Loaded index with {len(self.metadata)} chunks")

    def search(self, query: str, top_k: int = 5, rerank: bool = True) -> List[Dict]:
        """
        Search for relevant SOP chunks.

        Args:
            query: Natural language query
            top_k: Number of results to return
            rerank: Whether to use cross-encoder re-ranking (default: True)

        Returns:
            List of matching chunks with metadata
        """
        # Check cache first
        cache_key = f"{query}|{top_k}|{rerank}"
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]

        # Embed query
        query_embedding = self.model.encode([query], convert_to_numpy=True)

        # Normalize query embedding for cosine similarity
        query_embedding = query_embedding.astype('float32')
        faiss.normalize_L2(query_embedding)

        # Search FAISS index (get more results for re-ranking)
        retrieve_k = top_k * 3 if (rerank and self.use_reranker) else top_k
        distances, indices = self.index.search(
            query_embedding,
            retrieve_k
        )

        # Prepare results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for missing results
                continue

            chunk = self.metadata[idx]
            result = {
                'rank': i + 1,
                'score': float(distance),
                'similarity': self._distance_to_similarity(float(distance)),
                'chunk_id': chunk['chunk_id'],
                'doc_name': chunk['doc_name'],
                'page': chunk['page'],
                'section': chunk['section'],
                'text': chunk['text']
            }
            results.append(result)

        # Re-rank with cross-encoder if enabled
        if rerank and self.use_reranker and len(results) > 0:
            results = self._rerank_results(query, results)[:top_k]
        else:
            results = results[:top_k]

        # Update ranks after re-ranking
        for i, result in enumerate(results):
            result['rank'] = i + 1

        # Add to cache (with simple LRU eviction)
        if len(self.query_cache) >= self.cache_size:
            # Remove oldest entry
            self.query_cache.pop(next(iter(self.query_cache)))
        self.query_cache[cache_key] = results

        return results

    def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Re-rank results using cross-encoder for better relevance.

        Args:
            query: User query
            results: Initial retrieval results

        Returns:
            Re-ranked results
        """
        # Prepare pairs for cross-encoder
        pairs = [(query, result['text']) for result in results]

        # Get cross-encoder scores
        rerank_scores = self.reranker.predict(pairs)

        # Add rerank scores to results
        for result, score in zip(results, rerank_scores):
            result['rerank_score'] = float(score)

        # Sort by rerank score (descending)
        reranked = sorted(results, key=lambda x: x['rerank_score'], reverse=True)

        return reranked

    def _distance_to_similarity(self, score: float) -> float:
        """Convert cosine similarity score (0-1)."""
        # For cosine similarity (IndexFlatIP), score is already similarity (0-1)
        # Higher is better, 1 is perfect match
        return max(0.0, min(1.0, score))

    def search_by_document(self, query: str, doc_name: str, top_k: int = 5) -> List[Dict]:
        """Search within a specific document."""
        # Get all results
        results = self.search(query, top_k=50)

        # Filter by document
        filtered = [r for r in results if r['doc_name'] == doc_name]

        return filtered[:top_k]

    def get_document_list(self) -> List[str]:
        """Get list of all indexed documents."""
        docs = set(chunk['doc_name'] for chunk in self.metadata)
        return sorted(list(docs))

    def get_stats(self) -> Dict:
        """Get index statistics."""
        docs = self.get_document_list()
        return {
            'total_chunks': len(self.metadata),
            'total_documents': len(docs),
            'documents': docs,
            'embedding_dimension': self.config['embedding_dim'],
            'index_type': self.config['index_type']
        }

    def generate_answer(
        self,
        query: str,
        results: List[Dict],
        max_context_chunks: int = 5
    ) -> Optional[str]:
        """
        Generate an answer using Ollama based on retrieved chunks.

        Args:
            query: User question
            results: Search results (chunks)
            max_context_chunks: Maximum number of chunks to use as context

        Returns:
            Generated answer or None if Ollama is not available
        """
        if not self.use_ollama:
            return None

        # Prepare context from top chunks
        context_chunks = results[:max_context_chunks]
        context = "\n\n".join([
            f"[Source: {chunk['doc_name']}, Page {chunk['page']}]\n{chunk['text']}"
            for chunk in context_chunks
        ])

        # Create prompt for Ollama
        system_prompt = """You are a helpful assistant specialized in Standard Operating Procedures (SOPs).
Your role is to answer questions based ONLY on the provided context from SOP documents.

Guidelines:
- Provide clear, accurate answers based on the context
- Cite specific sources when possible (e.g., "According to page X...")
- If the answer is not in the context, say so clearly
- Be concise but complete
- Use bullet points or numbered lists when appropriate
- Highlight any safety warnings or important notes"""

        user_prompt = f"""Context from SOP documents:
{context}

Question: {query}

Please provide a clear answer based on the context above."""

        try:
            # Call Ollama
            response = ollama.chat(
                model=self.ollama_model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                options={
                    'temperature': 0.3,  # Lower temperature for factual responses
                    'num_predict': 500   # Max tokens for answer
                }
            )

            return response['message']['content']

        except Exception as e:
            print(f"Error generating answer with Ollama: {e}")
            return None

    def search_and_answer(
        self,
        query: str,
        top_k: int = 5,
        rerank: bool = True
    ) -> Dict:
        """
        Complete RAG pipeline: search + answer generation.

        Args:
            query: User question
            top_k: Number of chunks to retrieve
            rerank: Whether to use re-ranking

        Returns:
            Dictionary with results and generated answer
        """
        # Search for relevant chunks
        results = self.search(query, top_k=top_k, rerank=rerank)

        # Generate answer if Ollama is available
        answer = None
        if self.use_ollama and len(results) > 0:
            answer = self.generate_answer(query, results)

        return {
            'query': query,
            'results': results,
            'answer': answer,
            'count': len(results)
        }


def main():
    """Interactive search demo."""
    base_dir = Path(__file__).parent.parent
    index_dir = base_dir / 'data' / 'index'

    searcher = SOPSearcher(index_dir)

    # Show stats
    stats = searcher.get_stats()
    print("=== SOP Quick-Finder Ready ===")
    print(f"Indexed documents: {stats['total_documents']}")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"\nDocuments:")
    for doc in stats['documents']:
        print(f"  - {doc}")
    print("\n")

    # Interactive search
    while True:
        query = input("Enter search query (or 'quit' to exit): ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            break

        if not query:
            continue

        results = searcher.search(query, top_k=5)

        print(f"\n{'='*80}")
        print(f"Results for: '{query}'")
        print(f"{'='*80}\n")

        if not results:
            print("No results found.")
            continue

        for result in results:
            print(f"[{result['rank']}] {result['doc_name']} (Page {result['page']})")
            print(f"    Section: {result['section']}")
            print(f"    Similarity: {result['similarity']:.2%}")
            print(f"    Text: {result['text'][:200]}...")
            print()


if __name__ == '__main__':
    main()
