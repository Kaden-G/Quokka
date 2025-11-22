#!/usr/bin/env python3
"""
SOP Search Script
Semantic search over indexed SOP chunks using FAISS.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


class SOPSearcher:
    """Semantic search over SOP documents."""

    def __init__(self, index_dir: str):
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
        model_name = self.config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
        self.model = SentenceTransformer(model_name)

        print(f"Loaded index with {len(self.metadata)} chunks")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for relevant SOP chunks.

        Args:
            query: Natural language query
            top_k: Number of results to return

        Returns:
            List of matching chunks with metadata
        """
        # Embed query
        query_embedding = self.model.encode([query], convert_to_numpy=True)

        # Search FAISS index
        distances, indices = self.index.search(
            query_embedding.astype('float32'),
            top_k
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

        return results

    def _distance_to_similarity(self, distance: float) -> float:
        """Convert L2 distance to similarity score (0-1)."""
        # For L2 distance, smaller is better
        # Convert to similarity where 1 is perfect match
        return max(0.0, 1.0 - (distance / 10.0))

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
