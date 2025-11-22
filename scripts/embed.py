#!/usr/bin/env python3
"""
SOP Embedding Script
Creates vector embeddings and builds FAISS index for semantic search.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


class EmbeddingIndexer:
    """Create and manage vector embeddings for SOP chunks."""

    def __init__(
        self,
        processed_dir: str,
        index_dir: str,
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'
    ):
        self.processed_dir = Path(processed_dir)
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    def load_chunks(self) -> List[Dict]:
        """Load processed chunks from JSON."""
        chunks_file = self.processed_dir / 'chunks.json'

        if not chunks_file.exists():
            raise FileNotFoundError(
                f"Chunks file not found: {chunks_file}\n"
                "Run chunk.py first to create chunks."
            )

        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        print(f"Loaded {len(chunks)} chunks")
        return chunks

    def create_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """Generate embeddings for all chunks."""
        print("Generating embeddings...")

        texts = [chunk['text'] for chunk in chunks]

        # Encode in batches for efficiency
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )

        print(f"Embeddings shape: {embeddings.shape}")
        return embeddings

    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build FAISS index for fast similarity search."""
        print("Building FAISS index...")

        dimension = embeddings.shape[1]

        # Use simple flat index (best for < 1M vectors)
        # For larger datasets, consider IndexIVFFlat
        index = faiss.IndexFlatL2(dimension)

        # Add vectors to index
        index.add(embeddings.astype('float32'))

        print(f"Index built with {index.ntotal} vectors")
        return index

    def save_index(
        self,
        index: faiss.Index,
        chunks: List[Dict],
        embeddings: np.ndarray
    ):
        """Save FAISS index and metadata."""
        # Save FAISS index
        index_file = self.index_dir / 'faiss.index'
        faiss.write_index(index, str(index_file))
        print(f"FAISS index saved: {index_file}")

        # Save chunk metadata (without text to save space)
        metadata = [
            {
                'chunk_id': c['chunk_id'],
                'doc_name': c['doc_name'],
                'page': c['page'],
                'section': c['section'],
                'text': c['text']  # Keep text for display
            }
            for c in chunks
        ]

        metadata_file = self.index_dir / 'metadata.pkl'
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"Metadata saved: {metadata_file}")

        # Save embeddings for future use
        embeddings_file = self.index_dir / 'embeddings.npy'
        np.save(embeddings_file, embeddings)
        print(f"Embeddings saved: {embeddings_file}")

        # Save index config
        config = {
            'num_chunks': len(chunks),
            'embedding_dim': embeddings.shape[1],
            'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
            'index_type': 'IndexFlatL2'
        }

        config_file = self.index_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Config saved: {config_file}")

    def build_index(self):
        """Full pipeline: load chunks, embed, build index."""
        # Load chunks
        chunks = self.load_chunks()

        # Create embeddings
        embeddings = self.create_embeddings(chunks)

        # Build FAISS index
        index = self.build_faiss_index(embeddings)

        # Save everything
        self.save_index(index, chunks, embeddings)

        print("\n=== Index Build Complete ===")
        print(f"Total chunks indexed: {len(chunks)}")
        print(f"Index directory: {self.index_dir}")


def main():
    """Build the embedding index."""
    base_dir = Path(__file__).parent.parent
    processed_dir = base_dir / 'data' / 'processed'
    index_dir = base_dir / 'data' / 'index'

    indexer = EmbeddingIndexer(processed_dir, index_dir)
    indexer.build_index()


if __name__ == '__main__':
    main()
