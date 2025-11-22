#!/usr/bin/env python3
"""
Complete SOP Quick-Finder Pipeline Runner
Runs extract, chunk, and embed in sequence.
"""

import sys
from pathlib import Path

# Add scripts directory to path
base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir / 'scripts'))

from extract import DocumentExtractor
from chunk import SOPChunker
from embed import EmbeddingIndexer


def run_full_pipeline():
    """Run the complete indexing pipeline."""

    print("="*80)
    print("GSE SOP Quick-Finder - Full Pipeline")
    print("="*80)
    print()

    # Define directories
    raw_dir = base_dir / 'data' / 'raw'
    processed_dir = base_dir / 'data' / 'processed'
    index_dir = base_dir / 'data' / 'index'

    # Step 1: Extract
    print("STEP 1/3: Extracting text from documents...")
    print("-"*80)
    try:
        extractor = DocumentExtractor(raw_dir, processed_dir)
        results = extractor.extract_all()
        print(f"✓ Extraction complete: {len(results)} documents processed")
    except Exception as e:
        print(f"✗ Error during extraction: {e}")
        return False

    print()

    # Step 2: Chunk
    print("STEP 2/3: Chunking documents...")
    print("-"*80)
    try:
        chunker = SOPChunker(processed_dir, chunk_size=800, overlap=100)
        chunks = chunker.process_all()
        print(f"✓ Chunking complete: {len(chunks)} chunks created")
    except Exception as e:
        print(f"✗ Error during chunking: {e}")
        return False

    print()

    # Step 3: Embed & Index
    print("STEP 3/3: Building search index...")
    print("-"*80)
    print("Note: First run will download the embedding model (~80MB)")
    try:
        indexer = EmbeddingIndexer(processed_dir, index_dir)
        indexer.build_index()
        print(f"✓ Index build complete")
    except Exception as e:
        print(f"✗ Error during indexing: {e}")
        return False

    print()
    print("="*80)
    print("✓ PIPELINE COMPLETE!")
    print("="*80)
    print()
    print("Next steps:")
    print("  1. Run the search server: python app/server.py")
    print("  2. Open browser to: http://127.0.0.1:5000")
    print("  3. Or use CLI search: python scripts/search.py")
    print()

    return True


if __name__ == '__main__':
    success = run_full_pipeline()
    sys.exit(0 if success else 1)
