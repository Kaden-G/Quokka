#!/usr/bin/env python3
"""
SOP Chunking Script
Splits extracted text into semantic chunks while preserving metadata.
"""

import json
import re
from pathlib import Path
from typing import List, Dict


class SOPChunker:
    """Split SOP text into retrievable chunks."""

    def __init__(
        self,
        processed_dir: str,
        chunk_size: int = 800,
        overlap: int = 100
    ):
        self.processed_dir = Path(processed_dir)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def detect_section_headers(self, text: str) -> List[tuple]:
        """Detect section headers in text. Returns (char_offset, header_text) tuples."""
        # Common SOP header patterns
        patterns = [
            r'^(\d+\.\d*\.?\d*)\s+([A-Z][^\n]+)',  # 1.2.3 HEADER
            r'^([A-Z][A-Z\s]{3,}[A-Z])$',          # ALL CAPS HEADER
            r'^(STEP\s+\d+[:\.])\s*(.+)',          # STEP 1: Description
            r'^(Procedure\s+\d+[:\.])\s*(.+)',     # Procedure 1: Description
        ]

        headers = []
        char_offset = 0
        for line in text.split('\n'):
            for pattern in patterns:
                match = re.match(pattern, line.strip())
                if match:
                    headers.append((char_offset, line.strip()))
                    break
            char_offset += len(line) + 1  # +1 for the newline
        return headers

    def chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        """Split text into overlapping chunks."""
        chunks = []
        words = text.split()

        # Detect sections for better chunking (returns char offsets)
        sections = self.detect_section_headers(text)

        # Build word-to-character offset map
        word_char_offsets = []
        pos = 0
        for word in words:
            idx = text.find(word, pos)
            word_char_offsets.append(idx)
            pos = idx + len(word)

        # Calculate word-based chunk size
        words_per_chunk = self.chunk_size // 5  # Rough estimate
        overlap_words = self.overlap // 5

        for i in range(0, len(words), words_per_chunk - overlap_words):
            chunk_words = words[i:i + words_per_chunk]
            chunk_text = ' '.join(chunk_words)

            # Compute actual character offsets
            char_start = word_char_offsets[i]
            last_word_idx = min(i + len(chunk_words) - 1, len(words) - 1)
            char_end = word_char_offsets[last_word_idx] + len(words[last_word_idx])

            # Find closest section header using character offsets
            section_header = "Unknown Section"
            for sec_offset, sec_name in sections:
                if sec_offset <= char_start:
                    section_header = sec_name

            chunk = {
                'chunk_id': f"{metadata['doc_name']}_p{metadata['page']}_c{len(chunks)}",
                'doc_name': metadata['doc_name'],
                'page': metadata['page'],
                'section': section_header,
                'text': chunk_text,
                'char_start': char_start,
                'char_end': char_end
            }
            chunks.append(chunk)

            # Stop if we've processed all words
            if i + words_per_chunk >= len(words):
                break

        return chunks

    def process_all(self) -> List[Dict]:
        """Process all extracted documents into chunks."""
        all_chunks = []

        # Process each JSON file in processed directory
        for json_file in self.processed_dir.glob('*.json'):
            if json_file.name == 'manifest.json':
                continue

            print(f"Chunking: {json_file.name}")

            with open(json_file, 'r', encoding='utf-8') as f:
                pages = json.load(f)

            for page in pages:
                chunks = self.chunk_text(page['text'], page)
                all_chunks.extend(chunks)

        # Save all chunks
        chunks_file = self.processed_dir / 'chunks.json'
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)

        print(f"\nChunking complete: {len(all_chunks)} chunks created")
        print(f"Saved to: {chunks_file}")

        return all_chunks


def main():
    """Run chunking on all processed documents."""
    base_dir = Path(__file__).parent.parent
    processed_dir = base_dir / 'data' / 'processed'

    chunker = SOPChunker(processed_dir, chunk_size=800, overlap=100)
    chunks = chunker.process_all()

    print(f"\nTotal chunks: {len(chunks)}")


if __name__ == '__main__':
    main()
