#!/usr/bin/env python3
"""
SOP Document Extraction Script
Extracts text from PDF and Word documents for indexing.
"""

import os
import json
from pathlib import Path
from typing import Dict, List
import pypdf
from docx import Document


class DocumentExtractor:
    """Extract text from SOPs with metadata preservation."""

    def __init__(self, raw_dir: str, processed_dir: str):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def extract_pdf(self, pdf_path: Path) -> List[Dict]:
        """Extract text from PDF with page numbers."""
        chunks = []
        try:
            with open(pdf_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                for page_num, page in enumerate(reader.pages, 1):
                    text = page.extract_text()
                    if text.strip():
                        chunks.append({
                            'doc_name': pdf_path.stem,
                            'page': page_num,
                            'text': text.strip(),
                            'doc_type': 'pdf'
                        })
        except Exception as e:
            print(f"Error extracting {pdf_path.name}: {e}")
        return chunks

    def extract_docx(self, docx_path: Path) -> List[Dict]:
        """Extract text from Word documents."""
        chunks = []
        try:
            doc = Document(docx_path)
            full_text = []
            for para in doc.paragraphs:
                if para.text.strip():
                    full_text.append(para.text.strip())

            # Word docs don't have explicit pages, so we chunk by paragraphs
            chunks.append({
                'doc_name': docx_path.stem,
                'page': 1,
                'text': '\n'.join(full_text),
                'doc_type': 'docx'
            })
        except Exception as e:
            print(f"Error extracting {docx_path.name}: {e}")
        return chunks

    def extract_txt(self, txt_path: Path) -> List[Dict]:
        """Extract text from plain text files."""
        chunks = []
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read()

            if text.strip():
                chunks.append({
                    'doc_name': txt_path.stem,
                    'page': 1,
                    'text': text.strip(),
                    'doc_type': 'txt'
                })
        except Exception as e:
            print(f"Error extracting {txt_path.name}: {e}")
        return chunks

    def extract_all(self) -> Dict[str, List[Dict]]:
        """Process all documents in raw directory."""
        all_extracts = {}

        # Process PDFs
        for pdf_file in self.raw_dir.glob('*.pdf'):
            print(f"Extracting: {pdf_file.name}")
            extracts = self.extract_pdf(pdf_file)
            all_extracts[pdf_file.stem] = extracts

            # Save individual file
            output_file = self.processed_dir / f"{pdf_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(extracts, f, indent=2, ensure_ascii=False)

        # Process Word docs
        for docx_file in self.raw_dir.glob('*.docx'):
            if docx_file.name.startswith('~$'):  # Skip temp files
                continue
            print(f"Extracting: {docx_file.name}")
            extracts = self.extract_docx(docx_file)
            all_extracts[docx_file.stem] = extracts

            # Save individual file
            output_file = self.processed_dir / f"{docx_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(extracts, f, indent=2, ensure_ascii=False)

        # Process text files
        for txt_file in self.raw_dir.glob('*.txt'):
            print(f"Extracting: {txt_file.name}")
            extracts = self.extract_txt(txt_file)
            all_extracts[txt_file.stem] = extracts

            # Save individual file
            output_file = self.processed_dir / f"{txt_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(extracts, f, indent=2, ensure_ascii=False)

        # Save combined manifest
        manifest_file = self.processed_dir / "manifest.json"
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump({
                'total_docs': len(all_extracts),
                'documents': list(all_extracts.keys())
            }, f, indent=2)

        print(f"\nExtraction complete: {len(all_extracts)} documents processed")
        return all_extracts


def main():
    """Run extraction on all SOPs."""
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / 'data' / 'raw'
    processed_dir = base_dir / 'data' / 'processed'

    extractor = DocumentExtractor(raw_dir, processed_dir)
    results = extractor.extract_all()

    print(f"\nResults saved to: {processed_dir}")
    print(f"Total documents: {len(results)}")


if __name__ == '__main__':
    main()
