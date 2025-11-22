#!/bin/bash
# Complete setup and indexing pipeline for SOP Quick-Finder

set -e  # Exit on error

echo "========================================="
echo "GSE SOP Quick-Finder Setup Pipeline"
echo "========================================="
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Warning: Virtual environment not activated"
    echo "Run: source venv/bin/activate"
    exit 1
fi

# Check if SOPs exist
SOP_COUNT=$(find ../data/raw -type f \( -name "*.pdf" -o -name "*.docx" \) | wc -l)
if [ "$SOP_COUNT" -eq 0 ]; then
    echo "⚠️  No SOP files found in data/raw/"
    echo "Please add PDF or Word documents to data/raw/ first"
    exit 1
fi

echo "✅ Found $SOP_COUNT SOP document(s)"
echo ""

# Step 1: Extract
echo "Step 1/3: Extracting text from documents..."
python extract.py
if [ $? -ne 0 ]; then
    echo "❌ Extraction failed"
    exit 1
fi
echo "✅ Extraction complete"
echo ""

# Step 2: Chunk
echo "Step 2/3: Chunking documents..."
python chunk.py
if [ $? -ne 0 ]; then
    echo "❌ Chunking failed"
    exit 1
fi
echo "✅ Chunking complete"
echo ""

# Step 3: Embed
echo "Step 3/3: Building search index..."
echo "(First run will download embedding model ~80MB)"
python embed.py
if [ $? -ne 0 ]; then
    echo "❌ Indexing failed"
    exit 1
fi
echo "✅ Indexing complete"
echo ""

echo "========================================="
echo "✅ Setup Complete!"
echo "========================================="
echo ""
echo "To start searching:"
echo "  Web UI:  python ../app/server.py"
echo "  CLI:     python search.py"
echo ""
