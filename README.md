# GSE SOP Quick-Finder

**100% Offline, Secure SOP Retrieval System for Ground Systems Engineers**

A lightweight, fully-local semantic search tool for finding procedures in Standard Operating Procedures (SOPs) using plain-English queries. Built for classified/isolated environments with no internet connectivity.

---

## Features

✅ **Fully Offline** - No cloud services, no external APIs
✅ **Semantic Search** - Natural language queries (e.g., "How do I start the cooling system?")
✅ **Multi-Format Support** - PDF and Word documents
✅ **Exact Retrieval** - Returns precise text snippets, never generates new content
✅ **Fast & Lightweight** - Uses local embedding model and FAISS vector search
✅ **Secure** - All processing happens on your machine
✅ **Easy to Deploy** - Simple Python setup, no complex infrastructure

---

## Quick Start

### 1. Installation

```bash
cd sop-quickfinder
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add Your SOP Documents

Place your PDF or Word SOP files in the `data/raw/` directory:

```bash
cp /path/to/your/SOPs/*.pdf data/raw/
```

### 3. Build the Index (One-Time Setup)

Run these scripts in order:

```bash
# Step 1: Extract text from documents
python scripts/extract.py

# Step 2: Chunk documents into searchable segments
python scripts/chunk.py

# Step 3: Build the search index (downloads model on first run)
python scripts/embed.py
```

**Note:** The first run of `embed.py` will download the embedding model (~80MB). This only happens once.

### 4. Start the Search Server

```bash
python app/server.py
```

Open your browser to: **http://127.0.0.1:5000**

---

## Usage

### Web UI

1. Open http://127.0.0.1:5000
2. Enter a natural language query (e.g., "emergency shutdown procedure")
3. View results with document names, page numbers, and relevant text snippets

### Command-Line Search

```bash
python scripts/search.py
```

Interactive prompt for testing queries directly.

### API

**Search Endpoint:**

```bash
curl -X POST http://127.0.0.1:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "How to initialize the system?", "top_k": 5}'
```

**Get Statistics:**

```bash
curl http://127.0.0.1:5000/api/stats
```

---

## Project Structure

```
sop-quickfinder/
│
├── data/
│   ├── raw/           # Your original SOP PDFs/Docs (add files here)
│   ├── processed/     # Extracted text (auto-generated)
│   └── index/         # FAISS index + embeddings (auto-generated)
│
├── scripts/
│   ├── extract.py     # Extract text from PDFs/Docs
│   ├── chunk.py       # Segment text into chunks
│   ├── embed.py       # Generate embeddings & build index
│   └── search.py      # Command-line search interface
│
├── app/
│   ├── server.py      # Flask web server
│   └── ui.html        # Web UI
│
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

---

## How It Works

1. **Document Ingestion** (`extract.py`)
   - Extracts text from PDF/Word files
   - Preserves metadata (doc name, page number, section)

2. **Chunking** (`chunk.py`)
   - Splits documents into 500-1000 character chunks
   - Detects section headers automatically
   - Maintains context with overlap

3. **Embedding** (`embed.py`)
   - Uses `sentence-transformers/all-MiniLM-L6-v2` (local model)
   - Generates 384-dimensional vectors for each chunk
   - Builds FAISS index for fast similarity search

4. **Retrieval** (`search.py`, `server.py`)
   - User query → embedded locally
   - FAISS finds most similar chunks
   - Returns exact text with metadata (no generation)

---

## Security & Offline Operation

- ✅ **No Internet Required** - All processing is local
- ✅ **No Telemetry** - No data sent anywhere
- ✅ **Auditable** - Simple Python code, easy to review
- ✅ **Isolated** - Binds to localhost (127.0.0.1) only
- ✅ **No LLM Generation** - Pure retrieval, no content creation

---

## Configuration

### Adjust Chunk Size

Edit `scripts/chunk.py`:

```python
chunker = SOPChunker(processed_dir, chunk_size=800, overlap=100)
```

### Change Number of Results

Edit `scripts/search.py`:

```python
results = searcher.search(query, top_k=10)  # Default is 5
```

### Use Different Embedding Model

Edit `scripts/embed.py`:

```python
model_name = 'sentence-transformers/all-MiniLM-L6-v2'  # Change to another model
```

---

## Troubleshooting

### "Search engine not initialized"
Run the indexing pipeline:
```bash
python scripts/extract.py
python scripts/chunk.py
python scripts/embed.py
```

### "No results found"
- Check that SOPs are in `data/raw/`
- Verify index was built successfully
- Try broader search terms

### "Module not found"
```bash
pip install -r requirements.txt
```

### Re-index After Adding Documents
```bash
python scripts/extract.py
python scripts/chunk.py
python scripts/embed.py
```

---

## Performance

- **Index Build Time:** ~1-5 minutes for 50 SOPs (one-time)
- **Query Time:** <100ms per search
- **Memory Usage:** ~500MB RAM
- **Disk Space:** ~50MB per 100 SOPs

---

## Deployment on Secure Systems

### For Classified Environments:

1. **Build on Internet-Connected Machine:**
   ```bash
   pip download -r requirements.txt -d deps/
   python scripts/embed.py  # Downloads model
   ```

2. **Transfer to Secure System:**
   - Copy entire `sop-quickfinder/` directory
   - Copy `deps/` directory with pip packages
   - Copy embedding model from `~/.cache/torch/sentence_transformers/`

3. **Install Offline:**
   ```bash
   pip install --no-index --find-links=deps/ -r requirements.txt
   ```

---

## Weekend Build Timeline

### Day 1: Indexing Pipeline
- ✅ Extract text from SOPs
- ✅ Chunk documents
- ✅ Generate embeddings
- ✅ Build FAISS index

### Day 2: Search & UI
- ✅ Implement search logic
- ✅ Build Flask API
- ✅ Create web UI
- ✅ Test with sample SOPs

**Total Time:** ~6-8 hours for experienced developers

---

## License

This is a reference implementation for secure, offline document retrieval. Adapt as needed for your environment.

---

## Support

This tool is designed to be simple and self-contained. For issues:

1. Check `Troubleshooting` section above
2. Verify all dependencies are installed
3. Ensure index is built correctly
4. Review logs in terminal output

---

**Built for GSEs who need fast, accurate SOP access without compromising security.**
