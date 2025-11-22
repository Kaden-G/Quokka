# Getting Started with SOP Quick-Finder

This guide will get you up and running in just a few minutes!

## Prerequisites

- Python 3.8 or higher
- 500MB free disk space
- macOS, Linux, or Windows

## Step-by-Step Setup

### 1. Install Dependencies

```bash
cd sop-quickfinder

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate     # On Windows

# Install packages
pip install -r requirements.txt
```

### 2. Add Your SOP Documents

Copy your SOP files to the `data/raw/` directory:

```bash
# Example
cp ~/my-sops/*.pdf data/raw/
```

**Supported formats:**
- PDF files (`.pdf`)
- Word documents (`.docx`)
- Plain text files (`.txt`)

We've included a sample SOP (`sample-sop.txt`) for testing!

### 3. Build the Search Index

**Option A: Run Complete Pipeline (Recommended)**

```bash
python scripts/run_pipeline.py
```

This runs all three steps automatically:
1. Extracts text from your documents
2. Chunks them into searchable segments
3. Builds the vector search index

**Option B: Run Steps Individually**

```bash
python scripts/extract.py   # Step 1: Extract text
python scripts/chunk.py     # Step 2: Create chunks
python scripts/embed.py     # Step 3: Build index
```

**Note:** The first run will download a ~80MB AI model. This only happens once.

### 4. Start Searching!

**Web Interface (Recommended):**

```bash
python app/server.py
```

Then open your browser to: **http://127.0.0.1:5000**

**Command-Line Interface:**

```bash
python scripts/search.py
```

## Quick Test

Try these example queries on the sample SOP:

- "How do I start the cooling system?"
- "Emergency shutdown procedure"
- "What should I do if temperature is rising?"
- "Power distribution activation steps"
- "Troubleshooting communications failure"

## What's Happening Under the Hood?

1. **Extract** - Reads your PDFs/Docs and saves the text as JSON
2. **Chunk** - Splits documents into 800-character chunks with 100-character overlap
3. **Embed** - Converts each chunk into a 384-dimensional vector using a local AI model
4. **Index** - Stores vectors in FAISS for ultra-fast similarity search
5. **Search** - Your query gets embedded, then we find the most similar chunks

## Adding More Documents

When you add new SOPs:

```bash
# 1. Copy new files
cp /path/to/new-sops/*.pdf data/raw/

# 2. Re-run the pipeline
python scripts/run_pipeline.py

# 3. Restart the server
python app/server.py
```

## Troubleshooting

### "ModuleNotFoundError"
Make sure your virtual environment is activated:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### "No such file or directory: data/raw"
Run from the project root directory:
```bash
cd /path/to/sop-quickfinder
python scripts/run_pipeline.py
```

### "Search engine not initialized"
You need to build the index first:
```bash
python scripts/run_pipeline.py
```

### "No results found"
- Make sure SOPs are in `data/raw/`
- Try broader search terms
- Check that indexing completed successfully

## Project Structure

```
sop-quickfinder/
├── data/
│   ├── raw/           ← Put your SOPs here
│   ├── processed/     ← Auto-generated JSON files
│   └── index/         ← Auto-generated search index
│
├── scripts/
│   ├── extract.py     ← Step 1: Extract text
│   ├── chunk.py       ← Step 2: Create chunks
│   ├── embed.py       ← Step 3: Build index
│   ├── search.py      ← CLI search tool
│   └── run_pipeline.py ← Run all steps
│
├── app/
│   ├── server.py      ← Web server
│   └── ui.html        ← Web UI
│
└── requirements.txt   ← Python dependencies
```

## API Usage

The web server provides REST endpoints:

**Search:**
```bash
curl -X POST http://127.0.0.1:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "cooling system startup", "top_k": 5}'
```

**Get Statistics:**
```bash
curl http://127.0.0.1:5000/api/stats
```

**List Documents:**
```bash
curl http://127.0.0.1:5000/api/documents
```

## Security Features

✅ **100% Offline** - No internet required after initial setup
✅ **Local Processing** - All data stays on your machine
✅ **No Telemetry** - Zero tracking or data collection
✅ **Localhost Only** - Server binds to 127.0.0.1
✅ **No Generation** - Only retrieves existing text, never creates content

## Performance

- **Query Time:** < 100ms
- **Memory Usage:** ~500MB RAM
- **Index Size:** ~50MB per 100 SOPs
- **Supported Scale:** Thousands of documents

## Next Steps

1. Try the sample SOP search
2. Add your own SOP documents
3. Customize chunk size in [chunk.py](scripts/chunk.py) if needed
4. Explore the API endpoints
5. Integrate with your existing tools

## Need Help?

- Check [README.md](README.md) for full documentation
- Review [QUICKSTART.md](QUICKSTART.md) for condensed instructions
- All code is heavily commented - read the source!

---

**You're all set! Start searching your SOPs with natural language queries.**
