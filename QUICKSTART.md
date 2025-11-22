# Quick Start Guide - GSE SOP Quick-Finder

## 5-Minute Setup

### Step 1: Install Dependencies
```bash
cd ~/Documents/sop-quickfinder
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Add Your SOPs
```bash
# Copy your SOP files to data/raw/
cp /path/to/your/SOPs/*.pdf data/raw/
```

### Step 3: Build the Index
```bash
# Run these three commands in order:
python scripts/extract.py
python scripts/chunk.py
python scripts/embed.py
```

**Note:** First run downloads the embedding model (~80MB). This happens once.

### Step 4: Start Searching
```bash
# Option A: Web UI
python app/server.py
# Open browser to http://127.0.0.1:5000

# Option B: Command Line
python scripts/search.py
```

---

## Example Workflow

1. **Initial Setup** (one time):
   ```bash
   cd ~/Documents/sop-quickfinder
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Add SOPs**:
   ```bash
   cp ~/my-sops/*.pdf data/raw/
   ```

3. **Build Index**:
   ```bash
   python scripts/extract.py && python scripts/chunk.py && python scripts/embed.py
   ```

4. **Search**:
   ```bash
   python app/server.py
   ```
   Go to http://127.0.0.1:5000 and search!

---

## Updating with New SOPs

When you add new documents:

```bash
# 1. Copy new files
cp /path/to/new-sops/*.pdf data/raw/

# 2. Re-run the pipeline
python scripts/extract.py
python scripts/chunk.py
python scripts/embed.py

# 3. Restart server
python app/server.py
```

---

## Testing Without SOPs

Want to test before adding real SOPs? Create a dummy SOP:

```bash
echo "SAMPLE SOP

1. INITIALIZATION
To initialize the system, press the green button.

2. SHUTDOWN
To shut down, press the red button.

3. EMERGENCY PROCEDURES
In case of emergency, activate the alarm." > data/raw/sample-sop.txt
```

Then convert to PDF or just test with text files (you'll need to add `.txt` support to `extract.py`).

---

## Quick Commands Reference

| Task | Command |
|------|---------|
| Activate environment | `source venv/bin/activate` |
| Extract text | `python scripts/extract.py` |
| Create chunks | `python scripts/chunk.py` |
| Build index | `python scripts/embed.py` |
| CLI search | `python scripts/search.py` |
| Web UI | `python app/server.py` |
| Re-index all | `python scripts/extract.py && python scripts/chunk.py && python scripts/embed.py` |

---

## What Each Script Does

- **extract.py**: Reads PDFs/Docs → saves text as JSON
- **chunk.py**: Splits text → creates searchable chunks
- **embed.py**: Converts chunks → vector embeddings → builds FAISS index
- **search.py**: Query interface (command line)
- **server.py**: Query interface (web UI)

---

**That's it! You now have a fully functional offline SOP search engine.**
