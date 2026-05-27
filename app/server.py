#!/usr/bin/env python3
"""
Quokka Flask Server
Provides web UI and API for searching SOPs.
"""

import time
import threading

from flask import Flask, request, jsonify, render_template_string, send_from_directory
from pathlib import Path
from werkzeug.utils import secure_filename
import os
import sys

# Add scripts directory to path
base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir / 'scripts'))

from search import SOPSearcher
from metrics import MetricsTracker
from extract import DocumentExtractor
from chunk import SOPChunker
from embed import EmbeddingIndexer

app = Flask(__name__)

# Initialize searcher and metrics tracker
index_dir = base_dir / 'data' / 'index'
raw_dir = base_dir / 'data' / 'raw'
searcher = None
metrics_tracker = None

ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.txt'}

# Pipeline build state
_build_lock = threading.Lock()
_build_status = {'running': False, 'step': '', 'error': '', 'done': False}


def init_searcher():
    """Initialize the search engine and metrics tracker."""
    global searcher, metrics_tracker
    try:
        searcher = SOPSearcher(str(index_dir))
        metrics_tracker = MetricsTracker()
        print(f"Search engine initialized with {len(searcher.metadata)} chunks")
        print("Metrics tracking enabled")
    except Exception as e:
        print(f"Error initializing searcher: {e}")
        print("Make sure to run extract.py, chunk.py, and embed.py first!")


@app.route('/')
def index():
    """Serve the main UI."""
    ui_file = Path(__file__).parent / 'ui.html'
    with open(ui_file, 'r') as f:
        return render_template_string(f.read())


@app.route('/api/search', methods=['POST'])
def api_search():
    """Search API endpoint with metrics tracking."""
    if searcher is None:
        return jsonify({'error': 'Search engine not initialized'}), 500

    data = request.json
    if not data or not isinstance(data, dict):
        return jsonify({'error': 'Request body must be JSON'}), 400

    query = str(data.get('query', '')).strip()
    top_k = data.get('top_k', 5)
    doc_filter = data.get('document', None)
    generate_answer = data.get('generate_answer', False)

    if not query:
        return jsonify({'error': 'Query cannot be empty'}), 400
    if len(query) > 2000:
        return jsonify({'error': 'Query too long (max 2000 characters)'}), 400
    if not isinstance(top_k, int) or top_k < 1 or top_k > 100:
        return jsonify({'error': 'top_k must be an integer between 1 and 100'}), 400

    try:
        start_time = time.time()

        # Use RAG pipeline if answer generation is requested
        if generate_answer:
            response = searcher.search_and_answer(query, top_k=top_k)
            results = response['results']
        else:
            # Otherwise, just return search results
            if doc_filter:
                results = searcher.search_by_document(query, doc_filter, top_k=top_k)
            else:
                results = searcher.search(query, top_k=top_k)

            response = {
                'query': query,
                'results': results,
                'count': len(results)
            }

        # Log metrics
        latency = time.time() - start_time
        if metrics_tracker:
            query_id = metrics_tracker.log_query(
                query=query,
                results=results,
                latency=latency,
                top_k=top_k,
                cache_hit=False,  # Could track this from searcher cache
                use_rerank=True,
                generate_answer=generate_answer
            )
            response['query_id'] = query_id  # For feedback collection

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/documents', methods=['GET'])
def api_documents():
    """Get list of indexed documents."""
    if searcher is None:
        return jsonify({'error': 'Search engine not initialized'}), 500

    try:
        docs = searcher.get_document_list()
        return jsonify({'documents': docs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def api_stats():
    """Get index statistics."""
    if searcher is None:
        return jsonify({'error': 'Search engine not initialized'}), 500

    try:
        stats = searcher.get_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    status = 'ok' if searcher is not None else 'not_initialized'
    return jsonify({'status': status})


@app.route('/api/settings', methods=['GET'])
def api_settings_get():
    """Get current LLM settings."""
    if searcher is None:
        return jsonify({'provider': 'none', 'model': '', 'api_base': '', 'has_key': False})
    return jsonify(searcher.get_llm_status())


@app.route('/api/settings', methods=['POST'])
def api_settings_post():
    """Update LLM settings at runtime."""
    data = request.json
    if not data or not isinstance(data, dict):
        return jsonify({'error': 'Request body must be JSON'}), 400

    provider = str(data.get('provider', '')).strip()
    api_key = str(data.get('apiKey', '')).strip()
    api_base = str(data.get('apiBase', '')).strip()
    model = str(data.get('model', '')).strip()
    ollama_model = str(data.get('ollamaModel', '')).strip()

    if searcher is not None:
        searcher.configure_llm(
            provider=provider,
            api_key=api_key,
            api_base=api_base,
            model=model,
            ollama_model=ollama_model
        )
        return jsonify({'status': 'ok', **searcher.get_llm_status()})
    else:
        return jsonify({'error': 'Search engine not initialized — build the index first'}), 400


@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    """Collect user feedback for a query."""
    if metrics_tracker is None:
        return jsonify({'error': 'Metrics tracker not initialized'}), 500

    data = request.json
    if not data or not isinstance(data, dict):
        return jsonify({'error': 'Request body must be JSON'}), 400

    query_id = data.get('query_id')
    rating = data.get('rating')  # 1-5
    relevant_results = data.get('relevant_results', [])
    comments = str(data.get('comments', ''))[:1000]  # Cap comment length

    if not query_id or not isinstance(query_id, int):
        return jsonify({'error': 'query_id must be an integer'}), 400
    if not isinstance(rating, int) or rating < 1 or rating > 5:
        return jsonify({'error': 'rating must be an integer between 1 and 5'}), 400

    try:
        metrics_tracker.log_feedback(
            query_id=query_id,
            rating=rating,
            relevant_results=relevant_results,
            comments=comments
        )
        return jsonify({'status': 'success', 'message': 'Feedback recorded'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/metrics', methods=['GET'])
def api_metrics():
    """Get system metrics."""
    if metrics_tracker is None:
        return jsonify({'error': 'Metrics tracker not initialized'}), 500

    try:
        days = min(max(int(request.args.get('days', 7)), 1), 365)
        metrics = {
            'query_stats': metrics_tracker.get_query_stats(days),
            'top_queries': metrics_tracker.get_top_queries(10),
            'feedback_stats': metrics_tracker.get_feedback_stats()
        }
        return jsonify(metrics)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/pouch', methods=['GET'])
def api_pouch_list():
    """List files currently in the pouch (data/raw/)."""
    try:
        raw_dir.mkdir(parents=True, exist_ok=True)
        files = []
        for f in sorted(raw_dir.iterdir()):
            if f.is_file() and not f.name.startswith('~$') and f.suffix.lower() in ALLOWED_EXTENSIONS:
                files.append({
                    'name': f.name,
                    'size': f.stat().st_size,
                    'ext': f.suffix.lower()
                })
        return jsonify({'files': files, 'count': len(files)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/pouch', methods=['POST'])
def api_pouch_upload():
    """Upload files to the pouch."""
    try:
        raw_dir.mkdir(parents=True, exist_ok=True)

        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400

        uploaded = []
        skipped = []
        for f in request.files.getlist('files'):
            if not f.filename:
                continue
            fname = secure_filename(f.filename)
            ext = Path(fname).suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                skipped.append({'name': f.filename, 'reason': f'Unsupported type ({ext}). Use .pdf, .docx, or .txt'})
                continue
            dest = raw_dir / fname
            f.save(str(dest))
            uploaded.append({'name': fname, 'size': dest.stat().st_size})

        return jsonify({
            'uploaded': uploaded,
            'skipped': skipped,
            'message': f'{len(uploaded)} file(s) added to the pouch'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/pouch/build', methods=['POST'])
def api_pouch_build():
    """Run the indexing pipeline and re-initialize the searcher."""
    global _build_status

    if _build_status['running']:
        return jsonify({'error': 'Build already in progress'}), 409

    _build_status = {'running': True, 'step': 'Starting...', 'error': '', 'done': False}
    thread = threading.Thread(target=_run_pipeline_thread, daemon=True)
    thread.start()
    return jsonify({'message': 'Build started'})


@app.route('/api/pouch/build/status', methods=['GET'])
def api_pouch_build_status():
    """Get the current build status."""
    return jsonify(_build_status)


def _run_pipeline_thread():
    """Run the full pipeline in a background thread."""
    global searcher, metrics_tracker, _build_status

    processed_dir = base_dir / 'data' / 'processed'

    try:
        # Step 1: Extract
        _build_status['step'] = 'Extracting text from documents...'
        extractor = DocumentExtractor(str(raw_dir), str(processed_dir))
        results = extractor.extract_all()
        _build_status['step'] = f'Extracted {len(results)} documents. Chunking...'

        # Step 2: Chunk
        chunker = SOPChunker(str(processed_dir), chunk_size=800, overlap=100)
        chunks = chunker.process_all()
        _build_status['step'] = f'{len(chunks)} chunks created. Building index...'

        # Step 3: Embed & Index
        indexer = EmbeddingIndexer(str(processed_dir), str(index_dir))
        indexer.build_index()

        # Save source manifest
        current_files = {}
        for f in extractor.get_raw_files():
            stat = f.stat()
            current_files[f.name] = {'mtime': stat.st_mtime, 'size': stat.st_size}
        indexer.save_source_manifest(current_files)

        # Re-initialize searcher
        _build_status['step'] = 'Initializing search engine...'
        searcher = SOPSearcher(str(index_dir))
        if metrics_tracker is None:
            metrics_tracker = MetricsTracker()

        _build_status['step'] = 'Complete'
        _build_status['done'] = True
        _build_status['running'] = False
        print('Pipeline complete — searcher re-initialized.')

    except Exception as e:
        _build_status['error'] = str(e)
        _build_status['running'] = False
        print(f'Pipeline error: {e}')


@app.route('/api/pouch/<filename>', methods=['DELETE'])
def api_pouch_delete(filename):
    """Remove a file from the pouch."""
    try:
        fname = secure_filename(filename)
        target = raw_dir / fname
        if not target.exists():
            return jsonify({'error': 'File not found'}), 404
        target.unlink()
        return jsonify({'message': f'{fname} removed from the pouch'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def main():
    """Run the Flask server."""
    init_searcher()

    print("\n" + "="*60)
    print("Quokka — SOP Search")
    print("="*60)
    print(f"Access the UI at: http://127.0.0.1:5000")
    print(f"API endpoint: http://127.0.0.1:5000/api/search")
    print("="*60 + "\n")

    # Run server (localhost only for security)
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=False,  # Disable debug in production
        threaded=True
    )


if __name__ == '__main__':
    main()
