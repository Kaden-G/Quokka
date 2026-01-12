#!/usr/bin/env python3
"""
SOP Quick-Finder Flask Server
Provides web UI and API for searching SOPs.
"""

from flask import Flask, request, jsonify, render_template_string, send_from_directory
from pathlib import Path
import sys

# Add scripts directory to path
base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir / 'scripts'))

from search import SOPSearcher

app = Flask(__name__)

# Initialize searcher
index_dir = base_dir / 'data' / 'index'
searcher = None


def init_searcher():
    """Initialize the search engine."""
    global searcher
    try:
        searcher = SOPSearcher(str(index_dir))
        print(f"Search engine initialized with {len(searcher.metadata)} chunks")
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
    """Search API endpoint."""
    if searcher is None:
        return jsonify({'error': 'Search engine not initialized'}), 500

    data = request.json
    query = data.get('query', '').strip()
    top_k = data.get('top_k', 5)
    doc_filter = data.get('document', None)
    generate_answer = data.get('generate_answer', False)

    if not query:
        return jsonify({'error': 'Query cannot be empty'}), 400

    try:
        # Use RAG pipeline if answer generation is requested
        if generate_answer:
            response = searcher.search_and_answer(query, top_k=top_k)
            return jsonify(response)

        # Otherwise, just return search results
        if doc_filter:
            results = searcher.search_by_document(query, doc_filter, top_k=top_k)
        else:
            results = searcher.search(query, top_k=top_k)

        return jsonify({
            'query': query,
            'results': results,
            'count': len(results)
        })
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


def main():
    """Run the Flask server."""
    init_searcher()

    print("\n" + "="*60)
    print("GSE SOP Quick-Finder Server")
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
