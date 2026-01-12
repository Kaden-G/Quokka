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
from metrics import MetricsTracker

app = Flask(__name__)

# Initialize searcher and metrics tracker
index_dir = base_dir / 'data' / 'index'
searcher = None
metrics_tracker = None


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
    query = data.get('query', '').strip()
    top_k = data.get('top_k', 5)
    doc_filter = data.get('document', None)
    generate_answer = data.get('generate_answer', False)

    if not query:
        return jsonify({'error': 'Query cannot be empty'}), 400

    try:
        import time
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


@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    """Collect user feedback for a query."""
    if metrics_tracker is None:
        return jsonify({'error': 'Metrics tracker not initialized'}), 500

    data = request.json
    query_id = data.get('query_id')
    rating = data.get('rating')  # 1-5
    relevant_results = data.get('relevant_results', [])
    comments = data.get('comments', '')

    if not query_id or not rating:
        return jsonify({'error': 'query_id and rating required'}), 400

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
        days = int(request.args.get('days', 7))
        metrics = {
            'query_stats': metrics_tracker.get_query_stats(days),
            'top_queries': metrics_tracker.get_top_queries(10),
            'feedback_stats': metrics_tracker.get_feedback_stats()
        }
        return jsonify(metrics)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
