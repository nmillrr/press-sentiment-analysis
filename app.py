import os
import io
import base64
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from press_sentiment_analysis import (
    press_sentiment_tool, analyze_batch_urls, analyze_topic, visualize_sentiment
)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')
CORS(app)  # Enable CORS for all routes

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

@app.route('/analyze/url', methods=['POST'])
def analyze_url():
    """Analyze a single URL"""
    data = request.json
    if not data or 'url' not in data:
        return jsonify({'error': 'URL is required'}), 400
    
    url = data['url']
    result = press_sentiment_tool(url)
    
    # Generate visualization if successful
    if 'error' not in result:
        # Create a visualization
        buffer = io.BytesIO()
        visualize_sentiment([result], save_path=buffer)
        buffer.seek(0)
        
        # Convert to base64 for embedding in response
        img_str = base64.b64encode(buffer.read()).decode('utf-8')
        result['visualization'] = f"data:image/png;base64,{img_str}"
    
    return jsonify(result)

@app.route('/analyze/batch', methods=['POST'])
def analyze_batch():
    """Analyze a batch of URLs"""
    data = request.json
    if not data or 'urls' not in data or not isinstance(data['urls'], list):
        return jsonify({'error': 'URLs array is required'}), 400
    
    urls = data['urls']
    if not urls:
        return jsonify({'error': 'At least one URL is required'}), 400
    
    results = analyze_batch_urls(urls)
    
    # Generate visualization if successful
    if results:
        # Create a visualization
        buffer = io.BytesIO()
        visualize_sentiment(results, save_path=buffer)
        buffer.seek(0)
        
        # Convert to base64 for embedding in response
        img_str = base64.b64encode(buffer.read()).decode('utf-8')
        return jsonify({
            'results': results,
            'visualization': f"data:image/png;base64,{img_str}"
        })
    
    return jsonify({'results': results})

@app.route('/analyze/topic', methods=['POST'])
def analyze_news_topic():
    """Analyze news articles about a specific topic"""
    data = request.json
    if not data or 'topic' not in data:
        return jsonify({'error': 'Topic is required'}), 400
    
    topic = data['topic']
    max_articles = data.get('max_articles', 5)
    
    results = analyze_topic(topic, max_articles)
    
    # Handle error cases
    if isinstance(results, dict) and 'error' in results:
        return jsonify(results), 400
    
    # Generate visualization if successful
    if results:
        # Create a visualization
        buffer = io.BytesIO()
        visualize_sentiment(results, save_path=buffer)
        buffer.seek(0)
        
        # Convert to base64 for embedding in response
        img_str = base64.b64encode(buffer.read()).decode('utf-8')
        return jsonify({
            'results': results,
            'visualization': f"data:image/png;base64,{img_str}"
        })
    
    return jsonify({'results': results})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)