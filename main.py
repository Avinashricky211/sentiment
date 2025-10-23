from flask import Flask, request, jsonify
from transformers import pipeline
import os

# Initialize Flask app
app = Flask(__name__)

# Initialize the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

@app.route('/', methods=['GET'])
def home():
    """Root endpoint with API information"""
    return jsonify({
        'message': 'Sentiment Analysis API',
        'version': '1.0',
        'endpoints': {
            'POST /analyze': 'Analyze sentiment from JSON payload',
            'GET /analyze': 'Analyze sentiment from query parameter',
            'GET /health': 'Health check'
        },
        'usage_examples': {
            'POST': 'curl -X POST http://localhost:5000/analyze -H "Content-Type: application/json" -d \'{"text":"I love this!"}\'',
            'GET': 'curl "http://localhost:5000/analyze?text=I%20love%20this!"',
            'health': 'curl http://localhost:5000/health'
        }
    }), 200

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    """
    POST endpoint for sentiment analysis
    Expects JSON with 'text' field
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate input
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Please provide text in JSON format with "text" key'
            }), 400
        
        text = data['text']
        
        # Validate text is not empty
        if not text.strip():
            return jsonify({
                'error': 'Text cannot be empty'
            }), 400
        
        # Perform sentiment analysis
        results = sentiment_analyzer(text)
        
        # Format response
        response = {
            'input_text': text,
            'sentiment_results': []
        }
        
        for result in results:
            response['sentiment_results'].append({
                'label': result['label'],
                'score': round(result['score'], 4)
            })
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'error': f'An error occurred during sentiment analysis: {str(e)}'
        }), 500

@app.route('/analyze', methods=['GET'])
def analyze_sentiment_get():
    """
    GET endpoint for sentiment analysis
    Accepts text as query parameter
    """
    try:
        # Get text from query parameters
        text = request.args.get('text', '')
        
        # Validate input
        if not text.strip():
            return jsonify({
                'error': 'Please provide text using the "text" query parameter'
            }), 400
        
        # Perform sentiment analysis
        results = sentiment_analyzer(text)
        
        # Format response
        response = {
            'input_text': text,
            'sentiment_results': []
        }
        
        for result in results:
            response['sentiment_results'].append({
                'label': result['label'],
                'score': round(result['score'], 4)
            })
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'error': f'An error occurred during sentiment analysis: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Sentiment Analysis API',
        'model': 'cardiffnlp/twitter-roberta-base-sentiment-latest'
    }), 200

if __name__ == '__main__':
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    # Get host from environment variable or default to localhost
    # Use '0.0.0.0' for production environments to accept external connections
    host = os.environ.get('HOST', '127.0.0.1')
    
    # Debug mode based on environment
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    print(f"Starting Sentiment Analysis API on {host}:{port}")
    print(f"Debug mode: {debug}")
    print("Available endpoints:")
    print("  GET  /          - API information")
    print("  POST /analyze   - Analyze sentiment from JSON payload")
    print("  GET  /analyze   - Analyze sentiment from query parameter")
    print("  GET  /health    - Health check")
    
    app.run(host=host, port=port, debug=debug)