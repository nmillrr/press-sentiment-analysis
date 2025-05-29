# Press Sentiment Analysis App

This application analyzes sentiment in news articles and provides visualizations of the results. It includes both a command-line interface and a web interface built with Flask and Radix UI.

## Features

- **Single URL Analysis**: Analyze sentiment and extract entities from a single article
- **Batch URL Analysis**: Process multiple articles and compare their sentiment
- **Topic Search**: Find and analyze recent news articles about specific topics
- **Entity Recognition**: Identify organizations and people mentioned in articles
- **Sentiment Visualization**: Generate visualizations of sentiment analysis results

## Setup

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your NewsAPI key:

```
NEWSAPI_KEY=your_api_key_here
```

You can get a free API key from [NewsAPI.org](https://newsapi.org/).

## Usage

### Web Interface

Run the Flask app:

```bash
python app.py
```

Then open a web browser and navigate to `http://localhost:5000`.

### Command-Line Interface

The script can be used directly from the command line:

```bash
# Analyze a single URL
python press_sentiment_analysis.py url https://example.com/article

# Analyze multiple URLs
python press_sentiment_analysis.py batch https://example.com/article1 https://example.com/article2

# Search and analyze articles by topic
python press_sentiment_analysis.py topic "climate change" 5
```

## Technologies Used

- **Backend**:
  - Python with Flask
  - Hugging Face Transformers for NLP
  - NewsAPI for article search
  - Beautiful Soup for web scraping
  - Matplotlib and Seaborn for visualization

- **Frontend**:
  - React
  - Radix UI components
  - Custom CSS
