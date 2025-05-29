import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
import time

# Load environment variables for API keys
load_dotenv()

def fetch_article_text(url):
    """Fetch and extract text from a given article URL."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, timeout=15, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract text from common article tags
        article_text = ''
        
        # Try to find main content container
        main_content = soup.find('article') or soup.find('main') or soup.find('div', class_=re.compile('article|content|main'))
        
        if main_content:
            for tag in main_content.find_all(['p', 'h1', 'h2', 'h3']):
                article_text += tag.get_text(strip=True) + ' '
        else:
            # Fallback to all paragraphs
            for tag in soup.find_all(['p', 'article', 'div']):
                if tag.name == 'p' or (tag.name == 'div' and any(cls in tag.get('class', []) for cls in ['article', 'content'])):
                    article_text += tag.get_text(strip=True) + ' '
        
        # Clean text: remove extra spaces, newlines, and special characters
        cleaned_text = re.sub(r'\s+', ' ', article_text).strip()
        return cleaned_text[:20000]  # Limit to 20000 chars for performance
    except Exception as e:
        return f"Error fetching article: {str(e)}"

def analyze_sentiment(text):
    """Analyze sentiment of the given text using a model with neutral labels."""
    if not text or len(text.strip()) < 10:
        return {"error": "Text is too short or empty"}
    
    # Load pre-trained sentiment analysis pipeline with 5-class model
    # This model outputs labels: 1-5 stars (1=very negative, 3=neutral, 5=very positive)
    classifier = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
    
    # Perform sentiment analysis
    result = classifier(text, truncation=True, max_length=512)
    
    # Map 5-star rating to sentiment label
    rating = int(result[0]['label'].split()[0])
    sentiment_mapping = {
        1: "negative",
        2: "negative",
        3: "neutral",
        4: "positive",
        5: "positive"
    }
    
    return {
        "sentiment": sentiment_mapping[rating],
        "rating": rating,
        "confidence": round(result[0]['score'], 4)
    }

def extract_entities(text):
    """Extract named entities from the text using Hugging Face's NER pipeline."""
    if not text or len(text.strip()) < 10:
        return {"error": "Text is too short or empty"}
    
    try:
        # Load pre-trained NER pipeline
        ner_pipeline = pipeline('ner', grouped_entities=True)
        
        # Extract entities (process in chunks to avoid token limits)
        max_length = 500  # Characters per chunk
        chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
        
        all_entities = []
        for chunk in chunks:
            entities = ner_pipeline(chunk)
            all_entities.extend(entities)
        
        # Filter and format entities
        formatted_entities = {}
        for entity in all_entities:
            entity_type = entity['entity_group']
            entity_text = entity['word']
            
            # Only keep ORG (organizations) and PER (people)
            if entity_type in ['ORG', 'PER']:
                entity_category = 'organization' if entity_type == 'ORG' else 'person'
                if entity_category not in formatted_entities:
                    formatted_entities[entity_category] = []
                if entity_text not in formatted_entities[entity_category]:
                    formatted_entities[entity_category].append(entity_text)
        
        # Count entities
        entity_counts = {}
        for category, entities in formatted_entities.items():
            entity_counts[category] = dict(Counter([e.lower() for e in entities]))
        
        return {
            "entities": formatted_entities,
            "entity_counts": entity_counts
        }
    except Exception as e:
        return {"error": f"Error extracting entities: {str(e)}"}

def fetch_news_by_topic(topic, max_results=5):
    """Fetch news articles about a specific topic using NewsAPI."""
    try:
        # Get API key from environment
        api_key = os.getenv('NEWSAPI_KEY')
        if not api_key:
            return {"error": "NewsAPI key not found. Set NEWSAPI_KEY in environment or .env file."}
        
        # Build the request
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": topic,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": max_results,
            "apiKey": api_key
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        if data["status"] != "ok":
            return {"error": f"API error: {data.get('message', 'Unknown error')}"}
        
        articles = data["articles"]
        return [{"title": article["title"], 
                 "url": article["url"], 
                 "source": article["source"]["name"],
                 "published_at": article["publishedAt"]} 
                for article in articles]
        
    except Exception as e:
        return {"error": f"Error fetching news: {str(e)}"}

def visualize_sentiment(results, save_path=None):
    """Create visualizations for sentiment analysis results."""
    if not results or len(results) == 0:
        return {"error": "No results to visualize"}
    
    # Create a DataFrame from results
    df = pd.DataFrame(results)
    
    # Set up plotting environment
    plt.figure(figsize=(14, 10))
    plt.style.use('ggplot')
    
    # 1. Sentiment distribution pie chart
    plt.subplot(2, 2, 1)
    sentiment_counts = df['sentiment'].value_counts()
    colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', 
            colors=[colors.get(s, 'blue') for s in sentiment_counts.index])
    plt.title('Sentiment Distribution')
    
    # 2. Confidence by sentiment boxplot
    plt.subplot(2, 2, 2)
    sns.boxplot(x='sentiment', y='confidence', data=df, palette=colors)
    plt.title('Confidence by Sentiment')
    
    # 3. Source-sentiment heatmap (if enough data)
    if len(df) >= 3 and 'source' in df.columns:
        plt.subplot(2, 2, 3)
        source_sentiment = pd.crosstab(df['source'], df['sentiment'])
        sns.heatmap(source_sentiment, cmap='YlGnBu', annot=True, fmt='d')
        plt.title('Source vs Sentiment')
    
    # 4. Entity frequency bar chart (if entity data exists)
    if 'top_entities' in df.columns and df['top_entities'].notna().any():
        plt.subplot(2, 2, 4)
        
        # Collect all entities
        all_entities = []
        for entities in df['top_entities'].dropna():
            all_entities.extend(entities.split(', '))
        
        # Count and plot top 10
        entity_counts = Counter(all_entities).most_common(10)
        entities, counts = zip(*entity_counts) if entity_counts else ([], [])
        plt.barh(list(entities), list(counts))
        plt.xlabel('Frequency')
        plt.title('Top 10 Mentioned Entities')
    
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path)
        return {"message": f"Visualization saved to {save_path}"}
    else:
        # In an interactive environment, plt.show() would be called here
        # Since we're likely in a script, we'll save to a default location
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = f"sentiment_analysis_{timestamp}.png"
        plt.savefig(default_path)
        return {"message": f"Visualization saved to {default_path}"}

def analyze_batch_urls(urls):
    """Process a batch of article URLs and analyze their sentiment."""
    results = []
    
    for url in urls:
        print(f"Processing: {url}")
        result = press_sentiment_tool(url)
        if "error" not in result:
            results.append(result)
        else:
            print(f"Error with {url}: {result['error']}")
        
        # Be kind to servers - add a delay between requests
        time.sleep(1)
        
    return results

def press_sentiment_tool(url):
    """Analyze sentiment and entities in a press article from a URL."""
    # Fetch article text
    article_text = fetch_article_text(url)
    if article_text.startswith("Error"):
        return {"error": article_text}
    
    # Analyze sentiment
    sentiment_result = analyze_sentiment(article_text)
    if "error" in sentiment_result:
        return sentiment_result
    
    # Extract entities
    entity_result = extract_entities(article_text)
    
    # Get top entities (if available)
    top_entities = []
    if "entity_counts" in entity_result:
        # Get top 5 organizations
        if "organization" in entity_result["entity_counts"]:
            top_orgs = sorted(entity_result["entity_counts"]["organization"].items(), 
                             key=lambda x: x[1], reverse=True)[:5]
            top_entities.extend([org[0] for org in top_orgs])
        
        # Get top 5 people
        if "person" in entity_result["entity_counts"]:
            top_people = sorted(entity_result["entity_counts"]["person"].items(),
                               key=lambda x: x[1], reverse=True)[:5]
            top_entities.extend([person[0] for person in top_people])
    
    # Format result
    return {
        "url": url,
        "article_excerpt": article_text[:200] + "...",
        "sentiment": sentiment_result["sentiment"],
        "rating": sentiment_result.get("rating", 0),  # 1-5 scale
        "confidence": sentiment_result["confidence"],
        "top_entities": ", ".join(top_entities) if top_entities else None,
        "entity_data": entity_result.get("entity_counts", {})
    }

def analyze_topic(topic, max_articles=5):
    """Fetch and analyze recent news articles about a specific topic."""
    # Fetch articles about the topic
    articles = fetch_news_by_topic(topic, max_results=max_articles)
    
    if "error" in articles:
        return {"error": articles["error"]}
    
    if not articles:
        return {"error": "No articles found for this topic"}
    
    print(f"Found {len(articles)} articles about '{topic}'")
    
    # Analyze each article
    results = []
    for article in articles:
        url = article["url"]
        print(f"Analyzing: {article['title']} from {article['source']}")
        
        # Add source to the result
        result = press_sentiment_tool(url)
        if "error" not in result:
            result["source"] = article["source"]
            result["title"] = article["title"]
            result["published_at"] = article["published_at"]
            results.append(result)
        else:
            print(f"Error analyzing {url}: {result['error']}")
        
        # Be kind to servers - add a delay between requests
        time.sleep(1)
    
    # Create visualizations if we have results
    if results:
        viz_result = visualize_sentiment(results)
        print(viz_result.get("message", "Visualization complete"))
    
    return results

if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single URL analysis:  python press_sentiment_analysis.py url https://example.com/article")
        print("  Batch URL analysis:   python press_sentiment_analysis.py batch url1 url2 url3")
        print("  Topic analysis:       python press_sentiment_analysis.py topic \"climate change\" [max_articles]")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "url" and len(sys.argv) >= 3:
        url = sys.argv[2]
        result = press_sentiment_tool(url)
        print(json.dumps(result, indent=2))
    
    elif command == "batch" and len(sys.argv) >= 3:
        urls = sys.argv[2:]
        results = analyze_batch_urls(urls)
        print(f"Analyzed {len(results)} articles successfully")
        print(json.dumps(results, indent=2))
        visualize_sentiment(results)
    
    elif command == "topic" and len(sys.argv) >= 3:
        topic = sys.argv[2]
        max_articles = int(sys.argv[3]) if len(sys.argv) >= 4 else 5
        results = analyze_topic(topic, max_articles)
        print(json.dumps(results, indent=2))
    
    else:
        print("Invalid command or missing arguments")
        print("Usage:")
        print("  Single URL analysis:  python press_sentiment_analysis.py url https://example.com/article")
        print("  Batch URL analysis:   python press_sentiment_analysis.py batch url1 url2 url3")
        print("  Topic analysis:       python press_sentiment_analysis.py topic \"climate change\" [max_articles]")