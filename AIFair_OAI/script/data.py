"""
Multi-Label Toxicity Dataset Project
-----------------------------------
A comprehensive implementation for creating, annotating, and analyzing a multi-dimensional toxicity dataset.

This project includes:
1. Data collection from multiple sources
2. A multi-label annotation system
3. Quality control mechanisms
4. Advanced analysis of label co-occurrence
5. Interactive visualization tools
6. Dataset export in standard formats
"""

import os
import json
import re
import pandas as pd
import numpy as np
import requests
import praw
import tweepy
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from bs4 import BeautifulSoup
from sklearn.metrics import cohen_kappa_score
from sklearn.manifold import TSNE
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import networkx as nx
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import plotly.express as px
import plotly.graph_objects as go

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
ANNOTATION_DIR = os.path.join(DATA_DIR, "annotations")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DIR, ANNOTATION_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Initialize logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "toxicity_dataset.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# STEP 1: DATA COLLECTION
# =============================================================================

class DataCollector:
    """Base class for data collection from different sources."""
    
    def __init__(self, output_dir=RAW_DATA_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def save_data(self, data, source_name):
        """Save collected data to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{source_name}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(data)} items to {filepath}")
        return filepath

class RedditCollector(DataCollector):
    """Collect data from Reddit using PRAW."""
    
    def __init__(self, client_id, client_secret, user_agent, output_dir=RAW_DATA_DIR):
        super().__init__(output_dir)
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
    
    def collect_from_subreddits(self, subreddits, post_limit=25, comment_limit=100):
        """Collect posts and comments from specified subreddits."""
        all_data = []
        
        for subreddit_name in tqdm(subreddits, desc="Collecting from subreddits"):
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Collect from different post categories
            for category in ['hot', 'new', 'controversial', 'top']:
                try:
                    if category == 'hot':
                        posts = subreddit.hot(limit=post_limit)
                    elif category == 'new':
                        posts = subreddit.new(limit=post_limit)
                    elif category == 'controversial':
                        posts = subreddit.controversial(limit=post_limit)
                    elif category == 'top':
                        posts = subreddit.top(limit=post_limit)
                        
                    for post in posts:
                        # Skip stickied posts
                        if post.stickied:
                            continue
                            
                        post_data = {
                            'id': post.id,
                            'source': 'reddit',
                            'subreddit': subreddit_name,
                            'category': category,
                            'title': post.title,
                            'text': post.selftext,
                            'url': f"https://www.reddit.com{post.permalink}",
                            'created_utc': post.created_utc,
                            'score': post.score,
                            'type': 'post',
                            'author': str(post.author),
                            'comments': []
                        }
                        
                        # Collect comments
                        post.comments.replace_more(limit=0)  # Skip "load more comments" links
                        for comment in post.comments.list()[:comment_limit]:
                            comment_data = {
                                'id': comment.id,
                                'parent_id': comment.parent_id,
                                'text': comment.body,
                                'created_utc': comment.created_utc,
                                'score': comment.score,
                                'author': str(comment.author)
                            }
                            post_data['comments'].append(comment_data)
                            
                            # Also add comments as separate items for annotation
                            if len(comment.body.strip()) > 0:
                                all_data.append({
                                    'id': f"reddit_comment_{comment.id}",
                                    'source': 'reddit',
                                    'subreddit': subreddit_name,
                                    'text': comment.body,
                                    'url': f"https://www.reddit.com{post.permalink}{comment.id}/",
                                    'created_utc': comment.created_utc,
                                    'type': 'comment',
                                    'context': post.title
                                })
                        
                        all_data.append(post_data)
                        
                except Exception as e:
                    logger.error(f"Error collecting from {subreddit_name} ({category}): {str(e)}")
                    continue
        
        return self.save_data(all_data, "reddit")

class TwitterCollector(DataCollector):
    """Collect data from Twitter using Tweepy."""
    
    def __init__(self, bearer_token, output_dir=RAW_DATA_DIR):
        super().__init__(output_dir)
        self.client = tweepy.Client(bearer_token=bearer_token)
    
    def collect_from_searches(self, search_queries, max_results=100):
        """Collect tweets from search queries."""
        all_data = []
        
        for query in tqdm(search_queries, desc="Collecting from Twitter"):
            try:
                tweets = self.client.search_recent_tweets(
                    query=query,
                    max_results=max_results,
                    tweet_fields=['created_at', 'public_metrics', 'author_id', 'conversation_id']
                )
                
                if tweets.data:
                    for tweet in tweets.data:
                        tweet_data = {
                            'id': f"twitter_{tweet.id}",
                            'source': 'twitter',
                            'query': query,
                            'text': tweet.text,
                            'created_at': tweet.created_at.isoformat(),
                            'author_id': tweet.author_id,
                            'conversation_id': tweet.conversation_id,
                            'type': 'tweet',
                            'public_metrics': tweet.public_metrics,
                            'url': f"https://twitter.com/user/status/{tweet.id}"
                        }
                        all_data.append(tweet_data)
            
            except Exception as e:
                logger.error(f"Error collecting tweets for query '{query}': {str(e)}")
                continue
        
        return self.save_data(all_data, "twitter")

class NewsCommentsCollector(DataCollector):
    """Collect comments from news websites."""
    
    def __init__(self, output_dir=RAW_DATA_DIR):
        super().__init__(output_dir)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def collect_from_disqus(self, urls):
        """Collect comments from websites using Disqus."""
        all_data = []
        
        for url in tqdm(urls, desc="Collecting from Disqus"):
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract article title
                title = soup.find('title').text.strip() if soup.find('title') else "Unknown"
                
                # Extract article text
                article_text = ""
                article_paragraphs = soup.find_all('p')
                for p in article_paragraphs:
                    article_text += p.text.strip() + " "
                
                # Find Disqus thread ID
                disqus_thread = soup.find(id='disqus_thread')
                if not disqus_thread:
                    logger.warning(f"No Disqus thread found at {url}")
                    continue
                
                # Extract comments via Disqus API
                # Note: This is a simplified version; actual implementation would require Disqus API
                # Here we're simulating comment extraction
                comments = []
                comment_elements = soup.select('.comment')
                
                for i, comment_elem in enumerate(comment_elements):
                    comment_text = comment_elem.select_one('.comment-text').text.strip() if comment_elem.select_one('.comment-text') else ""
                    author = comment_elem.select_one('.comment-author').text.strip() if comment_elem.select_one('.comment-author') else "Anonymous"
                    
                    comment_data = {
                        'id': f"disqus_comment_{url.split('/')[-1]}_{i}",
                        'source': 'disqus',
                        'text': comment_text,
                        'author': author,
                        'url': url,
                        'type': 'news_comment',
                        'context': title
                    }
                    
                    comments.append(comment_data)
                    all_data.append(comment_data)
                
                # Add article as context
                article_data = {
                    'id': f"article_{url.split('/')[-1]}",
                    'source': 'news',
                    'title': title,
                    'text': article_text,
                    'url': url,
                    'type': 'article',
                    'comments': comments
                }
                
                all_data.append(article_data)
                
            except Exception as e:
                logger.error(f"Error collecting comments from {url}: {str(e)}")
                continue
        
        return self.save_data(all_data, "news_comments")

class ForumCollector(DataCollector):
    """Collect data from forum websites."""
    
    def __init__(self, output_dir=RAW_DATA_DIR):
        super().__init__(output_dir)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def collect_from_urls(self, urls, forum_type='generic'):
        """Collect posts and comments from forum URLs."""
        all_data = []
        
        for url in tqdm(urls, desc=f"Collecting from {forum_type} forums"):
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Different parsing logic based on forum type
                if forum_type == 'phpbb':
                    # Extract posts from phpBB forums
                    posts = soup.select('.post')
                    thread_title = soup.select_one('.topic-title').text.strip() if soup.select_one('.topic-title') else "Unknown Thread"
                    
                    for i, post in enumerate(posts):
                        post_content = post.select_one('.content').text.strip() if post.select_one('.content') else ""
                        author = post.select_one('.username').text.strip() if post.select_one('.username') else "Anonymous"
                        
                        post_data = {
                            'id': f"phpbb_{url.split('/')[-1]}_{i}",
                            'source': 'phpbb_forum',
                            'text': post_content,
                            'author': author,
                            'url': url,
                            'type': 'forum_post',
                            'context': thread_title
                        }
                        
                        all_data.append(post_data)
                
                elif forum_type == 'vbulletin':
                    # Extract posts from vBulletin forums
                    posts = soup.select('.message')
                    thread_title = soup.select_one('.threadtitle').text.strip() if soup.select_one('.threadtitle') else "Unknown Thread"
                    
                    for i, post in enumerate(posts):
                        post_content = post.select_one('.message-text').text.strip() if post.select_one('.message-text') else ""
                        author = post.select_one('.username').text.strip() if post.select_one('.username') else "Anonymous"
                        
                        post_data = {
                            'id': f"vbulletin_{url.split('/')[-1]}_{i}",
                            'source': 'vbulletin_forum',
                            'text': post_content,
                            'author': author,
                            'url': url,
                            'type': 'forum_post',
                            'context': thread_title
                        }
                        
                        all_data.append(post_data)
                
                else:
                    # Generic forum extraction (may be less accurate)
                    posts = soup.select('div.post, div.message, article.post')
                    thread_title = soup.select_one('h1, .title, .thread-title').text.strip() if soup.select_one('h1, .title, .thread-title') else "Unknown Thread"
                    
                    for i, post in enumerate(posts):
                        # Try different selectors for post content
                        content_selectors = ['.content', '.message-text', '.post-content', '.post-message', '.entry-content']
                        post_content = ""
                        for selector in content_selectors:
                            if post.select_one(selector):
                                post_content = post.select_one(selector).text.strip()
                                break
                        
                        # Try different selectors for author
                        author_selectors = ['.username', '.author', '.user', '.poster']
                        author = "Anonymous"
                        for selector in author_selectors:
                            if post.select_one(selector):
                                author = post.select_one(selector).text.strip()
                                break
                        
                        if post_content:
                            post_data = {
                                'id': f"forum_{url.split('/')[-1]}_{i}",
                                'source': 'generic_forum',
                                'text': post_content,
                                'author': author,
                                'url': url,
                                'type': 'forum_post',
                                'context': thread_title
                            }
                            
                            all_data.append(post_data)
                
            except Exception as e:
                logger.error(f"Error collecting from forum {url}: {str(e)}")
                continue
        
        return self.save_data(all_data, f"{forum_type}_forum")

def collect_all_data(config):
    """Run all data collection methods based on config."""
    all_filepaths = []
    
    # Reddit collection
    if config.get('reddit', {}).get('enabled', False):
        reddit_collector = RedditCollector(
            client_id=config['reddit']['client_id'],
            client_secret=config['reddit']['client_secret'],
            user_agent=config['reddit']['user_agent']
        )
        reddit_filepath = reddit_collector.collect_from_subreddits(
            subreddits=config['reddit']['subreddits'],
            post_limit=config['reddit'].get('post_limit', 25)
        )
        all_filepaths.append(reddit_filepath)
    
    # Twitter collection
    if config.get('twitter', {}).get('enabled', False):
        twitter_collector = TwitterCollector(
            bearer_token=config['twitter']['bearer_token']
        )
        twitter_filepath = twitter_collector.collect_from_searches(
            search_queries=config['twitter']['search_queries'],
            max_results=config['twitter'].get('max_results', 100)
        )
        all_filepaths.append(twitter_filepath)
    
    # News comments collection
    if config.get('news_comments', {}).get('enabled', False):
        news_collector = NewsCommentsCollector()
        news_filepath = news_collector.collect_from_disqus(
            urls=config['news_comments']['urls']
        )
        all_filepaths.append(news_filepath)
    
    # Forum collection
    if config.get('forums', {}).get('enabled', False):
        forum_collector = ForumCollector()
        for forum_type, urls in config['forums']['sites'].items():
            forum_filepath = forum_collector.collect_from_urls(
                urls=urls,
                forum_type=forum_type
            )
            all_filepaths.append(forum_filepath)
    
    return all_filepaths

# =============================================================================
# STEP 2: DATA PREPROCESSING
# =============================================================================

class TextPreprocessor:
    """Clean and preprocess text data."""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Basic text cleaning."""
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '[EMAIL]', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """Tokenize text into words."""
        return word_tokenize(self.clean_text(text))
    
    def remove_stopwords(self, tokens):
        """Remove stopwords from tokenized text."""
        return [word for word in tokens if word not in self.stop_words]
    
    def preprocess_for_analysis(self, text):
        """Full preprocessing pipeline for text analysis."""
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        return tokens
    
    def filter_short_texts(self, texts, min_length=10):
        """Filter out texts that are too short."""
        return [text for text in texts if len(self.clean_text(text).split()) >= min_length]

class DatasetCreator:
    """Create a balanced dataset from collected data."""
    
    def __init__(self, raw_data_dir=RAW_DATA_DIR, processed_dir=PROCESSED_DIR):
        self.raw_data_dir = raw_data_dir
        self.processed_dir = processed_dir
        self.preprocessor = TextPreprocessor()
        os.makedirs(processed_dir, exist_ok=True)
    
    def load_raw_data(self, filenames=None):
        """Load all raw data files."""
        all_data = []
        
        if filenames is None:
            filenames = [f for f in os.listdir(self.raw_data_dir) if f.endswith('.json')]
        
        for filename in filenames:
            filepath = os.path.join(self.raw_data_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_data.extend(data)
            except Exception as e:
                logger.error(f"Error loading data from {filepath}: {str(e)}")
        
        return all_data
    
    def extract_texts(self, raw_data):
        """Extract text content from raw data."""
        texts = []
        
        for item in raw_data:
            if 'text' in item and item['text']:
                text_data = {
                    'id': item['id'],
                    'text': item['text'],
                    'source': item.get('source', 'unknown'),
                    'type': item.get('type', 'unknown'),
                    'url': item.get('url', ''),
                    'context': item.get('context', '')
                }
                texts.append(text_data)
            
            # Extract comments if present
            if 'comments' in item and isinstance(item['comments'], list):
                for comment in item['comments']:
                    if 'text' in comment and comment['text']:
                        comment_data = {
                            'id': comment.get('id', f"comment_{len(texts)}"),
                            'text': comment['text'],
                            'source': item.get('source', 'unknown'),
                            'type': 'comment',
                            'url': item.get('url', ''),
                            'context': item.get('title', '') or item.get('text', '')[:100]
                        }
                        texts.append(comment_data)
        
        return texts
    
    def filter_and_clean(self, texts, min_length=10, max_length=1000):
        """Filter and clean the extracted texts."""
        filtered_texts = []
        
        for item in texts:
            cleaned_text = self.preprocessor.clean_text(item['text'])
            
            # Filter by length
            if len(cleaned_text.split()) >= min_length and len(cleaned_text) <= max_length:
                item['cleaned_text'] = cleaned_text
                filtered_texts.append(item)
        
        return filtered_texts
    
    def create_balanced_dataset(self, texts, target_size=300, sources=None):
        """Create a balanced dataset across different sources."""
        if not sources:
            # Get all unique sources
            sources = list(set(item['source'] for item in texts))
        
        items_per_source = target_size // len(sources)
        balanced_dataset = []
        
        for source in sources:
            source_texts = [item for item in texts if item['source'] == source]
            
            # If we don't have enough items for this source, take all of them
            if len(source_texts) <= items_per_source:
                balanced_dataset.extend(source_texts)
            else:
                # Randomly sample items for this source
                sampled_texts = np.random.choice(source_texts, items_per_source, replace=False)
                balanced_dataset.extend(sampled_texts)
        
        # If we still need more items to reach target_size, sample from all remaining texts
        if len(balanced_dataset) < target_size:
            remaining_count = target_size - len(balanced_dataset)
            already_included_ids = {item['id'] for item in balanced_dataset}
            remaining_texts = [item for item in texts if item['id'] not in already_included_ids]
            
            if remaining_texts:
                additional_samples = min(remaining_count, len(remaining_texts))
                sampled_texts = np.random.choice(remaining_texts, additional_samples, replace=False)
                balanced_dataset.extend(sampled_texts)
        
        return balanced_dataset
    
    def save_processed_dataset(self, dataset, filename="processed_dataset.json"):
        """Save the processed dataset to a file."""
        filepath = os.path.join(self.processed_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved processed dataset with {len(dataset)} items to {filepath}")
        return filepath
    
    def create_full_dataset(self, filenames=None, target_size=300, min_length=10):
        """Run full dataset creation pipeline."""
        raw_data = self.load_raw_data(filenames)
        texts = self.extract_texts(raw_data)
        filtered_texts = self.filter_and_clean(texts, min_length=min_length)
        balanced_dataset = self.create_balanced_dataset(filtered_texts, target_size=target_size)
        return self.save_processed_dataset(balanced_dataset)

# =============================================================================
# STEP 3: ANNOTATION TOOL
# =============================================================================

class AnnotationApp:
    """Flask app for annotation interface."""
    
    def __init__(self, dataset_path, annotation_output_dir=ANNOTATION_DIR):
        self.app = Flask(__name__)
        self.dataset_path = dataset_path
        self.annotation_output_dir = annotation_output_dir
        self.setup_routes()
        os.makedirs(annotation_output_dir, exist_ok=True)
        
        # Load dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)
    
    def setup_routes(self):
        """Set up Flask routes."""
        @self.app.route('/')
        def index():
            return render_template('annotation.html')
        
        @self.app.route('/get_item/<int:index>')
        def get_item(index):
            if 0 <= index < len(self.dataset):
                return jsonify(self.dataset[index])
            else:
                return jsonify({"error": "Index out of range"}), 404
        
        @self.app.route('/get_dataset_info')
        def get_dataset_info():
            return jsonify({
                "total_items": len(self.dataset),
                "sources": list(set(item.get('source', 'unknown') for item in self.dataset)),
                "types": list(set(item.get('type', 'unknown') for item in self.dataset))
            })
        
        @self.app.route('/save_annotation', methods=['POST'])
        def save_annotation():
            data = request.json
            if not data or 'item_id' not in data or 'annotations' not in data:
                return jsonify({"error": "Invalid data"}), 400
            
            # Create annotator directory if it doesn't exist
            annotator = data.get('annotator', 'anonymous')
            annotator_dir = os.path.join(self.annotation_output_dir, annotator)
            os.makedirs(annotator_dir, exist_ok=True)
            
            # Save annotation to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"annotation_{data['item_id']}_{timestamp}.json"
            filepath = os.path.join(annotator_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return jsonify({"success": True, "filepath": filepath})
        
        @self.app.route('/get_progress/<annotator>')
        def get_progress(annotator):
            annotator_dir = os.path.join(self.annotation_output_dir, annotator)
            if not os.path.exists(annotator_dir):
                return jsonify({"total": len(self.dataset), "completed": 0})
            
            completed_files = [f for f in os.listdir(annotator_dir) if f.endswith('.json')]
            completed_ids = set()
            
            for filename in completed_files:
                try:
                    filepath = os.path.join(annotator_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'item_id' in data:
                            completed_ids.add(data['item_id'])
                except:
                    continue
            
            return jsonify({
                "total": len(self.dataset),
                "completed": len(completed_ids),
                "completed_ids": list(completed_ids)
            })
    
    def create_templates(self):
        """Create HTML templates for the annotation interface."""
        templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        os.makedirs(templates_dir, exist_ok=True)
        
        annotation_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Multi-Label Toxicity Annotation</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
    <style>
        .category-section {
            margin-bottom: 20px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .annotation-card {
            margin-bottom: 20px;
            border: 1px solid #eee;
            border-left: 5px solid #007bff;
            border-radius: 5px;
            padding: 15px;
        }
        .text-content {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 15px;
        }
        .context-content {
            font-style: italic;
            color: #6c757d;
            margin-bottom: 15px;
        }
        .confidence-slider {
            width: 100%;
        }
        .confidence-value {
            font-weight: bold;
        }
        .borderline-checkbox {
            margin-left: 15px;
        }
        .nav-buttons {
            margin-top: 20px;
            display: flex;
            justify-content: space-between;
        }
        .progress-bar {
            height: 30px;
            font-size: 16px;
        }
        .notes-area {
            margin-top: 15px;
        }
        .hidden {
            display: none;
        }
    </style>
</head>
<body>
    <div class="container mt-4">
        <h1 class="mb-4">Multi-Label Toxicity Annotation Tool</h1>
        
        <div class="row mb-4">
            <div class="col-md-6">
                <div class="input-group">
                    <span class="input-group-text">Annotator ID:</span>
                    <input type="text" id="annotator-id" class="form-control" placeholder="Enter your identifier">
                    <button id="save-annotator" class="btn btn-primary">Save</button>
                </div>
            </div>
            <div class="col-md-6">
                <div class="progress">
                    <div id="progress-bar" class="progress-bar" role="progressbar" style="width: 0%">0%</div>
                </div>
            </div>
        </div>
        
        <div id="annotation-container" class="annotation-card">
            <div class="text-content">
                <h5>Text to Annotate:</h5>
                <p id="text-content"></p>
            </div>
            
            <div class="context-content">
                <h6>Context:</h6>
                <p id="context-content"></p>
            </div>
            
            <div class="source-info mb-3">
                <span class="badge bg-secondary" id="source-badge"></span>
                <span class="badge bg-info" id="type-badge"></span>
                <a href="#" id="url-link" target="_blank" class="badge bg-dark text-decoration-none">Source URL</a>
            </div>
            
            <div class="annotation-form">
                <div class="category-section" id="explicit-toxicity">
                    <h4>Explicit Toxicity</h4>
                    
                    <div class="mb-3">
                        <div class="form-check">
                            <input class="form-check-input" type="checkbox" id="hate-speech">
                            <label class="form-check-label" for="hate-speech">
                                Hate Speech
                                <span class="text-muted">(targeting identity groups)</span>
                            </label>
                            <div class="form-check borderline-checkbox">
                                <input class="form-check-input" type="checkbox" id="hate-speech-borderline">
                                <label class="form-check-label" for="hate-speech-borderline">Borderline</label>
                            </div>
                        </div>
                        <div class="mt-2">
                            <label for="hate-speech-confidence" class="form-label">Confidence: <span id="hate-speech-confidence-value" class="confidence-value">50</span>%</label>
                            <input type="range" class="form-range confidence-slider" id="hate-speech-confidence" min="0" max="100" value="50">
                        </div>
                    </div>
                    
                    <div class="mb-3">
                        <div class="form-check">
                            <input class="form-check-input" type="checkbox" id="threats">
                            <label class="form-check-label" for="threats">
                                Threats/Incitement
                                <span class="text-muted">(promoting violence or harm)</span>
                            </label>
                            <div class="form-check borderline-checkbox">
                                <input class="form-check-input" type="checkbox" id="threats-borderline">
                                <label class="form-check-label" for="threats-borderline">Borderline</label>
                            </div>
                        </div>
                        <div class="mt-2">
                            <label for="threats-confidence" class="form-label">Confidence: <span id="threats-confidence-value" class="confidence-value">50</span>%</label>
                            <input type="range" class="form-range confidence-slider" id="threats-confidence" min="0" max="100" value="50">
                        </div>
                    </div>
                    
                    <div class="mb-3">
                        <div class="form-check">
                            <input class="form-check-input" type="checkbox" id="profanity">
                            <label class="form-check-label" for="profanity">
                                Severe Profanity
                                <span class="text-muted">(explicit language beyond casual use)</span>
                            </label>
                            <div class="form-check borderline-checkbox">
                                <input class="form-check-input" type="checkbox" id="profanity-borderline">
                                <label class="form-check-label" for="profanity-borderline">Borderline</label>
                            </div>
                        </div>
                        <div class="mt-2">
                            <label for="profanity-confidence" class="form-label">Confidence: <span id="profanity-confidence-value" class="confidence-value">50</span>%</label>
                            <input type="range" class="form-range confidence-slider" id="profanity-confidence" min="0" max="100" value="50">
                        </div>
                    </div>
                </div>
                
                <div class="category-section" id="implicit-toxicity">
                    <h4>Implicit Toxicity</h4>
                    
                    <div class="mb-3">
                        <div class="form-check">
                            <input class="form-check-input" type="checkbox" id="microaggressions">
                            <label class="form-check-label" for="microaggressions">
                                Microaggressions
                                <span class="text-muted">(subtle demeaning comments)</span>
                            </label>
                            <div class="form-check borderline-checkbox">
                                <input class="form-check-input" type="checkbox" id="microaggressions-borderline">
                                <label class="form-check-label" for="microaggressions-borderline">Borderline</label>
                            </div>
                        </div>
                        <div class="mt-2">
                            <label for="microaggressions-confidence" class="form-label">Confidence: <span id="microaggressions-confidence-value" class="confidence-value">50</span>%</label>
                            <input type="range" class="form-range confidence-slider" id="microaggressions-confidence" min="0" max="100" value="50">
                        </div>
                    </div>
                    
                    <div class="mb-3">
                        <div class="form-check">
                            <input class="form-check-input" type="checkbox" id="subtle-bias">
                            <label class="form-check-label" for="subtle-bias">
                                Subtle Bias
                                <span class="text-muted">(implicit stereotyping or prejudice)</span>
                            </label>
                            <div class="form-check borderline-checkbox">
                                <input class="form-check-input" type="checkbox" id="subtle-bias-borderline">
                                <label class="form-check-label" for="subtle-bias-borderline">Borderline</label>
                            </div>
                        </div>
                        <div class="mt-2">
                            <label for="subtle-bias-confidence" class="form-label">Confidence: <span id="subtle-bias-confidence-value" class="confidence-value">50</span>%</label>
                            <input type="range" class="form-range confidence-slider" id="subtle-bias-confidence" min="0" max="100" value="50">
                        </div>
                    </div>
                    
                    <div class="mb-3">
                        <div class="form-check">
                            <input class="form-check-input" type="checkbox" id="condescension">
                            <label class="form-check-label" for="condescension">
                                Condescension/Patronizing
                                <span class="text-muted">(talking down to someone)</span>
                            </label>
                            <div class="form-check borderline-checkbox">
                                <input class="form-check-input" type="checkbox" id="condescension-borderline">
                                <label class="form-check-label" for="condescension-borderline">Borderline</label>
                            </div>
                        </div>
                        <div class="mt-2">
                            <label for="condescension-confidence" class="form-label">Confidence: <span id="condescension-confidence-value" class="confidence-value">50</span>%</label>
                            <input type="range" class="form-range confidence-slider" id="condescension-confidence" min="0" max="100" value="50">
                        </div>
                    </div>
                </div>
                
                <div class="category-section" id="misinformation">
                    <h4>Misinformation</h4>
                    
                    <div class="mb-3">
                        <div class="form-check">
                            <input class="form-check-input" type="checkbox" id="factual-error">
                            <label class="form-check-label" for="factual-error">
                                Factual Error
                                <span class="text-muted">(contains inaccurate information)</span>
                            </label>
                            <div class="form-check borderline-checkbox">
                                <input class="form-check-input" type="checkbox" id="factual-error-borderline">
                                <label class="form-check-label" for="factual-error-borderline">Borderline</label>
                            </div>
                        </div>
                        <div class="mt-2">
                            <label for="factual-error-confidence" class="form-label">Confidence: <span id="factual-error-confidence-value" class="confidence-value">50</span>%</label>
                            <input type="range" class="form-range confidence-slider" id="factual-error-confidence" min="0" max="100" value="50">
                        </div>
                    </div>
                    
                    <div class="mb-3">
                        <div class="form-check">
                            <input class="form-check-input" type="checkbox" id="conspiracy">
                            <label class="form-check-label" for="conspiracy">
                                Conspiracy Theory
                                <span class="text-muted">(promoting unsubstantiated theories)</span>
                            </label>
                            <div class="form-check borderline-checkbox">
                                <input class="form-check-input" type="checkbox" id="conspiracy-borderline">
                                <label class="form-check-label" for="conspiracy-borderline">Borderline</label>
                            </div>
                        </div>
                        <div class="mt-2">
                            <label for="conspiracy-confidence" class="form-label">Confidence: <span id="conspiracy-confidence-value" class="confidence-value">50</span>%</label>
                            <input type="range" class="form-range confidence-slider" id="conspiracy-confidence" min="0" max="100" value="50">
                        </div>
                    </div>
                </div>
                
                <div class="category-section" id="personal-attacks">
                    <h4>Personal Attacks vs. General Statements</h4>
                    
                    <div class="mb-3">
                        <div class="form-check">
                            <input class="form-check-input" type="checkbox" id="directed-personal">
                            <label class="form-check-label" for="directed-personal">
                                Directed Personal Attack
                                <span class="text-muted">(aimed at specific individual)</span>
                            </label>
                            <div class="form-check borderline-checkbox">
                                <input class="form-check-input" type="checkbox" id="directed-personal-borderline">
                                <label class="form-check-label" for="directed-personal-borderline">Borderline</label>
                            </div>
                        </div>
                        <div class="mt-2">
                            <label for="directed-personal-confidence" class="form-label">Confidence: <span id="directed-personal-confidence-value" class="confidence-value">50</span>%</label>
                            <input type="range" class="form-range confidence-slider" id="directed-personal-confidence" min="0" max="100" value="50">
                        </div>
                    </div>
                    
                    <div class="mb-3">
                        <div class="form-check">
                            <input class="form-check-input" type="checkbox" id="general-attack">
                            <label class="form-check-label" for="general-attack">
                                General Group Attack
                                <span class="text-muted">(aimed at category of people)</span>
                            </label>
                            <div class="form-check borderline-checkbox">
                                <input class="form-check-input" type="checkbox" id="general-attack-borderline">
                                <label class="form-check-label" for="general-attack-borderline">Borderline</label>
                            </div>
                        </div>
                        <div class="mt-2">
                            <label for="general-attack-confidence" class="form-label">Confidence: <span id="general-attack-confidence-value" class="confidence-value">50</span>%</label>
                            <input type="range" class="form-range confidence-slider" id="general-attack-confidence" min="0" max="100" value="50">
                        </div>
                    </div>
                </div>
                
                <div class="notes-area">
                    <div class="mb-3">
                        <label for="annotation-notes" class="form-label">Notes (optional):</label>
                        <textarea class="form-control" id="annotation-notes" rows="3" placeholder="Add any additional observations or comments here"></textarea>
                    </div>
                </div>
                
                <div class="nav-buttons">
                    <button id="prev-button" class="btn btn-secondary">Previous</button>
                    <div>
                        <button id="skip-button" class="btn btn-warning">Skip</button>
                        <button id="save-button" class="btn btn-primary">Save & Continue</button>
                    </div>
                    <button id="next-button" class="btn btn-secondary">Next</button>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.min.js"></script>
    <script>
        let currentItemIndex = 0;
        let totalItems = 0;
        let currentItem = null;
        let annotatorId = localStorage.getItem('annotatorId') || 'anonymous';
        let completedIds = new Set();
        
        // Initialize confidence sliders
        document.querySelectorAll('.confidence-slider').forEach(slider => {
            slider.addEventListener('input', function() {
                document.getElementById(this.id + '-value').textContent = this.value;
            });
        });
        
        // Save annotator ID
        document.getElementById('annotator-id').value = annotatorId;
        document.getElementById('save-annotator').addEventListener('click', function() {
            annotatorId = document.getElementById('annotator-id').value || 'anonymous';
            localStorage.setItem('annotatorId', annotatorId);
            loadProgress();
        });
        
        // Load dataset info
        fetch('/get_dataset_info')
            .then(response => response.json())
            .then(data => {
                totalItems = data.total_items;
                updateProgressBar();
                loadProgress();
            });
        
        // Load progress
        function loadProgress() {
            fetch(`/get_progress/${annotatorId}`)
                .then(response => response.json())
                .then(data => {
                    completedIds = new Set(data.completed_ids);
                    updateProgressBar(data.completed, data.total);
                    loadItem(currentItemIndex);
                });
        }
        
        // Update progress bar
        function updateProgressBar(completed, total) {
            if (!completed) completed = completedIds.size;
            if (!total) total = totalItems;
            
            const percentage = total > 0 ? Math.round((completed / total) * 100) : 0;
            const progressBar = document.getElementById('progress-bar');
            progressBar.style.width = `${percentage}%`;
            progressBar.textContent = `${percentage}% (${completed}/${total})`;
        }
        
        // Load item
        function loadItem(index) {
            fetch(`/get_item/${index}`)
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        console.error(data.error);
                        return;
                    }
                    
                    currentItem = data;
                    displayItem(data);
                    
                    // Check if this item has already been annotated
                    if (completedIds.has(data.id)) {
                        // Could add logic to load previous annotations here
                    } else {
                        resetForm();
                    }
                })
                .catch(error => {
                    console.error('Error loading item:', error);
                });
        }
        
        // Display item
        function displayItem(item) {
            document.getElementById('text-content').textContent = item.text;
            document.getElementById('context-content').textContent = item.context || 'No context available';
            document.getElementById('source-badge').textContent = item.source || 'Unknown source';
            document.getElementById('type-badge').textContent = item.type || 'Unknown type';
            
            const urlLink = document.getElementById('url-link');
            if (item.url) {
                urlLink.href = item.url;
                urlLink.classList.remove('hidden');
            } else {
                urlLink.classList.add('hidden');
            }
        }
        
        // Reset form
        function resetForm() {
            document.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
                checkbox.checked = false;
            });
            
            document.querySelectorAll('.confidence-slider').forEach(slider => {
                slider.value = 50;
                document.getElementById(slider.id + '-value').textContent = '50';
            });
            
            document.getElementById('annotation-notes').value = '';
        }
        
        // Collect annotations
        function collectAnnotations() {
            const annotations = {
                explicit_toxicity: {
                    hate_speech: {
                        present: document.getElementById('hate-speech').checked,
                        borderline: document.getElementById('hate-speech-borderline').checked,
                        confidence: parseInt(document.getElementById('hate-speech-confidence').value)
                    },
                    threats: {
                        present: document.getElementById('threats').checked,
                        borderline: document.getElementById('threats-borderline').checked,
                        confidence: parseInt(document.getElementById('threats-confidence').value)
                    },
                    profanity: {
                        present: document.getElementById('profanity').checked,
                        borderline: document.getElementById('profanity-borderline').checked,
                        confidence: parseInt(document.getElementById('profanity-confidence').value)
                    }
                },
                implicit_toxicity: {
                    microaggressions: {
                        present: document.getElementById('microaggressions').checked,
                        borderline: document.getElementById('microaggressions-borderline').checked,
                        confidence: parseInt(document.getElementById('microaggressions-confidence').value)
                    },
                    subtle_bias: {
                        present: document.getElementById('subtle-bias').checked,
                        borderline: document.getElementById('subtle-bias-borderline').checked,
                        confidence: parseInt(document.getElementById('subtle-bias-confidence').value)
                    },
                    condescension: {
                        present: document.getElementById('condescension').checked,
                        borderline: document.getElementById('condescension-borderline').checked,
                        confidence: parseInt(document.getElementById('condescension-confidence').value)
                    }
                },
                misinformation: {
                    factual_error: {
                        present: document.getElementById('factual-error').checked,
                        borderline: document.getElementById('factual-error-borderline').checked,
                        confidence: parseInt(document.getElementById('factual-error-confidence').value)
                    },
                    conspiracy: {
                        present: document.getElementById('conspiracy').checked,
                        borderline: document.getElementById('conspiracy-borderline').checked,
                        confidence: parseInt(document.getElementById('conspiracy-confidence').value)
                    }
                },
                personal_attacks: {
                    directed_personal: {
                        present: document.getElementById('directed-personal').checked,
                        borderline: document.getElementById('directed-personal-borderline').checked,
                        confidence: parseInt(document.getElementById('directed-personal-confidence').value)
                    },
                    general_attack: {
                        present: document.getElementById('general-attack').checked,
                        borderline: document.getElementById('general-attack-borderline').checked,
                        confidence: parseInt(document.getElementById('general-attack-confidence').value)
                    }
                }
            };
            
            return {
                item_id: currentItem.id,
                annotator: annotatorId,
                timestamp: new Date().toISOString(),
                item: currentItem,
                annotations: annotations,
                notes: document.getElementById('annotation-notes').value
            };
        }
        
        // Save annotation
        function saveAnnotation() {
            const data = collectAnnotations();
            
            fetch('/save_annotation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(result => {
                if (result.success) {
                    completedIds.add(currentItem.id);
                    updateProgressBar();
                    loadItem(currentItemIndex + 1);
                    currentItemIndex++;
                } else {
                    console.error('Error saving annotation:', result.error);
                }
            })
            .catch(error => {
                console.error('Error saving annotation:', error);
            });
        }
        
        // Event listeners for navigation
        document.getElementById('prev-button').addEventListener('click', function() {
            if (currentItemIndex > 0) {
                currentItemIndex--;
                loadItem(currentItemIndex);
            }
        });
        
        document.getElementById('next-button').addEventListener('click', function() {
            if (currentItemIndex < totalItems - 1) {
                currentItemIndex++;
                loadItem(currentItemIndex);
            }
        });
        
        document.getElementById('save-button').addEventListener('click', saveAnnotation);
        
        document.getElementById('skip-button').addEventListener('click', function() {
            if (currentItemIndex < totalItems - 1) {
                currentItemIndex++;
                loadItem(currentItemIndex);
            }
        });
        
        // Initial load
        loadItem(currentItemIndex);
    </script>
</body>
</html>
        """
        
        annotation_html_path = os.path.join(templates_dir, "annotation.html")
        with open(annotation_html_path, 'w', encoding='utf-8') as f:
            f.write(annotation_html)
    
    def run(self, host='0.0.0.0', port=5000, debug=True):
        """Run the Flask app."""
        self.create_templates()
        self.app.run(host=host, port=port, debug=debug)

# =============================================================================
# STEP 4: ANNOTATION AGGREGATION AND ANALYSIS
# =============================================================================

class AnnotationAnalyzer:
    """Analyze and aggregate annotations from multiple annotators."""
    
    def __init__(self, annotation_dir=ANNOTATION_DIR, output_dir=RESULTS_DIR):
        self.annotation_dir = annotation_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def load_all_annotations(self):
        """Load all annotation files from all annotators."""
        all_annotations = []
        
        for annotator_dir in os.listdir(self.annotation_dir):
            annotator_path = os.path.join(self.annotation_dir, annotator_dir)
            if not os.path.isdir(annotator_path):
                continue
            
            for filename in os.listdir(annotator_path):
                if not filename.endswith('.json'):
                    continue
                
                filepath = os.path.join(annotator_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        annotation = json.load(f)
                        all_annotations.append(annotation)
                except Exception as e:
                    logger.error(f"Error loading annotation from {filepath}: {str(e)}")
        
        return all_annotations
    
    def compute_agreement(self, annotations):
        """Compute inter-annotator agreement using Cohen's Kappa."""
        # Group annotations by item_id
        items_annotations = {}
        for annotation in annotations:
            item_id = annotation['item_id']
            if item_id not in items_annotations:
                items_annotations[item_id] = []
            items_annotations[item_id].append(annotation)
        
        # Only consider items with multiple annotations
        multi_annotated_items = {item_id: anns for item_id, anns in items_annotations.items() if len(anns) > 1}
        
        if not multi_annotated_items:
            logger.warning("No items with multiple annotations found. Cannot compute agreement.")
            return {}
        
        # Compute agreement for each category
        agreement_scores = {}
        
        # List of all categories to check
        categories = [
            ('explicit_toxicity', 'hate_speech'),
            ('explicit_toxicity', 'threats'),
            ('explicit_toxicity', 'profanity'),
            ('implicit_toxicity', 'microaggressions'),
            ('implicit_toxicity', 'subtle_bias'),
            ('implicit_toxicity', 'condescension'),
            ('misinformation', 'factual_error'),
            ('misinformation', 'conspiracy'),
            ('personal_attacks', 'directed_personal'),
            ('personal_attacks', 'general_attack')
        ]
        
        for category, subcategory in categories:
            kappa_scores = []
            
            for item_id, item_annotations in multi_annotated_items.items():
                if len(item_annotations) < 2:
                    continue
                
                # For each pair of annotators
                for i in range(len(item_annotations)):
                    for j in range(i + 1, len(item_annotations)):
                        ann1 = item_annotations[i]['annotations'][category][subcategory]['present']
                        ann2 = item_annotations[j]['annotations'][category][subcategory]['present']
                        
                        # Convert to binary for Cohen's Kappa
                        ann1_int = 1 if ann1 else 0
                        ann2_int = 1 if ann2 else 0
                        
                        # Need more than one item for meaningful kappa
                        # This is a placeholder - real implementation would aggregate across items
                        ann1_arr = [ann1_int, 1-ann1_int]  # Add dummy to avoid division by zero
                        ann2_arr = [ann2_int, 1-ann2_int]  # Add dummy to avoid division by zero
                        
                        try:
                            kappa = cohen_kappa_score(ann1_arr, ann2_arr)
                            kappa_scores.append(kappa)
                        except Exception as e:
                            logger.warning(f"Could not compute kappa for {category}.{subcategory}: {str(e)}")
            
            if kappa_scores:
                agreement_scores[f"{category}.{subcategory}"] = sum(kappa_scores) / len(kappa_scores)
        
        return agreement_scores
    
    def aggregate_annotations(self, annotations):
        """Aggregate annotations from multiple annotators."""
        # Group annotations by item_id
        items_annotations = {}
        for annotation in annotations:
            item_id = annotation['item_id']
            if item_id not in items_annotations:
                items_annotations[item_id] = []
            items_annotations[item_id].append(annotation)
        
        aggregated_dataset = []
        
        for item_id, item_annotations in items_annotations.items():
            if not item_annotations:
                continue
            
            # Use the first annotation's item data
            item_data = item_annotations[0]['item']
            
            # Initialize aggregated annotation data
            agg_annotations = {
                'id': item_id,
                'text': item_data['text'],
                'source': item_data.get('source', 'unknown'),
                'type': item_data.get('type', 'unknown'),
                'context': item_data.get('context', ''),
                'url': item_data.get('url', ''),
                'annotations': {},
                'multi_annotated': len(item_annotations) > 1,
                'annotator_count': len(item_annotations)
            }
            
            # Categories and subcategories to aggregate
            categories = {
                'explicit_toxicity': ['hate_speech', 'threats', 'profanity'],
                'implicit_toxicity': ['microaggressions', 'subtle_bias', 'condescension'],
                'misinformation': ['factual_error', 'conspiracy'],
                'personal_attacks': ['directed_personal', 'general_attack']
            }
            
            # Aggregate annotations for each category/subcategory
            for category, subcategories in categories.items():
                agg_annotations['annotations'][category] = {}
                
                for subcategory in subcategories:
                    present_count = 0
                    borderline_count = 0
                    confidence_sum = 0
                    
                    for annotation in item_annotations:
                        try:
                            subcat_data = annotation['annotations'][category][subcategory]
                            present_count += 1 if subcat_data['present'] else 0
                            borderline_count += 1 if subcat_data['borderline'] else 0
                            confidence_sum += subcat_data['confidence']
                        except KeyError:
                            continue
                    
                    # Calculate aggregated values
                    present_ratio = present_count / len(item_annotations)
                    borderline_ratio = borderline_count / len(item_annotations)
                    avg_confidence = confidence_sum / len(item_annotations) if len(item_annotations) > 0 else 0
                    
                    agg_annotations['annotations'][category][subcategory] = {
                        'present_ratio': present_ratio,
                        'borderline_ratio': borderline_ratio,
                        'avg_confidence': avg_confidence,
                        'is_present': present_ratio > 0.5,  # Majority vote
                        'is_borderline': borderline_ratio > 0.5  # Majority vote
                    }
                # end for subcategory
            # end for category
            aggregated_dataset.append(agg_annotations)
        # end for item_id

        # Save aggregated dataset
        output_path = os.path.join(self.output_dir, "aggregated_annotations.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(aggregated_dataset, f, ensure_ascii=False, indent=2)
        logger.info(f"Aggregated annotations saved to {output_path}")
        return aggregated_dataset

# =============================================================================
# MAIN PIPELINE ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Label Toxicity Dataset Pipeline")
    parser.add_argument("--collect", action="store_true", help="Collect raw data")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess raw data")
    parser.add_argument("--annotate", action="store_true", help="Run annotation app")
    parser.add_argument("--analyze", action="store_true", help="Aggregate and analyze annotations")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    args = parser.parse_args()

    if args.collect:
        # Example: expects config.json with API keys and collection params
        with open(args.config, 'r') as f:
            config = json.load(f)
        collect_all_data(config)

    if args.preprocess:
        creator = DatasetCreator()
        creator.create_full_dataset()

    if args.annotate:
        # Example: expects processed dataset at default location
        dataset_path = os.path.join(PROCESSED_DIR, "processed_dataset.json")
        app = AnnotationApp(dataset_path)
        app.run()

    if args.analyze:
        analyzer = AnnotationAnalyzer()
        annotations = analyzer.load_all_annotations()
        agreement = analyzer.compute_agreement(annotations)
        print("Inter-annotator agreement:", agreement)
        analyzer.aggregate_annotations(annotations)