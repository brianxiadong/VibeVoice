import requests
import json
from typing import List, Dict, Any
from datetime import datetime, timedelta
import logging
import time

logger = logging.getLogger(__name__)

class NewsFetcher:
    """Fetches hot news from various sources"""
    
    def __init__(self, hours_filter: int = 24):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.hours_filter = hours_filter  # Filter news within this many hours
        self.cutoff_timestamp = time.time() - (hours_filter * 3600)
    
    def fetch_hacker_news_top(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch top stories from Hacker News with time filtering"""
        try:
            # Get top story IDs
            response = self.session.get('https://hacker-news.firebaseio.com/v0/topstories.json')
            story_ids = response.json()[:limit * 3]  # Get more to account for filtering
            
            stories = []
            for story_id in story_ids:
                if len(stories) >= limit:
                    break
                    
                story_response = self.session.get(f'https://hacker-news.firebaseio.com/v0/item/{story_id}.json')
                story_data = story_response.json()
                
                if story_data and story_data.get('title'):
                    # Check if story is within time filter
                    story_time = story_data.get('time', 0)
                    if story_time >= self.cutoff_timestamp:
                        stories.append({
                            'title': story_data.get('title', ''),
                            'url': story_data.get('url', ''),
                            'score': story_data.get('score', 0),
                            'text': story_data.get('text', ''),
                            'source': 'Hacker News',
                            'timestamp': story_time,
                            'published_time': datetime.fromtimestamp(story_time).strftime('%Y-%m-%d %H:%M:%S')
                        })
            
            logger.info(f"Fetched {len(stories)} recent Hacker News stories (within {self.hours_filter} hours)")
            return stories
            
        except Exception as e:
            logger.error(f"Error fetching Hacker News: {e}")
            return []
    
    def fetch_reddit_hot(self, subreddit: str = 'worldnews', limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch hot posts from Reddit with time filtering"""
        try:
            url = f'https://www.reddit.com/r/{subreddit}/hot.json?limit={limit * 3}'  # Get more to account for filtering
            response = self.session.get(url)
            data = response.json()
            
            stories = []
            for post in data.get('data', {}).get('children', []):
                if len(stories) >= limit:
                    break
                    
                post_data = post.get('data', {})
                created_utc = post_data.get('created_utc', 0)
                
                # Check if post is within time filter
                if created_utc >= self.cutoff_timestamp:
                    stories.append({
                        'title': post_data.get('title', ''),
                        'url': post_data.get('url', ''),
                        'score': post_data.get('score', 0),
                        'text': post_data.get('selftext', ''),
                        'source': f'Reddit r/{subreddit}',
                        'timestamp': created_utc,
                        'published_time': datetime.fromtimestamp(created_utc).strftime('%Y-%m-%d %H:%M:%S')
                    })
            
            logger.info(f"Fetched {len(stories)} recent Reddit r/{subreddit} posts (within {self.hours_filter} hours)")
            return stories
            
        except Exception as e:
            logger.error(f"Error fetching Reddit: {e}")
            return []
    

    
    def fetch_all_news(self, limit_per_source: int = 5) -> List[Dict[str, Any]]:
        """Fetch news from all sources (Hacker News and Reddit)"""
        all_news = []
        
        # Fetch from different sources - only time-sensitive news sources
        hacker_news = self.fetch_hacker_news_top(limit_per_source)
        reddit_tech = self.fetch_reddit_hot('technology', limit_per_source)
        reddit_world = self.fetch_reddit_hot('worldnews', limit_per_source)
        
        all_news.extend(hacker_news)
        all_news.extend(reddit_tech)
        all_news.extend(reddit_world)
        
        # Sort by score/popularity
        all_news.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return all_news
    
    def get_today_hot_news(self, total_limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent hot news with deduplication and time filtering"""
        news = self.fetch_all_news()
        
        # Filter by timestamp and sort by recency
        recent_news = [item for item in news if item.get('timestamp', 0) >= self.cutoff_timestamp]
        recent_news.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        
        # Simple deduplication based on title similarity
        unique_news = []
        seen_titles = set()
        
        for item in recent_news:
            title_lower = item.get('title', '').lower()
            # Simple check for similar titles
            is_duplicate = any(
                self._similar_titles(title_lower, seen_title) 
                for seen_title in seen_titles
            )
            
            if not is_duplicate and len(unique_news) < total_limit:
                unique_news.append(item)
                seen_titles.add(title_lower)
        
        logger.info(f"Filtered to {len(unique_news)} unique recent news items (within {self.hours_filter} hours)")
        return unique_news
    
    def select_best_news(self, news_items: List[Dict[str, Any]], count: int = 3) -> List[Dict[str, Any]]:
        """Select the best N news items based on score, recency, and content quality"""
        if not news_items:
            return []
        
        if len(news_items) <= count:
            return news_items
        
        # Calculate composite score for each news item
        scored_news = []
        current_time = time.time()
        
        for item in news_items:
            # Base score from the source (upvotes, etc.)
            base_score = item.get('score', 0)
            
            # Recency score (more recent = higher score)
            timestamp = item.get('timestamp', 0)
            hours_old = (current_time - timestamp) / 3600 if timestamp > 0 else 999
            recency_score = max(0, 24 - hours_old) / 24  # Normalize to 0-1
            
            # Content quality score based on title length and text availability
            title = item.get('title', '')
            text = item.get('text', '')
            
            # Prefer titles that are not too short or too long
            title_length_score = 1.0
            if len(title) < 20:
                title_length_score = 0.5
            elif len(title) > 150:
                title_length_score = 0.7
            
            # Bonus for having additional text content
            text_bonus = 0.2 if text and len(text) > 50 else 0
            
            # Composite score calculation
            composite_score = (
                base_score * 0.4 +  # 40% weight on original score
                recency_score * 100 * 0.3 +  # 30% weight on recency
                title_length_score * 50 * 0.2 +  # 20% weight on title quality
                text_bonus * 50 * 0.1  # 10% weight on text availability
            )
            
            scored_news.append({
                **item,
                'composite_score': composite_score,
                'hours_old': hours_old
            })
        
        # Sort by composite score and select top N
        scored_news.sort(key=lambda x: x['composite_score'], reverse=True)
        selected_news = scored_news[:count]
        
        logger.info(f"Selected {len(selected_news)} best news items from {len(news_items)} candidates")
        for i, item in enumerate(selected_news, 1):
            logger.info(f"  {i}. {item['title'][:60]}... (Score: {item['composite_score']:.1f}, {item['hours_old']:.1f}h old)")
        
        return selected_news
    
    def _similar_titles(self, title1: str, title2: str, threshold: float = 0.7) -> bool:
        """Check if two titles are similar"""
        # Simple word-based similarity check
        words1 = set(title1.split())
        words2 = set(title2.split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0
        return similarity >= threshold

if __name__ == "__main__":
    # Test the news fetcher with different time filters
    print("Testing with 24-hour filter:")
    fetcher = NewsFetcher(hours_filter=24)
    news = fetcher.get_today_hot_news(10)
    
    for i, item in enumerate(news, 1):
        published_time = item.get('published_time', 'Unknown')
        print(f"{i}. {item['title']} (Source: {item['source']}, Score: {item['score']})")
        print(f"   Published: {published_time}")
        if item.get('text'):
            print(f"   {item['text'][:100]}...")
        print()
    
    print(f"\nTotal recent news found: {len(news)}")
    
    # Test with 48-hour filter for comparison
    print("\n" + "="*50)
    print("Testing with 48-hour filter:")
    fetcher_48h = NewsFetcher(hours_filter=48)
    news_48h = fetcher_48h.get_today_hot_news(10)
    print(f"Total news found with 48h filter: {len(news_48h)}")
    
    # Show time distribution
    if news_48h:
        timestamps = [item.get('timestamp', 0) for item in news_48h if item.get('timestamp', 0) > 0]
        if timestamps:
            oldest = min(timestamps)
            newest = max(timestamps)
            print(f"Time range: {datetime.fromtimestamp(oldest)} to {datetime.fromtimestamp(newest)}")