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
    
    def fetch_github_trending(self, language: str = '', limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch trending repositories from GitHub with time filtering"""
        try:
            # Use GitHub's search API for trending repos
            today = datetime.now()
            hours_ago = today - timedelta(hours=self.hours_filter)
            date_filter = hours_ago.strftime('%Y-%m-%dT%H:%M:%S')
            
            query = f"created:>{date_filter}"
            if language:
                query += f" language:{language}"
            
            url = f'https://api.github.com/search/repositories'
            params = {
                'q': query,
                'sort': 'stars',
                'order': 'desc',
                'per_page': limit
            }
            
            response = self.session.get(url, params=params)
            data = response.json()
            
            stories = []
            for repo in data.get('items', []):
                created_at = repo.get('created_at', '')
                stories.append({
                    'title': f"{repo.get('name', '')} - {repo.get('description', '')}",
                    'url': repo.get('html_url', ''),
                    'score': repo.get('stargazers_count', 0),
                    'text': repo.get('description', ''),
                    'source': 'GitHub Trending',
                    'timestamp': datetime.fromisoformat(created_at.replace('Z', '+00:00')).timestamp() if created_at else 0,
                    'published_time': created_at
                })
            
            logger.info(f"Fetched {len(stories)} recent GitHub trending repos (within {self.hours_filter} hours)")
            return stories
            
        except Exception as e:
            logger.error(f"Error fetching GitHub trending: {e}")
            return []
    
    def fetch_all_news(self, limit_per_source: int = 5) -> List[Dict[str, Any]]:
        """Fetch news from all sources"""
        all_news = []
        
        # Fetch from different sources
        hacker_news = self.fetch_hacker_news_top(limit_per_source)
        reddit_tech = self.fetch_reddit_hot('technology', limit_per_source)
        reddit_world = self.fetch_reddit_hot('worldnews', limit_per_source)
        github_trending = self.fetch_github_trending(limit=limit_per_source)
        
        all_news.extend(hacker_news)
        all_news.extend(reddit_tech)
        all_news.extend(reddit_world)
        all_news.extend(github_trending)
        
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