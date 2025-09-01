import argparse
import os
import logging
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from .news_fetcher import NewsFetcher
from .ollama_processor import OllamaProcessor
from .audio_generator import PodcastAudioGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NewsPodcastGenerator:
    """Main class for generating news podcasts"""
    
    def __init__(
        self,
        model_path: str = "microsoft/VibeVoice-1.5B",
        ollama_url: str = "http://172.36.237.245:11434",
        ollama_model: str = "qwen2.5-coder:1.5b",
        device: str = "cuda",
        hours_filter: int = 24
    ):
        self.model_path = model_path
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.device = device
        self.hours_filter = hours_filter
        
        # Initialize components
        self.news_fetcher = NewsFetcher(hours_filter=hours_filter)
        self.ollama_processor = OllamaProcessor(ollama_url, ollama_model)
        self.audio_generator = None  # Initialize lazily to avoid loading model unless needed
        
        # Check connections
        self._check_connections()
    
    def _check_connections(self):
        """Check if all services are available"""
        logger.info("Checking service connections...")
        
        # Check Ollama connection
        if self.ollama_processor.check_ollama_connection():
            logger.info(f"✓ Ollama connection successful ({self.ollama_url})")
            models = self.ollama_processor.list_available_models()
            logger.info(f"Available models: {models}")
            if self.ollama_model not in [model.split(':')[0] for model in models]:
                logger.warning(f"Specified model '{self.ollama_model}' not found in available models")
        else:
            logger.error(f"✗ Cannot connect to Ollama at {self.ollama_url}")
            raise ConnectionError(f"Cannot connect to Ollama at {self.ollama_url}")
    
    def _initialize_audio_generator(self):
        """Initialize audio generator only when needed"""
        if self.audio_generator is None:
            logger.info(f"Initializing VibeVoice model on device: {self.device}...")
            self.audio_generator = PodcastAudioGenerator(self.model_path, device=self.device)
    
    def fetch_news(self, news_limit: int = 15) -> list:
        """Fetch today's hot news"""
        logger.info("Fetching today's hot news...")
        news = self.news_fetcher.get_today_hot_news(news_limit)
        logger.info(f"Found {len(news)} news items")
        return news
    
    def process_news_to_podcast(
        self,
        news_items: list,
        num_speakers: int = 2,
        max_news_items: int = 10
    ) -> str:
        """Process news items into podcast dialogue"""
        logger.info(f"Processing {len(news_items)} news items into podcast dialogue...")
        
        # Summarize news
        summary = self.ollama_processor.summarize_news(news_items, max_news_items)
        logger.info("News summarization completed")
        
        # Create dialogue
        dialogue = self.ollama_processor.create_podcast_dialogue(summary, num_speakers)
        logger.info(f"Created dialogue with {num_speakers} speakers")
        
        # Enhance for audio
        enhanced_dialogue = self.ollama_processor.enhance_for_audio(dialogue)
        logger.info("Dialogue enhancement completed")
        
        return enhanced_dialogue
    
    def generate_audio(
        self,
        dialogue: str,
        output_path: str,
        voice_preferences: Optional[Dict[str, str]] = None
    ) -> bool:
        """Generate audio from dialogue"""
        logger.info(f"Generating audio to {output_path}...")
        
        self._initialize_audio_generator()
        
        success = self.audio_generator.generate_podcast_audio(
            dialogue, output_path, voice_preferences
        )
        
        if success:
            logger.info(f"✓ Audio generated successfully: {output_path}")
        else:
            logger.error("✗ Audio generation failed")
        
        return success
    
    def generate_single_news_podcast(
        self,
        news_item: Dict[str, Any],
        output_dir: str,
        timestamp: str,
        num_speakers: int = 2,
        voice_preferences: Optional[Dict[str, str]] = None,
        save_intermediate: bool = True
    ) -> Dict[str, Any]:
        """Generate podcast for a single news item"""
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename based on news title and timestamp
        safe_title = "".join(c for c in news_item['title'][:50] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = safe_title.replace(' ', '_')
        base_filename = f"{timestamp}_{safe_title}"
        
        results = {
            'timestamp': timestamp,
            'success': False,
            'files': {},
            'stats': {},
            'news_item': news_item
        }
        
        try:
            # Process single news item to dialogue
            logger.info(f"Processing news: {news_item['title'][:60]}...")
            dialogue = self.process_news_to_podcast([news_item], num_speakers, 1)
            results['stats']['speakers'] = num_speakers
            
            if not dialogue:
                logger.error("Failed to generate dialogue")
                return results
            
            # Save dialogue if requested
            if save_intermediate:
                dialogue_file = output_path / f"{base_filename}_dialogue.txt"
                with open(dialogue_file, 'w', encoding='utf-8') as f:
                    f.write(dialogue)
                results['files']['dialogue'] = str(dialogue_file)
                logger.info(f"Dialogue saved to: {dialogue_file}")
            
            # Generate Chinese translation
            logger.info("Generating Chinese translation...")
            chinese_dialogue = self.ollama_processor.translate_to_chinese(dialogue)
            
            if chinese_dialogue and save_intermediate:
                chinese_dialogue_file = output_path / f"{base_filename}_dialogue_chinese.txt"
                with open(chinese_dialogue_file, 'w', encoding='utf-8') as f:
                    f.write(chinese_dialogue)
                results['files']['chinese_dialogue'] = str(chinese_dialogue_file)
                logger.info(f"Chinese dialogue saved to: {chinese_dialogue_file}")
            
            # Generate audio
            audio_file = output_path / f"{base_filename}.wav"
            audio_success = self.generate_audio(dialogue, str(audio_file), voice_preferences)
            
            if audio_success:
                results['files']['audio'] = str(audio_file)
                results['success'] = True
                
                # Get audio file size
                if audio_file.exists():
                    results['stats']['audio_size_mb'] = round(audio_file.stat().st_size / 1024 / 1024, 2)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in single news podcast generation: {e}")
            results['error'] = str(e)
            return results
    
    def generate_multiple_podcasts(
        self,
        output_dir: str = "./output",
        num_speakers: int = 2,
        news_limit: int = 15,
        podcast_count: int = 3,
        voice_preferences: Optional[Dict[str, str]] = None,
        save_intermediate: bool = True
    ) -> Dict[str, Any]:
        """Generate multiple individual podcasts, one for each selected news item"""
        
        # Create base output directory
        base_output_path = Path(output_dir)
        base_output_path.mkdir(exist_ok=True)
        
        overall_results = {
            'success': False,
            'podcasts': [],
            'stats': {
                'total_requested': podcast_count,
                'total_generated': 0,
                'total_successful': 0
            }
        }
        
        try:
            # Step 1: Fetch and select best news
            logger.info(f"Fetching news and selecting best {podcast_count} items...")
            all_news = self.fetch_news(news_limit)
            
            if not all_news:
                logger.error("No news items found")
                return overall_results
            
            # Select best news items
            selected_news = self.news_fetcher.select_best_news(all_news, podcast_count)
            overall_results['stats']['news_available'] = len(all_news)
            overall_results['stats']['news_selected'] = len(selected_news)
            
            if not selected_news:
                logger.error("No suitable news items selected")
                return overall_results
            
            # Step 2: Generate individual podcasts
            for i, news_item in enumerate(selected_news, 1):
                logger.info(f"\n=== Generating podcast {i}/{len(selected_news)} ===")
                
                # Create timestamp with slight delay to ensure uniqueness
                import time
                time.sleep(1)  # Ensure different timestamps
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Create individual output directory with timestamp
                individual_output_dir = base_output_path / timestamp
                
                # Generate single podcast
                podcast_result = self.generate_single_news_podcast(
                    news_item=news_item,
                    output_dir=str(individual_output_dir),
                    timestamp=timestamp,
                    num_speakers=num_speakers,
                    voice_preferences=voice_preferences,
                    save_intermediate=save_intermediate
                )
                
                overall_results['podcasts'].append(podcast_result)
                overall_results['stats']['total_generated'] += 1
                
                if podcast_result['success']:
                    overall_results['stats']['total_successful'] += 1
                    logger.info(f"✓ Podcast {i} generated successfully")
                else:
                    logger.error(f"✗ Podcast {i} generation failed")
            
            # Update overall success status
            overall_results['success'] = overall_results['stats']['total_successful'] > 0
            
            return overall_results
            
        except Exception as e:
            logger.error(f"Error in multiple podcast generation: {e}")
            overall_results['error'] = str(e)
            return overall_results
    
    def generate_full_podcast(
        self,
        output_dir: str = "./output",
        num_speakers: int = 2,
        news_limit: int = 15,
        max_news_items: int = 10,
        voice_preferences: Optional[Dict[str, str]] = None,
        save_intermediate: bool = True,
        count: int = 3
    ) -> Dict[str, Any]:
        """Generate complete podcast with news fetching, processing, and audio generation"""
        try:
            logger.info("Starting full podcast generation...")
            
            # Check connections first
            if not self._check_connections():
                return {"success": False, "error": "Service connections failed"}
            
            # Fetch news
            logger.info("Fetching news...")
            news_items = self.fetch_news(news_limit)
            if not news_items:
                return {"success": False, "error": "No news items found"}
            
            logger.info(f"Found {len(news_items)} news items")
            
            # Select best news items
            selected_news = self.news_fetcher.select_best_news(news_items, count)
            logger.info(f"Selected {len(selected_news)} best news items for podcast generation")
            
            # Generate multiple podcasts
            podcast_results = self.generate_multiple_podcasts(
                output_dir=output_dir,
                num_speakers=num_speakers,
                news_limit=news_limit,
                podcast_count=count,
                voice_preferences=voice_preferences,
                save_intermediate=save_intermediate
            )
            
            if not podcast_results.get('podcasts'):
                return {"success": False, "error": "Failed to generate any podcasts"}
            
            # Count successful podcasts
            successful_podcasts = [p for p in podcast_results['podcasts'] if p.get('success', False)]
            logger.info(f"Successfully generated {len(successful_podcasts)} out of {len(podcast_results['podcasts'])} podcasts")
            
            # Copy files to daily repo with new structure
            if successful_podcasts:
                copy_success = self._copy_multiple_podcasts_to_daily_repo(successful_podcasts)
                if copy_success:
                    # Perform git operations
                    target_dir = Path("/root/news/english-news-daily")
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    git_success = self._git_commit_and_push(target_dir, timestamp)
                    
                    return {
                        "success": True,
                        "podcasts_generated": len(successful_podcasts),
                        "total_attempted": len(podcast_results['podcasts']),
                        "files_copied": copy_success,
                        "git_pushed": git_success,
                        "results": successful_podcasts
                    }
                else:
                    logger.warning("Failed to copy files to daily repo")
                    return {
                        "success": True,
                        "podcasts_generated": len(successful_podcasts),
                        "total_attempted": len(podcast_results['podcasts']),
                        "files_copied": False,
                        "git_pushed": False,
                        "results": successful_podcasts
                    }
            else:
                return {"success": False, "error": "No successful podcasts generated"}
            
        except Exception as e:
            logger.error(f"Error in full podcast generation: {e}")
            return {"success": False, "error": str(e)}
    
    def get_available_voices(self) -> Dict[str, str]:
        """Get available voice options"""
        self._initialize_audio_generator()
        return self.audio_generator.get_available_voices()
    
    def _copy_files_to_daily_repo(self, files: Dict[str, str], timestamp: str) -> bool:
        """Copy generated files to /root/news/english-news-daily with date-based organization"""
        try:
            # Get current date for folder organization
            current_date = datetime.now().strftime("%Y-%m-%d")
            year_month = datetime.now().strftime("%Y-%m")
            
            # Define target directory with two-level structure: year-month/year-month-day
            target_base_dir = Path("/root/news/english-news-daily")
            target_month_dir = target_base_dir / year_month
            target_date_dir = target_month_dir / current_date
            
            # Create target directory if it doesn't exist
            target_date_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created/verified target directory: {target_date_dir}")
            
            # Copy files
            copied_files = {}
            for file_type, file_path in files.items():
                if file_path and Path(file_path).exists():
                    source_file = Path(file_path)
                    # Create new filename with timestamp prefix
                    new_filename = f"{timestamp}_{source_file.name}"
                    target_file = target_date_dir / new_filename
                    
                    # Copy file
                    shutil.copy2(source_file, target_file)
                    copied_files[file_type] = str(target_file)
                    logger.info(f"Copied {file_type}: {source_file} -> {target_file}")
                else:
                    logger.warning(f"Source file not found for {file_type}: {file_path}")
            
            return len(copied_files) > 0
            
        except Exception as e:
            logger.error(f"Error copying files to daily repo: {e}")
            return False
    
    def _copy_multiple_podcasts_to_daily_repo(self, podcast_results: list) -> bool:
        """Copy multiple podcast files to /root/news/english-news-daily with timestamp-based organization"""
        try:
            # Get current date for folder organization
            current_date = datetime.now().strftime("%Y-%m-%d")
            year_month = datetime.now().strftime("%Y-%m")
            
            # Define target directory with two-level structure: year-month/year-month-day
            target_base_dir = Path("/root/news/english-news-daily")
            target_month_dir = target_base_dir / year_month
            target_date_dir = target_month_dir / current_date
            
            # Create target directory if it doesn't exist
            target_date_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created/verified target directory: {target_date_dir}")
            
            total_copied = 0
            
            # Copy files for each podcast
            for i, podcast_result in enumerate(podcast_results, 1):
                if not podcast_result.get('success', False):
                    logger.warning(f"Skipping failed podcast {i}")
                    continue
                
                timestamp = podcast_result.get('timestamp', '')
                files = podcast_result.get('files', {})
                
                # Create timestamp-based subdirectory
                timestamp_dir = target_date_dir / timestamp
                timestamp_dir.mkdir(exist_ok=True)
                
                # Copy files for this podcast
                for file_type, file_path in files.items():
                    if file_path and Path(file_path).exists():
                        source_file = Path(file_path)
                        target_file = timestamp_dir / source_file.name
                        
                        # Copy file
                        shutil.copy2(source_file, target_file)
                        total_copied += 1
                        logger.info(f"Copied podcast {i} {file_type}: {source_file} -> {target_file}")
                    else:
                        logger.warning(f"Source file not found for podcast {i} {file_type}: {file_path}")
                
                # Also save news metadata
                news_item = podcast_result.get('news_item', {})
                if news_item:
                    metadata_file = timestamp_dir / f"{timestamp}_metadata.json"
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump(news_item, f, indent=2, ensure_ascii=False)
                    total_copied += 1
                    logger.info(f"Saved podcast {i} metadata: {metadata_file}")
            
            return total_copied > 0
            
        except Exception as e:
            logger.error(f"Error copying multiple podcasts to daily repo: {e}")
            return False
    
    def _git_commit_and_push(self, target_dir: Path, timestamp: str) -> bool:
        """Perform git add, commit, and push operations"""
        try:
            # Change to target directory
            original_cwd = os.getcwd()
            os.chdir(target_dir)
            
            # Check if it's a git repository
            result = subprocess.run(['git', 'status'], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Target directory is not a git repository: {target_dir}")
                return False
            
            # Git add all changes
            result = subprocess.run(['git', 'add', '.'], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Git add failed: {result.stderr}")
                return False
            
            # Check if there are changes to commit
            result = subprocess.run(['git', 'diff', '--cached', '--quiet'], capture_output=True)
            if result.returncode == 0:
                logger.info("No changes to commit")
                return True
            
            # Git commit
            commit_message = f"Add news podcast for {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ({timestamp})"
            result = subprocess.run(['git', 'commit', '-m', commit_message], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Git commit failed: {result.stderr}")
                return False
            
            logger.info(f"Git commit successful: {commit_message}")
            
            # Git push
            result = subprocess.run(['git', 'push'], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Git push failed: {result.stderr}")
                return False
            
            logger.info("Git push successful")
            return True
            
        except Exception as e:
            logger.error(f"Error in git operations: {e}")
            return False
        finally:
            # Restore original working directory
            os.chdir(original_cwd)

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Generate news podcast using VibeVoice")
    
    parser.add_argument(
        "--model-path", 
        default="microsoft/VibeVoice-1.5B",
        help="Path to VibeVoice model (default: microsoft/VibeVoice-1.5B)"
    )
    parser.add_argument(
        "--ollama-url",
        default="http://172.36.237.245:11434",
        help="Ollama server URL (default: http://172.36.237.245:11434)"
    )
    parser.add_argument(
        "--ollama-model",
        default="qwen2.5-coder:1.5b",
        help="Ollama model name (default: qwen2.5-coder:1.5b)"
    )
    parser.add_argument(
        "--output-dir",
        default="./podcast_output",
        help="Output directory (default: ./podcast_output)"
    )
    parser.add_argument(
        "--speakers",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Number of speakers (2-4, default: 2)"
    )
    parser.add_argument(
        "--news-limit",
        type=int,
        default=15,
        help="Maximum number of news items to fetch (default: 15)"
    )
    parser.add_argument(
        "--max-news-items",
        type=int, 
        default=10,
        help="Maximum news items to include in podcast (default: 10)"
    )
    parser.add_argument(
        "--voice-config",
        help="Path to JSON file with voice preferences"
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List available voices and exit"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use for model inference (default: cuda). Examples: cuda, cuda:0, cuda:1, cpu"
    )
    parser.add_argument(
        "--hours-filter",
        type=int,
        default=24,
        help="Filter news within the last N hours (default: 24)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of individual news podcasts to generate (default: 3)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize generator
        generator = NewsPodcastGenerator(
            args.model_path,
            args.ollama_url, 
            args.ollama_model,
            args.device,
            getattr(args, 'hours_filter', 24)
        )
        
        # List voices if requested
        if args.list_voices:
            voices = generator.get_available_voices()
            print("Available voices:")
            for name, path in voices.items():
                print(f"  {name}: {path}")
            return
        
        # Load voice preferences if provided
        voice_preferences = None
        if args.voice_config:
            with open(args.voice_config, 'r') as f:
                voice_preferences = json.load(f)
        
        # Generate podcast
        print(f"Generating news podcast with {args.speakers} speakers...")
        results = generator.generate_full_podcast(
            output_dir=args.output_dir,
            num_speakers=args.speakers,
            news_limit=args.news_limit,
            max_news_items=args.max_news_items,
            voice_preferences=voice_preferences,
            count=args.count
        )
        
        # Print results
        if results['success']:
            print("\n✓ Podcast generation completed successfully!")
            print(f"📊 Statistics:")
            print(f"   Podcasts generated: {results.get('podcasts_generated', 0)} out of {results.get('total_attempted', 0)}")
            
            if results.get('results'):
                print(f"\n📁 Generated podcasts:")
                for i, podcast in enumerate(results['results'], 1):
                    print(f"   Podcast {i} (Timestamp: {podcast.get('timestamp', 'N/A')}):")
                    if 'files' in podcast:
                        for file_type, file_path in podcast['files'].items():
                            print(f"     {file_type}: {file_path}")
                    if 'news_item' in podcast:
                        news = podcast['news_item']
                        print(f"     News: {news.get('title', 'N/A')[:60]}...")
            
            # Print copy and git status
            print("\nPost-processing status:")
            if results.get('files_copied', False):
                print("  ✓ Files copied to ~/news/english-news-daily")
            else:
                print("  ✗ File copy failed")
            
            if results.get('git_pushed', False):
                print("  ✓ Git commit and push successful")
            elif results.get('files_copied', False):
                print("  ✗ Git operations failed (files copied but not committed)")
            else:
                print("  ✗ Git operations skipped (copy failed)")
        else:
            print("\n✗ Podcast generation failed")
            if 'error' in results:
                print(f"Error: {results['error']}")
            if results.get('results'):
                print("📁 Partial results:")
                for i, podcast in enumerate(results['results'], 1):
                    if podcast.get('success'):
                        print(f"   Podcast {i}: Success")
                    else:
                        print(f"   Podcast {i}: Failed - {podcast.get('error', 'Unknown error')}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()