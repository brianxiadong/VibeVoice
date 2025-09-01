import argparse
import os
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from .news_fetcher import NewsFetcher
from .ollama_processor import OllamaProcessor
from .audio_generator import PodcastAudioGenerator
from .logger import NewsLogger

class NewsPodcastGenerator:
    """Main class for generating news podcasts"""
    
    def __init__(
        self,
        model_path: str = "microsoft/VibeVoice-1.5B",
        ollama_url: str = "http://172.36.237.245:11434",
        ollama_model: str = "qwen2.5-coder:1.5b",
        device: str = "cuda",
        hours_filter: int = 24,
        debug: bool = False
    ):
        # Initialize logger first
        self.logger = NewsLogger(debug=debug).get_logger()
        
        self.model_path = model_path
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.device = device
        self.hours_filter = hours_filter
        
        self.logger.info(f"Initializing NewsPodcastGenerator with model: {model_path}, device: {device}")
        self.logger.info(f"Ollama URL: {ollama_url}, Model: {ollama_model}")
        
        try:
            # Initialize components with debug mode
            self.logger.info("Initializing news fetcher...")
            self.news_fetcher = NewsFetcher(hours_filter=hours_filter, debug=debug)
            
            self.logger.info("Initializing Ollama processor...")
            self.ollama_processor = OllamaProcessor(ollama_url, ollama_model, debug=debug)
            self.audio_generator = None  # Initialize lazily to avoid loading model unless needed
            
            self.logger.info("NewsPodcastGenerator initialization completed successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize NewsPodcastGenerator: {e}")
            raise
        
        # Check connections
        self._check_connections()
    
    def _check_connections(self):
        """Check if all services are available"""
        self.logger.info("Checking service connections...")
        
        # Check Ollama connection
        if self.ollama_processor.check_ollama_connection():
            self.logger.info(f"✓ Ollama connection successful ({self.ollama_url})")
            models = self.ollama_processor.list_available_models()
            self.logger.info(f"Available models: {models}")
            if self.ollama_model not in [model.split(':')[0] for model in models]:
                self.logger.warning(f"Specified model '{self.ollama_model}' not found in available models")
        else:
            self.logger.error(f"✗ Cannot connect to Ollama at {self.ollama_url}")
            raise ConnectionError(f"Cannot connect to Ollama at {self.ollama_url}")
    
    def _initialize_audio_generator(self):
        """Initialize audio generator only when needed"""
        if self.audio_generator is None:
            self.logger.info(f"Initializing VibeVoice model on device: {self.device}...")
            self.audio_generator = PodcastAudioGenerator(self.model_path, device=self.device)
    
    def fetch_news(self, news_limit: int = 15) -> list:
        """Fetch today's hot news"""
        self.logger.info("Fetching today's hot news...")
        news = self.news_fetcher.get_today_hot_news(news_limit)
        self.logger.info(f"Found {len(news)} news items")
        return news
    
    def process_news_to_podcast(
        self,
        news_items: list,
        num_speakers: int = 2,
        max_news_items: int = 10
    ) -> str:
        """Process news items into podcast dialogue"""
        self.logger.info(f"Processing {len(news_items)} news items into podcast dialogue...")
        
        # Summarize news
        summary = self.ollama_processor.summarize_news(news_items, max_news_items)
        self.logger.info("News summarization completed")
        
        # Create dialogue
        dialogue = self.ollama_processor.create_podcast_dialogue(summary, num_speakers)
        self.logger.info(f"Created dialogue with {num_speakers} speakers")
        
        # Enhance for audio
        enhanced_dialogue = self.ollama_processor.enhance_for_audio(dialogue)
        self.logger.info("Dialogue enhancement completed")
        
        return enhanced_dialogue
    
    def generate_audio(
        self,
        dialogue: str,
        output_path: str,
        voice_preferences: Optional[Dict[str, str]] = None
    ) -> bool:
        """Generate audio from dialogue"""
        self.logger.info(f"Generating audio to {output_path}...")
        
        self._initialize_audio_generator()
        
        success = self.audio_generator.generate_podcast_audio(
            dialogue, output_path, voice_preferences
        )
        
        if success:
            self.logger.info(f"✓ Audio generated successfully: {output_path}")
        else:
            self.logger.error("✗ Audio generation failed")
        
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
            self.logger.info(f"Processing news: {news_item['title'][:60]}...")
            dialogue = self.process_news_to_podcast([news_item], num_speakers, 1)
            results['stats']['speakers'] = num_speakers
            
            if not dialogue:
                self.logger.error("Failed to generate dialogue")
                return results
            
            # Save dialogue if requested
            if save_intermediate:
                dialogue_file = output_path / f"{base_filename}_dialogue.txt"
                with open(dialogue_file, 'w', encoding='utf-8') as f:
                    f.write(dialogue)
                results['files']['dialogue'] = str(dialogue_file)
                self.logger.info(f"Dialogue saved to: {dialogue_file}")
            
            # Generate Chinese translation
            self.logger.info("Generating Chinese translation...")
            chinese_dialogue = self.ollama_processor.translate_to_chinese(dialogue)
            
            if chinese_dialogue and save_intermediate:
                chinese_dialogue_file = output_path / f"{base_filename}_dialogue_chinese.txt"
                with open(chinese_dialogue_file, 'w', encoding='utf-8') as f:
                    f.write(chinese_dialogue)
                results['files']['chinese_dialogue'] = str(chinese_dialogue_file)
                self.logger.info(f"Chinese dialogue saved to: {chinese_dialogue_file}")
            
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
            self.logger.error(f"Error in single news podcast generation: {e}")
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
            self.logger.info(f"Fetching news and selecting best {podcast_count} items...")
            all_news = self.fetch_news(news_limit)
            
            if not all_news:
                self.logger.error("No news items found")
                return overall_results
            
            # Select best news items
            selected_news = self.news_fetcher.select_best_news(all_news, podcast_count)
            overall_results['stats']['news_available'] = len(all_news)
            overall_results['stats']['news_selected'] = len(selected_news)
            
            if not selected_news:
                self.logger.error("No suitable news items selected")
                return overall_results
            
            # Step 2: Generate individual podcasts
            for i, news_item in enumerate(selected_news, 1):
                self.logger.info(f"\n=== Generating podcast {i}/{len(selected_news)} ===")
                
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
                    self.logger.info(f"✓ Podcast {i} generated successfully")
                else:
                    self.logger.error(f"✗ Podcast {i} generation failed")
            
            # Update overall success status
            overall_results['success'] = overall_results['stats']['total_successful'] > 0
            
            return overall_results
            
        except Exception as e:
            self.logger.error(f"Error in multiple podcast generation: {e}")
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
            self.logger.info("Starting full podcast generation...")
            
            # Check connections first
            if not self._check_connections():
                return {"success": False, "error": "Service connections failed"}
            
            # Fetch news
            self.logger.info("Fetching news...")
            news_items = self.fetch_news(news_limit)
            if not news_items:
                return {"success": False, "error": "No news items found"}
            
            self.logger.info(f"Found {len(news_items)} news items")
            
            # Select best news items
            selected_news = self.news_fetcher.select_best_news(news_items, count)
            self.logger.info(f"Selected {len(selected_news)} best news items for podcast generation")
            
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
            self.logger.info(f"Successfully generated {len(successful_podcasts)} out of {len(podcast_results['podcasts'])} podcasts")
            
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
                    self.logger.warning("Failed to copy files to daily repo")
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
            self.logger.error(f"Error in full podcast generation: {e}")
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
            self.logger.info(f"Created/verified target directory: {target_date_dir}")
            
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
                    self.logger.info(f"Copied {file_type}: {source_file} -> {target_file}")
                else:
                    self.logger.warning(f"Source file not found for {file_type}: {file_path}")
            
            return len(copied_files) > 0
            
        except Exception as e:
            self.logger.error(f"Error copying files to daily repo: {e}")
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
            self.logger.info(f"Created/verified target directory: {target_date_dir}")
            
            total_copied = 0
            
            # Copy files for each podcast
            for i, podcast_result in enumerate(podcast_results, 1):
                if not podcast_result.get('success', False):
                    self.logger.warning(f"Skipping failed podcast {i}")
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
                        self.logger.info(f"Copied podcast {i} {file_type}: {source_file} -> {target_file}")
                    else:
                        self.logger.warning(f"Source file not found for podcast {i} {file_type}: {file_path}")
                
                # Also save news metadata
                news_item = podcast_result.get('news_item', {})
                if news_item:
                    metadata_file = timestamp_dir / f"{timestamp}_metadata.json"
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump(news_item, f, indent=2, ensure_ascii=False)
                    total_copied += 1
                    self.logger.info(f"Saved podcast {i} metadata: {metadata_file}")
            
            return total_copied > 0
            
        except Exception as e:
            self.logger.error(f"Error copying multiple podcasts to daily repo: {e}")
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
                self.logger.error(f"Target directory is not a git repository: {target_dir}")
                return False
            
            # Git add all changes
            result = subprocess.run(['git', 'add', '.'], capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error(f"Git add failed: {result.stderr}")
                return False
            
            # Check if there are changes to commit
            result = subprocess.run(['git', 'diff', '--cached', '--quiet'], capture_output=True)
            if result.returncode == 0:
                self.logger.info("No changes to commit")
                return True
            
            # Git commit
            commit_message = f"Add news podcast for {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ({timestamp})"
            result = subprocess.run(['git', 'commit', '-m', commit_message], capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error(f"Git commit failed: {result.stderr}")
                return False
            
            self.logger.info(f"Git commit successful: {commit_message}")
            
            # Git push
            result = subprocess.run(['git', 'push'], capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error(f"Git push failed: {result.stderr}")
                return False
            
            self.logger.info("Git push successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in git operations: {e}")
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Initialize logger early
    logger = NewsLogger(debug=args.debug).get_logger()
    logger.info(f"Starting news podcast generation with args: {vars(args)}")
    
    try:
        logger.info("Initializing NewsPodcastGenerator...")
        # Initialize generator
        generator = NewsPodcastGenerator(
            args.model_path,
            args.ollama_url, 
            args.ollama_model,
            args.device,
            args.hours_filter,
            debug=args.debug
        )
        
        # Handle list voices command
        if hasattr(args, 'list_voices') and args.list_voices:
            logger.info("Available voices:")
            voices = generator.get_available_voices()
            for voice_id, voice_info in voices.items():
                logger.info(f"  {voice_id}: {voice_info}")
            return
        
        # Load voice configuration if provided
        voice_config = None
        if args.voice_config:
            try:
                with open(args.voice_config, 'r', encoding='utf-8') as f:
                    voice_config = json.load(f)
                logger.info(f"Loaded voice configuration from {args.voice_config}")
            except Exception as e:
                logger.error(f"Failed to load voice configuration: {e}")
                return
        
        # Generate podcast
        result = generator.generate_full_podcast(
            news_limit=args.news_limit,
            count=args.count,
            num_speakers=args.speakers,
            max_news_items=args.max_news,
            voice_config=voice_config
        )
        
        if result['success']:
            logger.info("\n=== Podcast Generation Completed Successfully ===")
            logger.info(f"Generated {len(result.get('podcasts', []))} podcasts")
            if 'daily_repo_path' in result:
                logger.info(f"Files copied to: {result['daily_repo_path']}")
        else:
            logger.error(f"\n=== Podcast Generation Failed ===")
            logger.error(f"Error: {result.get('error', 'Unknown error')}")
        
            
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        print("\nProcess interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}", exc_info=True)
        print(f"\nUnexpected error occurred: {e}")
        print("Check the log file for detailed error information.")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()