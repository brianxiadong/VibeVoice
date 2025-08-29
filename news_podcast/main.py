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
        ollama_model: str = "qwen2.5-coder:1.5b"
    ):
        self.model_path = model_path
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        
        # Initialize components
        self.news_fetcher = NewsFetcher()
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
            logger.info("Initializing VibeVoice model...")
            self.audio_generator = PodcastAudioGenerator(self.model_path)
    
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
    
    def generate_full_podcast(
        self,
        output_dir: str = "./output",
        num_speakers: int = 2,
        news_limit: int = 15,
        max_news_items: int = 10,
        voice_preferences: Optional[Dict[str, str]] = None,
        save_intermediate: bool = True
    ) -> Dict[str, Any]:
        """Generate complete podcast from news to audio"""
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate timestamp-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"news_podcast_{timestamp}"
        
        results = {
            'timestamp': timestamp,
            'success': False,
            'files': {},
            'stats': {}
        }
        
        try:
            # Step 1: Fetch news
            news_items = self.fetch_news(news_limit)
            results['stats']['news_count'] = len(news_items)
            
            if not news_items:
                logger.error("No news items found")
                return results
            
            # Save news data if requested
            if save_intermediate:
                news_file = output_path / f"{base_filename}_news.json"
                with open(news_file, 'w', encoding='utf-8') as f:
                    json.dump(news_items, f, indent=2, ensure_ascii=False)
                results['files']['news_data'] = str(news_file)
                logger.info(f"News data saved to: {news_file}")
            
            # Step 2: Process to dialogue
            dialogue = self.process_news_to_podcast(news_items, num_speakers, max_news_items)
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
            
            # Step 3: Generate audio
            audio_file = output_path / f"{base_filename}.wav"
            audio_success = self.generate_audio(dialogue, str(audio_file), voice_preferences)
            
            if audio_success:
                results['files']['audio'] = str(audio_file)
                results['success'] = True
                
                # Get audio file size
                if audio_file.exists():
                    results['stats']['audio_size_mb'] = round(audio_file.stat().st_size / 1024 / 1024, 2)
                
                # Copy files to daily repository and perform git operations
                try:
                    logger.info("Starting file copy and git upload process...")
                    
                    # Copy files to ~/news/english-news-daily
                    copy_success = self._copy_files_to_daily_repo(results['files'], timestamp)
                    
                    if copy_success:
                        logger.info("Files copied successfully to daily repository")
                        results['copy_success'] = True
                        
                        # Perform git operations
                        target_base_dir = Path("/root/news/english-news-daily")
                        
                        git_success = self._git_commit_and_push(target_base_dir, timestamp)
                        
                        if git_success:
                            logger.info("Git operations completed successfully")
                            results['git_success'] = True
                        else:
                            logger.warning("Git operations failed, but files were copied")
                            results['git_success'] = False
                    else:
                        logger.warning("File copy to daily repository failed")
                        results['copy_success'] = False
                        results['git_success'] = False
                        
                except Exception as e:
                    logger.error(f"Error in post-processing (copy/git): {e}")
                    results['copy_success'] = False
                    results['git_success'] = False
                    # Don't fail the entire process if copy/git fails
            
            return results
            
        except Exception as e:
            logger.error(f"Error in podcast generation: {e}")
            results['error'] = str(e)
            return results
    
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
    
    args = parser.parse_args()
    
    try:
        # Initialize generator
        generator = NewsPodcastGenerator(
            args.model_path,
            args.ollama_url, 
            args.ollama_model
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
            voice_preferences=voice_preferences
        )
        
        # Print results
        if results['success']:
            print("\n✓ Podcast generated successfully!")
            print(f"Timestamp: {results['timestamp']}")
            print(f"News items processed: {results['stats']['news_count']}")
            print(f"Speakers: {results['stats']['speakers']}")
            
            if 'audio_size_mb' in results['stats']:
                print(f"Audio file size: {results['stats']['audio_size_mb']} MB")
            
            print("\nGenerated files:")
            for file_type, file_path in results['files'].items():
                print(f"  {file_type}: {file_path}")
            
            # Print copy and git status
            print("\nPost-processing status:")
            if results.get('copy_success', False):
                print("  ✓ Files copied to ~/news/english-news-daily")
            else:
                print("  ✗ File copy failed")
            
            if results.get('git_success', False):
                print("  ✓ Git commit and push successful")
            elif results.get('copy_success', False):
                print("  ✗ Git operations failed (files copied but not committed)")
            else:
                print("  ✗ Git operations skipped (copy failed)")
        else:
            print("\\n✗ Podcast generation failed")
            if 'error' in results:
                print(f"Error: {results['error']}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()