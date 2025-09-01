# News Podcast Generator

Generate engaging news podcasts from today's hot news using VibeVoice's multi-speaker text-to-speech technology and local LLM processing.

## Features

- 🗞️ **Automatic News Collection**: Fetches hot news from multiple sources (Hacker News, Reddit, GitHub)
- 🤖 **AI-Powered Processing**: Uses local Ollama LLM to summarize news and create engaging dialogue
- 🎙️ **Multi-Speaker Audio**: Generates natural-sounding podcasts with 2-4 different speakers
- 🎭 **Voice Customization**: Choose from multiple English voice presets for different speakers
- 📁 **Complete Output**: Saves news data, dialogue script, and final audio file

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- Local Ollama server running (configured in the code for http://172.36.237.245:11434)
- VibeVoice model (automatically downloaded)
- Additional Python packages: `soundfile`, `requests`

## Installation

1. Make sure you have VibeVoice installed:
```bash
cd /path/to/VibeVoice
pip install -e .
pip install soundfile  # For audio file saving
```

2. Ensure your Ollama server is running with a suitable model (e.g., llama3.1)

## Usage

### Command Line Interface

Basic usage:
```bash
python -m news_podcast.main
```

Advanced usage with options:
```bash
python -m news_podcast.main \
    --model-path "WestZhang/VibeVoice-Large-pt" \
    --speakers 3 \
    --output-dir "./my_podcasts" \
    --news-limit 20 \
    --voice-config voice_config_example.json
```

### Available Options

- `--model-path`: VibeVoice model path (default: microsoft/VibeVoice-1.5B)
- `--ollama-url`: Ollama server URL (default: http://172.36.237.245:11434)
- `--ollama-model`: Ollama model name (default: llama3.1)
- `--speakers`: Number of speakers 2-4 (default: 2)
- `--output-dir`: Output directory (default: ./podcast_output)
- `--news-limit`: Max news items to fetch (default: 15)
- `--max-news-items`: Max news items to include in podcast (default: 10)
- `--voice-config`: JSON file with voice preferences
- `--list-voices`: Show available voices
- `--device`: GPU device to use (default: cuda). Examples: cuda, cuda:0, cuda:1, cpu

### Voice Configuration

Create a JSON file to specify which voices to use for different speakers:

```json
{
  "Host": "Alice",
  "Expert": "Carter", 
  "Tech Expert": "Frank",
  "News Analyst": "Maya"
}
```

Available English voices:
- Alice (woman)
- Carter (man)
- Frank (man) 
- Maya (woman)

### GPU Device Selection

To solve CUDA out of memory issues or use specific GPUs:

```bash
# Use specific GPU (e.g., GPU 1 instead of GPU 0)
python -m news_podcast.main --device cuda:1

# Use CPU (slower but no memory limits)
python -m news_podcast.main --device cpu

# Check available GPUs
nvidia-smi
```

### Python API

```python
from news_podcast.main import NewsPodcastGenerator

# Initialize with specific GPU
generator = NewsPodcastGenerator(
    model_path="microsoft/VibeVoice-1.5B",
    ollama_url="http://172.36.237.245:11434",
    ollama_model="llama3.1",
    device="cuda:1"  # Use GPU 1
)

# Generate complete podcast
results = generator.generate_full_podcast(
    output_dir="./output",
    num_speakers=2,
    voice_preferences={"Host": "Alice", "Expert": "Carter"}
)

if results['success']:
    print(f"Podcast saved to: {results['files']['audio']}")
```

## How It Works

1. **News Collection**: Fetches trending content from:
   - Hacker News top stories
   - Reddit (r/technology, r/worldnews)
   - GitHub trending repositories

2. **Content Processing**: 
   - Uses local Ollama LLM to summarize news
   - Creates engaging multi-speaker dialogue in VibeVoice format
   - Applies format corrections to ensure compatibility
   - Enhances text for better speech synthesis

3. **Audio Generation**:
   - Parses dialogue in "Speaker 1:", "Speaker 2:" format
   - Assigns different voices to speakers
   - Generates audio for each dialogue segment
   - Concatenates segments with appropriate pauses

## Output Files

The generator creates several files:
- `news_podcast_TIMESTAMP.wav`: Final podcast audio
- `news_podcast_TIMESTAMP_dialogue.txt`: Generated dialogue script  
- `news_podcast_TIMESTAMP_news.json`: Raw news data used

## Configuration

### Ollama Model Requirements

The system works best with instruction-following models like:
- llama3.1 (recommended)
- llama2
- mistral
- qwen2.5

Make sure your model is loaded in Ollama:
```bash
ollama pull llama3.1
```

### VibeVoice Model Selection

- `microsoft/VibeVoice-1.5B`: Faster, good quality (default)
- `WestZhang/VibeVoice-Large-pt`: Higher quality, slower

## Troubleshooting

### CUDA Memory Issues

If you encounter "CUDA out of memory" errors:

1. **Use a different GPU**:
   ```bash
   # Check GPU memory usage
   nvidia-smi
   
   # Use GPU with more free memory
   python -m news_podcast.main --device cuda:1
   ```

2. **Use CPU mode** (slower but no memory limits):
   ```bash
   python -m news_podcast.main --device cpu
   ```

3. **Use smaller model**:
   ```bash
   python -m news_podcast.main --model-path "microsoft/VibeVoice-1.5B"
   ```

4. **Set PyTorch memory allocation** (as suggested in error message):
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   python -m news_podcast.main
   ```

### Memory Optimization Features

The system includes automatic memory management:
- GPU memory is cleared after each audio segment generation
- Models are moved between GPU/CPU as needed
- Memory usage is monitored and reported

### Other Issues

1. **Ollama Connection Issues**: Verify Ollama server is running and accessible at the configured URL
2. **Audio Quality**: Try the larger model for better quality (if you have enough GPU memory)
3. **Missing Voices**: Check that English voice files exist in `demo/voices/`

## Example Output

The system generates natural-sounding conversations like:

```
Host: Welcome to today's hot news podcast! We have some fascinating developments to discuss.
Expert: Absolutely! There's been quite a lot happening in the tech world today.
Host: Let's start with the AI breakthroughs. What caught your attention?
Expert: The new language model developments are particularly impressive...
```