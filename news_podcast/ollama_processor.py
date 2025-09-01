import requests
import json
from typing import List, Dict, Any, Optional
from .logger import NewsLogger

class OllamaProcessor:
    """Processes news content using local Ollama LLM"""
    
    def __init__(self, base_url: str = "http://172.36.237.245:11434", model: str = "qwen2.5-coder:1.5b", debug: bool = False):
        self.logger = NewsLogger(debug=debug).get_logger()
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.session = requests.Session()
        
        self.logger.info(f"Initializing OllamaProcessor with base_url={base_url}, model={model}")
        
        # Test connection on initialization
        if self.check_ollama_connection():
            self.logger.info("Successfully connected to Ollama service")
            available_models = self.list_available_models()
            if self.model in [m.split(':')[0] + ':' + m.split(':')[1] if ':' in m else m for m in available_models]:
                self.logger.info(f"Model '{self.model}' is available")
            else:
                self.logger.warning(f"Model '{self.model}' not found in available models: {available_models}")
        else:
            self.logger.error(f"Failed to connect to Ollama service at {base_url}")
            self.logger.error("Please ensure Ollama is running and accessible")
    
    def _call_ollama(self, prompt: str, system_prompt: str = "") -> Optional[str]:
        """Make a call to Ollama API"""
        prompt_preview = prompt[:100] + '...' if len(prompt) > 100 else prompt
        self.logger.debug(f"Calling Ollama API with prompt: {prompt_preview}")
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False
            }
            
            self.logger.debug(f"Sending request to {self.base_url}/api/generate with model {self.model}")
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            response_text = result.get('response', '').strip()
            
            if response_text:
                response_preview = response_text[:100] + '...' if len(response_text) > 100 else response_text
                self.logger.debug(f"Received response: {response_preview}")
                self.logger.info(f"Successfully generated {len(response_text)} characters of content")
            else:
                self.logger.warning("Received empty response from Ollama")
            
            return response_text
            
        except requests.RequestException as e:
            self.logger.error(f"Network error calling Ollama API: {e}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse Ollama response JSON: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error calling Ollama: {e}", exc_info=True)
            return None
    
    def summarize_news(self, news_items: List[Dict[str, Any]], max_items: int = 10) -> str:
        """Summarize news items into a coherent summary"""
        self.logger.info(f"Summarizing {len(news_items)} news items (max: {max_items})")
        
        # Select top news items
        selected_news = news_items[:max_items]
        self.logger.debug(f"Selected {len(selected_news)} news items for summarization")
        
        # Prepare news content for summarization
        news_content = []
        for i, item in enumerate(selected_news, 1):
            title = item.get('title', '')
            text = item.get('text', '')
            source = item.get('source', '')
            
            news_entry = f"{i}. {title} (Source: {source})"
            if text and len(text) > 20:
                news_entry += f"\\n   Summary: {text[:200]}..."
            news_content.append(news_entry)
        
        news_text = "\n\n".join(news_content)
        self.logger.debug(f"Prepared news content for summarization: {len(news_text)} characters")
        
        system_prompt = """You are a professional news analyst and podcast content creator. Your task is to analyze today's hot news and create a comprehensive summary that will be used for podcast generation."""
        
        prompt = f"""Please analyze the following hot news items and create a comprehensive summary in English:

{news_text}

Please provide:
1. A brief overview of the main themes and topics
2. Key highlights and important developments
3. Any connections or patterns between different stories
4. The potential impact or significance of these developments

Keep the summary informative, engaging, and suitable for audio content. Focus on the most important and interesting aspects."""

        self.logger.info("Generating news summary using Ollama...")
        summary = self._call_ollama(prompt, system_prompt)
        
        if summary:
            self.logger.info(f"Successfully generated news summary ({len(summary)} characters)")
            return summary
        else:
            self.logger.error("Failed to generate news summary, using fallback")
            return "Unable to generate summary."
    
    def create_podcast_dialogue(self, news_summary: str, num_speakers: int = 2) -> str:
        """Convert news summary into a multi-speaker podcast dialogue"""
        self.logger.info(f"Creating podcast dialogue with {num_speakers} speakers")
        self.logger.debug(f"Input summary length: {len(news_summary)} characters")
        
        system_prompt = f"""You are a professional podcast script writer. Create an engaging, natural conversation between {num_speakers} speakers discussing today's hot news.

CRITICAL FORMAT REQUIREMENTS - FOLLOW EXACTLY:
- Use EXACTLY this format: "Speaker 1:", "Speaker 2:", "Speaker 3:", "Speaker 4:" 
- NO markdown formatting (###), NO quotes around dialogue
- NO names like "Host:", "Expert:", "Sarah:", etc.
- Each speaker's dialogue should be on its own line
- Use natural conversational language with contractions, fillers, and emotional expressions
- Include natural speech patterns like "Well...", "You know", "Yeah", "Hmm", "Right?"

Example correct format:
Speaker 1: Hey there, welcome to today's news podcast!
Speaker 2: Thanks for having me. There's been quite a lot happening.
Speaker 1: Absolutely! Let's dive right in.

Guidelines:
- Make the conversation natural and engaging with real human speech patterns
- Include different perspectives and insights
- Use conversational English with natural transitions and reactions
- Each speaker should have distinct personality and viewpoints
- Include questions, agreements, disagreements, and natural dialogue flow
- Keep the content informative but accessible
- Aim for about 15-20 exchanges total
- Add natural pauses and emotional responses"""
        
        prompt = f"""Based on the following news summary, create an engaging podcast dialogue between {num_speakers} speakers:

News Summary:
{news_summary}

Create a natural conversation where the speakers discuss these topics, share insights, ask questions, and provide different perspectives. Make sure each speaker contributes meaningfully to the discussion.

REMEMBER: Use "Speaker 1:", "Speaker 2:", etc. format ONLY. Make it sound like real people talking naturally."""

        self.logger.info("Generating podcast dialogue using Ollama...")
        dialogue = self._call_ollama(prompt, system_prompt)
        
        if not dialogue:
            # Fallback dialogue if API fails
            self.logger.warning("Ollama dialogue generation failed, using fallback dialogue")
            dialogue = self._create_fallback_dialogue(news_summary, num_speakers)
        else:
            self.logger.info(f"Successfully generated dialogue ({len(dialogue)} characters)")
        
        # Post-process to fix format issues
        self.logger.debug("Post-processing dialogue format...")
        dialogue = self._fix_dialogue_format(dialogue)
        self.logger.debug(f"Final dialogue length: {len(dialogue)} characters")
        
        return dialogue
    
    def _create_fallback_dialogue(self, news_summary: str, num_speakers: int) -> str:
        """Create a simple fallback dialogue if LLM fails"""
        return f"""Speaker 1: Welcome to today's hot news podcast! We've got some fascinating developments to discuss.

Speaker 2: Absolutely! There's been quite a lot happening in the tech and news world today.

Speaker 1: Right? Let's dive into the main stories. From what I'm seeing, we have several interesting developments across different sectors.

Speaker 2: That's right. The technology sector seems particularly active, and there are some significant global news items worth discussing.

Speaker 1: What are your thoughts on the overall trends we're seeing?

Speaker 2: I think these developments show how rapidly the digital landscape is evolving. It's fascinating to see the interconnections between different stories.

Speaker 1: Excellent point. These stories really highlight the dynamic nature of today's news cycle.

Speaker 2: Thanks for joining us today, everyone. We'll be back tomorrow with more hot news analysis!"""
    
    def _fix_dialogue_format(self, dialogue: str) -> str:
        """Fix common formatting issues in generated dialogue"""
        import re
        
        lines = dialogue.strip().split('\n')
        fixed_lines = []
        current_speaker = None
        current_text = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a speaker line (various formats)
            speaker_match = re.match(r'^[*#]*\s*Speaker\s*(\d+)[*]*\s*:', line)
            if speaker_match:
                # Save previous speaker if exists
                if current_speaker and current_text:
                    fixed_lines.append(f"Speaker {current_speaker}: {current_text}")
                
                # Start new speaker
                current_speaker = speaker_match.group(1)
                remaining_text = line[speaker_match.end():].strip()
                
                # Remove markdown formatting
                remaining_text = re.sub(r'^[*#\s]*', '', remaining_text)
                remaining_text = re.sub(r'[*#]*$', '', remaining_text)
                
                # Remove quotes and clean up text
                remaining_text = re.sub(r'^"(.+)"$', r'\1', remaining_text)
                
                # Remove self-introductions and names  
                remaining_text = re.sub(r"^(Hi there!?\s+)?I'm \w+[,.]?\s*", 'Hi there! ', remaining_text, flags=re.IGNORECASE)
                remaining_text = re.sub(r"^(Well,?\s+)?(as an? \w+[,.]?\s*)", r'\1', remaining_text, flags=re.IGNORECASE)
                
                current_text = remaining_text.strip()
                
            elif current_speaker and line.startswith('"') and line.endswith('"'):
                # This is dialogue text with quotes
                text = line[1:-1]  # Remove quotes
                if current_text:
                    current_text += " " + text
                else:
                    current_text = text
            elif current_speaker and not re.match(r'^[*#]*[A-Z][a-z]+[*]*:', line):
                # This is continuation text (not another speaker)
                clean_line = re.sub(r"^I'm \w+[,.]?\s*", '', line, flags=re.IGNORECASE)
                if current_text:
                    current_text += " " + clean_line
                else:
                    current_text = clean_line
        
        # Don't forget the last speaker
        if current_speaker and current_text:
            fixed_lines.append(f"Speaker {current_speaker}: {current_text}")
        
        return '\n\n'.join(fixed_lines)
    
    def enhance_for_audio(self, dialogue: str) -> str:
        """Enhance dialogue for better audio generation"""
        self.logger.info("Enhancing dialogue for audio generation...")
        self.logger.debug(f"Input dialogue length: {len(dialogue)} characters")
        
        system_prompt = """You are an audio content specialist. Enhance the given dialogue to make it more suitable for text-to-speech generation by adding natural pauses, emphasis markers, and improving flow.

CRITICAL: Keep the exact "Speaker 1:", "Speaker 2:" format. Do NOT change it to any other format."""
        
        prompt = f"""Please enhance the following podcast dialogue for better audio generation:

{dialogue}

Improvements to make:
1. Add natural pauses with periods and commas
2. Ensure smooth transitions between speakers  
3. Make language more conversational and natural for speech
4. Fix any awkward phrasing that might sound odd when spoken
5. Keep the same content but improve readability for TTS
6. MAINTAIN the "Speaker 1:", "Speaker 2:" format exactly as is
7. Add natural speech fillers and expressions where appropriate

Enhanced dialogue:"""

        enhanced = self._call_ollama(prompt, system_prompt)
        
        if enhanced:
            self.logger.info(f"Successfully enhanced dialogue ({len(enhanced)} characters)")
            return enhanced
        else:
            self.logger.warning("Failed to enhance dialogue, returning original")
            return dialogue
    
    def translate_to_chinese(self, english_dialogue: str) -> str:
        """Translate English dialogue to Chinese"""
        self.logger.info("Translating English dialogue to Chinese...")
        self.logger.debug(f"Input dialogue length: {len(english_dialogue)} characters")
        
        system_prompt = """你是一个专业的翻译专家，专门负责将英文播客对话翻译成中文。请保持对话的自然性和流畅性，同时保持原有的格式。

重要格式要求：
- 保持 "Speaker 1:", "Speaker 2:" 等格式不变
- 翻译内容要自然流畅，符合中文表达习惯
- 保持对话的语气和情感
- 保持专业术语的准确性"""
        
        prompt = f"""请将以下英文播客对话翻译成中文，保持原有的Speaker格式：

{english_dialogue}

翻译要求：
1. 保持"Speaker 1:", "Speaker 2:"等格式完全不变
2. 翻译要自然流畅，符合中文播客对话习惯
3. 保持原对话的语气和情感
4. 专业术语要准确翻译
5. 保持对话的逻辑结构和流程

中文翻译："""
        
        chinese_dialogue = self._call_ollama(prompt, system_prompt)
        
        if chinese_dialogue:
            self.logger.info(f"Successfully translated dialogue to Chinese ({len(chinese_dialogue)} characters)")
            return chinese_dialogue
        else:
            # 如果翻译失败，返回一个简单的中文对话
            self.logger.warning("Translation failed, using fallback Chinese dialogue")
            chinese_dialogue = self._create_fallback_chinese_dialogue()
            return chinese_dialogue
    
    def _create_fallback_chinese_dialogue(self) -> str:
        """Create a simple fallback Chinese dialogue if translation fails"""
        return """Speaker 1: 欢迎收听今日热点新闻播客！我们有一些精彩的内容要和大家分享。

Speaker 2: 是的！今天科技和新闻界发生了很多有趣的事情。

Speaker 1: 没错！让我们深入了解主要新闻。从我看到的情况来看，各个行业都有一些有趣的发展。

Speaker 2: 确实如此。科技行业似乎特别活跃，还有一些值得讨论的重要全球新闻。

Speaker 1: 你对我们看到的整体趋势有什么看法？

Speaker 2: 我认为这些发展显示了数字化环境发展的速度有多快。看到不同新闻之间的相互联系真的很有趣。

Speaker 1: 说得很好。这些新闻真正突出了当今新闻周期的动态特性。

Speaker 2: 感谢大家今天的收听！我们明天会带来更多热点新闻分析！"""

    def check_ollama_connection(self) -> bool:
        """Check if Ollama is accessible with detailed debugging"""
        try:
            self.logger.debug(f"Testing connection to Ollama at {self.base_url}")
            
            # Test version endpoint first
            version_url = f"{self.base_url}/api/version"
            self.logger.debug(f"Attempting to connect to: {version_url}")
            
            response = self.session.get(version_url, timeout=10)  # Increased timeout
            
            if response.status_code == 200:
                version_data = response.json()
                self.logger.debug(f"Ollama connection successful - Version: {version_data.get('version', 'unknown')}")
                
                # Also test the tags endpoint to ensure full functionality
                try:
                    tags_response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
                    if tags_response.status_code == 200:
                        self.logger.debug("Tags endpoint also accessible")
                    else:
                        self.logger.warning(f"Tags endpoint returned status {tags_response.status_code}")
                except Exception as tags_error:
                    self.logger.warning(f"Tags endpoint test failed: {tags_error}")
                
                return True
            else:
                self.logger.error(f"Ollama connection failed with HTTP {response.status_code}")
                self.logger.error(f"Response headers: {dict(response.headers)}")
                try:
                    error_text = response.text[:500]  # First 500 chars of error
                    self.logger.error(f"Response body: {error_text}")
                except:
                    pass
                return False
                
        except requests.exceptions.ConnectTimeout as e:
            self.logger.error(f"Connection timeout to Ollama: {e}")
            self.logger.error("This usually means the server is not responding or is overloaded")
            return False
        except requests.exceptions.ConnectionError as e:
            self.logger.error(f"Connection error to Ollama: {e}")
            self.logger.error("This usually means the server is not running or not accessible")
            return False
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error to Ollama: {e}")
            self.logger.error(f"Error type: {type(e).__name__}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error testing Ollama connection: {e}")
            self.logger.error(f"Error type: {type(e).__name__}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def list_available_models(self) -> List[str]:
        """List available models in Ollama"""
        try:
            self.logger.debug("Fetching available models from Ollama...")
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = [model.get('name', '') for model in data.get('models', [])]
                self.logger.debug(f"Found {len(models)} available models: {models}")
                return models
            else:
                self.logger.warning(f"Failed to fetch models, status code: {response.status_code}")
        except requests.RequestException as e:
            self.logger.error(f"Network error listing models: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error listing models: {e}")
        return []

if __name__ == "__main__":
    # Test the processor
    processor = OllamaProcessor()
    
    if processor.check_ollama_connection():
        print("✓ Ollama connection successful")
        models = processor.list_available_models()
        print(f"Available models: {models}")
        
        # Test with sample news
        sample_news = [
            {
                'title': 'AI Breakthrough in Language Models',
                'text': 'Researchers have achieved a significant breakthrough in language model efficiency',
                'source': 'Tech News',
                'score': 100
            },
            {
                'title': 'New Programming Framework Released',
                'text': 'A new framework promises to revolutionize web development',
                'source': 'Developer News',
                'score': 85
            }
        ]
        
        summary = processor.summarize_news(sample_news)
        print(f"\\nNews Summary:\\n{summary}\\n")
        
        dialogue = processor.create_podcast_dialogue(summary, num_speakers=2)
        print(f"Podcast Dialogue:\\n{dialogue}")
    else:
        print("✗ Cannot connect to Ollama. Please check the connection and URL.")