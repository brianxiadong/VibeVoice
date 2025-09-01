import os
import re
from typing import List, Dict, Any, Optional, Tuple
import torch
from pathlib import Path

from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from .logger import NewsLogger

class PodcastAudioGenerator:
    """Generates multi-speaker podcast audio using VibeVoice"""
    
    def __init__(self, model_path: str = "microsoft/VibeVoice-1.5B", device: str = "cuda", debug: bool = False):
        # Initialize logger
        self.logger = NewsLogger.get_logger(self.__class__.__name__, debug=debug)
        
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None
        self.voice_presets = {}
        
        self.logger.info(f"Initializing PodcastAudioGenerator with model: {model_path}, device: {device}")
        
        try:
            self._setup_voice_presets()
            self._validate_device()
            self._load_model()
            self.logger.info("PodcastAudioGenerator initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize PodcastAudioGenerator: {e}")
            raise
    
    def _setup_voice_presets(self):
        """Setup voice presets by scanning the voices directory"""
        self.logger.debug("Setting up voice presets...")
        voices_dir = Path(__file__).parent.parent / "demo" / "voices"
        
        self.logger.debug(f"Scanning voices directory: {voices_dir}")
        if not voices_dir.exists():
            self.logger.warning(f"Voices directory not found at {voices_dir}")
            return
        
        # Scan for English voice files
        voice_count = 0
        for wav_file in voices_dir.glob("*.wav"):
            if wav_file.name.startswith("en-"):
                # Extract speaker name from filename
                # Format: en-{Name}_{gender}.wav
                name_part = wav_file.stem.replace("en-", "").split("_")[0]
                self.voice_presets[name_part] = str(wav_file)
                voice_count += 1
                self.logger.debug(f"Found voice: {name_part} -> {wav_file.name}")
        
        self.logger.info(f"Loaded {voice_count} voice presets: {list(self.voice_presets.keys())}")
        if not self.voice_presets:
            self.logger.warning("No English voice files found in voices directory")
    
    def _validate_device(self):
        """Validate and setup GPU device"""
        self.logger.debug(f"Validating device: {self.device}")
        
        if self.device.startswith('cuda'):
            if not torch.cuda.is_available():
                self.logger.warning("CUDA not available, falling back to CPU")
                self.device = "cpu"
            else:
                self.logger.debug(f"CUDA is available with {torch.cuda.device_count()} devices")
                
                # Check if specific GPU index is specified
                if ':' in self.device:
                    gpu_id = int(self.device.split(':')[1])
                    if gpu_id >= torch.cuda.device_count():
                        self.logger.warning(f"GPU {gpu_id} not available, using GPU 0")
                        self.device = "cuda:0"
                
                # Log GPU memory info
                if self.device != "cpu":
                    try:
                        gpu_id = 0 if ':' not in self.device else int(self.device.split(':')[1])
                        memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                        memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                        memory_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
                        self.logger.info(f"GPU {gpu_id} memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved, {memory_total:.2f}GB total")
                    except Exception as e:
                        self.logger.warning(f"Failed to get GPU memory info: {e}")
        
        self.logger.info(f"Using device: {self.device}")
    
    def _load_model(self):
        """Load VibeVoice model and processor"""
        try:
            self.logger.info(f"Loading VibeVoice model: {self.model_path} on device: {self.device}")
            
            # Load processor
            self.logger.debug("Loading VibeVoice processor...")
            self.processor = VibeVoiceProcessor.from_pretrained(self.model_path)
            self.logger.debug("Processor loaded successfully")
            
            # Determine device_map based on device setting
            if self.device == "cpu":
                device_map = "cpu"
                torch_dtype = torch.float32  # Use float32 for CPU
                self.logger.debug("Using CPU mode with float32 precision")
            else:
                # For GPU devices, avoid device_map="auto" to prevent parameter allocation issues
                # Instead, load on CPU first and then move to target device
                device_map = None  # Load on CPU first
                torch_dtype = torch.bfloat16
                self.logger.debug("Using GPU mode with bfloat16 precision")
            
            # Load model with correct parameters
            attn_implementation = "flash_attention_2" if self.device != "cpu" else "eager"
            
            self.logger.info(f"Using device_map: {device_map}, target_device: {self.device}, attention: {attn_implementation}")
            
            # Load model with device_map parameter
            if device_map is not None:
                self.logger.debug("Loading model directly to target device...")
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    attn_implementation=attn_implementation
                )
            else:
                # Load on CPU first, then move to target device
                self.logger.debug("Loading model on CPU first, then moving to target device...")
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=torch_dtype,
                    attn_implementation=attn_implementation
                )
                
                # Move entire model to target device
                self.logger.info(f"Moving model to {self.device}")
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self.model.set_ddpm_inference_steps(num_steps=10)
            self.logger.debug("Model set to evaluation mode with 10 DDPM inference steps")
            
            # Verify actual device placement
            if hasattr(self.model, 'device'):
                self.logger.debug(f"Model device after loading: {self.model.device}")
            elif hasattr(self.model, 'hf_device_map'):
                self.logger.debug(f"Model device map: {self.model.hf_device_map}")
            
            # Check individual component devices
            if hasattr(self.model, 'model'):
                if hasattr(self.model.model, 'device'):
                    self.logger.debug(f"Inner model device: {self.model.model.device}")
                if hasattr(self.model.model, 'language_model') and hasattr(self.model.model.language_model, 'device'):
                    self.logger.debug(f"Language model device: {self.model.model.language_model.device}")
            
            if hasattr(self.model.model, 'language_model'):
                self.logger.debug(f"Language model attention: {self.model.model.language_model.config._attn_implementation}")
            
            self.logger.info("Model loaded successfully")
            
            # Clear cache after loading
            if self.device != "cpu":
                torch.cuda.empty_cache()
                self.logger.debug("GPU memory cache cleared after model loading")
                
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def parse_dialogue(self, dialogue_text: str) -> List[Tuple[str, str]]:
        """Parse dialogue text into (speaker, text) pairs"""
        # Debug logging to see what dialogue_text is being received
        self.logger.info(f"Parsing dialogue text (length: {len(dialogue_text)})")
        self.logger.info(f"First 500 characters: {dialogue_text[:500]}")
        
        lines = dialogue_text.strip().split('\n')
        parsed_dialogue = []
        
        self.logger.info(f"Split into {len(lines)} lines")
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            self.logger.debug(f"Processing line {i+1}: '{line}'")
            
            # Look for various dialogue formats
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    speaker = parts[0].strip()
                    text = parts[1].strip()
                    
                    # Clean up speaker name - remove asterisk prefixes
                    # Handle formats like "**Speaker 1:**" or "**Speaker 2:**"
                    if speaker.startswith('**'):
                        speaker = speaker[2:].strip()  # Remove ** prefix
                        self.logger.debug(f"Cleaned speaker name from asterisk prefix: '{speaker}'")
                    elif speaker.startswith('*'):
                        speaker = speaker[1:].strip()  # Remove * prefix
                        self.logger.debug(f"Cleaned speaker name from asterisk prefix: '{speaker}'")
                    
                    # Clean up text - remove asterisk prefixes from text as well
                    if text.startswith('**'):
                        text = text[2:].strip()  # Remove ** prefix from text
                        self.logger.debug(f"Cleaned text from asterisk prefix")
                    elif text.startswith('*'):
                        text = text[1:].strip()  # Remove * prefix from text
                        self.logger.debug(f"Cleaned text from asterisk prefix")
                    
                    # Accept various speaker formats:
                    # - "Speaker 1", "Speaker 2", etc. (VibeVoice format)
                    # - "Host", "Analyst", "Reporter", etc. (named speakers)
                    # - Any text followed by colon
                    if text and speaker:  # Only add non-empty text and speaker
                        # Normalize speaker names for consistency
                        if speaker.startswith('Speaker '):
                            # Keep VibeVoice format as-is, but ensure proper format
                            normalized_speaker = speaker
                        else:
                            # Convert other formats to Speaker format for voice assignment
                            # Map common speaker names to Speaker format
                            speaker_mapping = {
                                'Host': 'Speaker 1',
                                'Analyst': 'Speaker 2', 
                                'Reporter': 'Speaker 1',
                                'Guest': 'Speaker 2'
                            }
                            normalized_speaker = speaker_mapping.get(speaker, speaker)
                        
                        parsed_dialogue.append((normalized_speaker, text))
                        self.logger.debug(f"Added dialogue pair: {normalized_speaker} -> {text[:50]}...")
                    else:
                        self.logger.debug(f"Skipped line with empty speaker or text: '{line}'")
            else:
                self.logger.debug(f"Line has no colon, skipping: '{line}'")
        
        self.logger.info(f"Parsed {len(parsed_dialogue)} dialogue pairs")
        return parsed_dialogue
    
    def assign_voices_to_speakers(self, speakers: List[str], voice_preferences: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Assign voice files to speakers"""
        voice_assignment = {}
        available_voices = list(self.voice_presets.keys())
        
        if not available_voices:
            self.logger.warning("No English voices available")
            return voice_assignment
        
        # Use preferences if provided
        if voice_preferences:
            for speaker, preferred_voice in voice_preferences.items():
                if preferred_voice in self.voice_presets:
                    voice_assignment[speaker] = self.voice_presets[preferred_voice]
        
        # Assign remaining speakers to available voices
        used_voices = set(voice_assignment.values())
        remaining_voices = [v for k, v in self.voice_presets.items() if v not in used_voices]
        
        for speaker in speakers:
            if speaker not in voice_assignment:
                if remaining_voices:
                    voice_assignment[speaker] = remaining_voices.pop(0)
                else:
                    # Reuse voices if we run out
                    voice_assignment[speaker] = list(self.voice_presets.values())[
                        len(voice_assignment) % len(self.voice_presets)
                    ]
        
        self.logger.info(f"Voice assignments: {voice_assignment}")
        return voice_assignment
    
    def generate_audio_segment(self, speaker: str, text: str, voice_path: str) -> torch.Tensor:
        """Generate audio for a single text segment with specified voice"""
        try:
            # Format text with speaker information as expected by VibeVoice
            formatted_text = f"{speaker}: {text}"
            
            # Process the input - use correct format with lists
            inputs = self.processor(
                text=[formatted_text],  # Wrap in list for batch processing
                voice_samples=[voice_path],  # Wrap in list for batch processing
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            
            # Move inputs to correct device
            if self.device != "cpu":
                for key in inputs:
                    if hasattr(inputs[key], 'to'):
                        inputs[key] = inputs[key].to(self.device)
            
            # Generate audio with correct parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=1.3,
                    tokenizer=self.processor.tokenizer,
                    generation_config={'do_sample': False},
                    verbose=True,
                )
            
            # Extract audio from outputs.speech_outputs[0]
            if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
                audio_output = outputs.speech_outputs[0]
                # Move to CPU to save GPU memory
                if hasattr(audio_output, 'cpu'):
                    audio_output = audio_output.cpu()
                
                # Clear GPU cache after each generation
                if self.device != "cpu":
                    torch.cuda.empty_cache()
                
                return audio_output
            else:
                self.logger.error("No audio output generated")
                return None
            
        except Exception as e:
            self.logger.error(f"Error generating audio for text '{formatted_text[:50]}...': {e}")
            # Clear cache on error as well
            if self.device != "cpu":
                torch.cuda.empty_cache()
            return None
    
    def concatenate_audio_segments(self, audio_segments: List[torch.Tensor]) -> torch.Tensor:
        """Concatenate multiple audio segments with brief pauses"""
        if not audio_segments:
            self.logger.warning("No audio segments to concatenate")
            return torch.tensor([])
        
        # Filter out None segments
        valid_segments = [seg for seg in audio_segments if seg is not None]
        
        if not valid_segments:
            self.logger.warning("No valid audio segments to concatenate")
            return torch.tensor([])
        
        self.logger.info(f"Concatenating {len(valid_segments)} audio segments")
        
        # Add brief silence between segments (0.5 seconds at 24kHz)
        sample_rate = 24000
        silence_duration = int(0.5 * sample_rate)
        silence = torch.zeros(silence_duration, dtype=valid_segments[0].dtype, device=valid_segments[0].device)
        
        self.logger.debug(f"Adding 0.5s silence ({silence_duration} samples) between segments")
        
        concatenated_segments = []
        total_samples = 0
        for i, segment in enumerate(valid_segments):
            # Ensure segment is properly shaped
            if segment.dim() > 1:
                segment = segment.squeeze()
            
            concatenated_segments.append(segment)
            total_samples += len(segment)
            
            # Add silence between segments (except after the last one)
            if i < len(valid_segments) - 1:
                concatenated_segments.append(silence)
                total_samples += silence_duration
        
        final_audio = torch.cat(concatenated_segments, dim=0)
        self.logger.info(f"Audio concatenation complete: {total_samples} total samples ({total_samples/sample_rate:.2f}s)")
        
        return final_audio
    
    def generate_podcast_audio(
        self, 
        dialogue_text: str, 
        output_path: str,
        voice_preferences: Optional[Dict[str, str]] = None
    ) -> bool:
        """Generate complete podcast audio from dialogue text"""
        
        if not self.model or not self.processor:
            self.logger.error("Model not loaded")
            return False
        
        try:
            # Parse the dialogue
            dialogue_pairs = self.parse_dialogue(dialogue_text)
            if not dialogue_pairs:
                self.logger.error("No valid dialogue found")
                return False
            
            # Get unique speakers
            speakers = list(set(pair[0] for pair in dialogue_pairs))
            self.logger.info(f"Found speakers: {speakers}")
            
            # Assign voices to speakers
            voice_assignment = self.assign_voices_to_speakers(speakers, voice_preferences)
            
            if not voice_assignment:
                self.logger.error("No voice assignments made")
                return False
            
            # Generate audio for each dialogue segment
            audio_segments = []
            self.logger.info(f"Generating audio for {len(dialogue_pairs)} segments")
            
            for i, (speaker, text) in enumerate(dialogue_pairs):
                self.logger.info(f"Generating segment {i+1}/{len(dialogue_pairs)}: {speaker}")
                
                voice_path = voice_assignment.get(speaker)
                if not voice_path:
                    self.logger.warning(f"No voice assigned to speaker: {speaker}")
                    continue
                
                audio_segment = self.generate_audio_segment(speaker, text, voice_path)
                if audio_segment is not None:
                    audio_segments.append(audio_segment)
                else:
                    self.logger.warning(f"Failed to generate audio for segment {i+1}")
            
            if not audio_segments:
                self.logger.error("No audio segments generated")
                return False
            
            # Concatenate all segments
            self.logger.info("Concatenating audio segments")
            final_audio = self.concatenate_audio_segments(audio_segments)
            
            # Save the audio
            self.save_audio(final_audio, output_path)
            self.logger.info(f"Podcast audio saved to: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating podcast audio: {e}")
            return False
    
    def save_audio(self, audio_tensor: torch.Tensor, output_path: str, sample_rate: int = 24000):
        """Save audio tensor to file using processor.save_audio"""
        try:
            self.logger.info(f"Saving audio to: {output_path}")
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                self.logger.debug(f"Created output directory: {output_dir}")
            
            # Log audio properties
            if hasattr(audio_tensor, 'shape'):
                self.logger.debug(f"Audio tensor shape: {audio_tensor.shape}")
            else:
                self.logger.debug(f"Audio tensor length: {len(audio_tensor)}")
            
            # Use processor.save_audio method (same as inference_from_file.py)
            self.processor.save_audio(
                audio_tensor,
                output_path=output_path,
            )
            
            # Verify file was created
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                duration = len(audio_tensor) / sample_rate
                self.logger.info(f"Audio saved successfully: {file_size} bytes, {duration:.2f}s duration")
            else:
                self.logger.error(f"Audio file was not created: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving audio: {e}")
            raise
    
    def get_available_voices(self) -> Dict[str, str]:
        """Get available voice presets"""
        return self.voice_presets.copy()
    
    def get_default_voice_preferences(self, speakers: List[str]) -> Dict[str, str]:
        """Get default voice preferences for speakers"""
        available_voices = list(self.voice_presets.keys())
        preferences = {}
        
        # Default speaker-to-voice mappings for VibeVoice format
        voice_mapping = {
            'Speaker 1': 'Alice',
            'Speaker 2': 'Carter', 
            'Speaker 3': 'Frank',
            'Speaker 4': 'Maya'
        }
        
        for speaker in speakers:
            if speaker in voice_mapping and voice_mapping[speaker] in available_voices:
                preferences[speaker] = voice_mapping[speaker]
            elif available_voices:
                # Assign round-robin style
                idx = len(preferences) % len(available_voices)
                preferences[speaker] = available_voices[idx]
        
        return preferences
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage information"""
        if self.device == "cpu":
            return {"device": "cpu", "memory_info": "CPU mode - no GPU memory tracking"}
        
        try:
            gpu_id = 0 if ':' not in self.device else int(self.device.split(':')[1])
            memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
            memory_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
            memory_free = memory_total - memory_reserved
            
            return {
                "device": self.device,
                "allocated_gb": round(memory_allocated, 2),
                "reserved_gb": round(memory_reserved, 2),
                "total_gb": round(memory_total, 2),
                "free_gb": round(memory_free, 2),
                "utilization_percent": round((memory_reserved / memory_total) * 100, 1)
            }
        except Exception as e:
            self.logger.error(f"Error getting memory usage: {e}")
            return {"error": str(e)}
    
    def clear_memory_cache(self):
        """Clear GPU memory cache"""
        if self.device != "cpu":
            torch.cuda.empty_cache()
            self.logger.info("GPU memory cache cleared")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test PodcastAudioGenerator")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda, cuda:0, cuda:1, cpu)")
    parser.add_argument("--model", default="microsoft/VibeVoice-1.5B", help="Model path")
    args = parser.parse_args()
    
    # Test the audio generator with specified device
    print(f"Initializing PodcastAudioGenerator with device: {args.device}")
    generator = PodcastAudioGenerator(model_path=args.model, device=args.device)
    
    # Show memory usage
    memory_info = generator.get_memory_usage()
    print(f"Memory usage: {memory_info}")
    
    # Test with sample dialogue
    sample_dialogue = """Speaker 1: Welcome to today's hot news podcast! We have some fascinating developments to discuss.
Speaker 2: Absolutely! There's been quite a lot happening in the tech world today.
Speaker 1: Let's dive into the main stories. What caught your attention?
Speaker 2: The AI breakthroughs and new programming frameworks are particularly interesting.
Speaker 1: Great insights! Thanks for joining us today."""
    
    print("Available voices:", generator.get_available_voices())
    
    # Generate podcast audio
    output_file = "/tmp/test_podcast.wav"
    success = generator.generate_podcast_audio(sample_dialogue, output_file)
    
    if success:
        print(f"Test podcast generated successfully: {output_file}")
        # Show final memory usage
        final_memory = generator.get_memory_usage()
        print(f"Final memory usage: {final_memory}")
    else:
        print("Failed to generate test podcast")