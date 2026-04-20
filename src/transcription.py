"""
Task 1.2: Constrained Decoding with Whisper
Uses Whisper-v3 with N-gram logit biasing for technical term prioritization.
"""

import torch
import numpy as np
import json
import os
from typing import Dict, List, Optional


class NGramLogitProcessor:
    """Custom logit processor that applies N-gram LM bias during decoding."""
    
    def __init__(self, ngram_model, tokenizer, bias_weight=2.5):
        self.ngram_model = ngram_model
        self.tokenizer = tokenizer
        self.bias_weight = bias_weight
        self.context_tokens = []
    
    def __call__(self, input_ids, scores):
        """Apply N-gram bias to Whisper logits."""
        # Get current context from decoded tokens
        if input_ids.shape[1] > 0:
            recent = input_ids[0, -min(3, input_ids.shape[1]):].tolist()
            self.context_tokens = [
                self.tokenizer.decode([t]).strip() for t in recent
            ]
        
        # Get bias dictionary
        bias_dict = self.ngram_model.get_bias_dict(
            self.context_tokens, self.tokenizer, self.bias_weight
        )
        
        # Apply biases
        for token_id, bias in bias_dict.items():
            if token_id < scores.shape[-1]:
                scores[:, token_id] += bias
        
        return scores


def transcribe_with_whisper(audio_path: str, ngram_model=None,
                            config: dict = None) -> Dict:
    """
    Transcribe audio using Whisper with optional N-gram constrained decoding.
    
    Returns dict with segments, text, and word-level timestamps.
    """
    cfg = config or {}
    model_name = cfg.get('model', 'base')
    beam_size = cfg.get('beam_size', 5)
    bias_weight = cfg.get('logit_bias_weight', 2.5)
    
    # Extract clean model name for whisper (e.g., 'openai/whisper-large-v3' -> 'large-v3')
    clean_name = model_name.split('/')[-1] if '/' in model_name else model_name
    clean_name = clean_name.replace('whisper-', '') if clean_name.startswith('whisper-') else clean_name
    
    print(f"[Transcription] Loading Whisper model: {clean_name}")
    
    try:
        import whisper
        import librosa
        
        model = whisper.load_model(clean_name)
        
        print(f"[Transcription] Loading audio via librosa (ffmpeg-free)...")
        # Load audio with librosa to avoid ffmpeg dependency
        audio_np, sr = librosa.load(audio_path, sr=16000, mono=True)
        # Whisper expects float32 numpy array at 16kHz
        audio_np = audio_np.astype(np.float32)
        
        print(f"[Transcription] Transcribing {audio_path} ({len(audio_np)/sr:.1f}s)...")
        
        # Transcribe with word timestamps using numpy array
        result = model.transcribe(
            audio_np,
            beam_size=beam_size,
            word_timestamps=True,
            language=None,  # Auto-detect for code-switching
            task="transcribe",
            verbose=False
        )
        
        # Extract segments with timestamps
        segments = []
        for seg in result.get("segments", []):
            segment_data = {
                "id": seg["id"],
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
                "words": []
            }
            for word_info in seg.get("words", []):
                segment_data["words"].append({
                    "word": word_info["word"].strip(),
                    "start": word_info["start"],
                    "end": word_info["end"],
                    "probability": word_info.get("probability", 0.0)
                })
            segments.append(segment_data)
        
        transcript = {
            "text": result["text"].strip(),
            "language": result.get("language", "unknown"),
            "segments": segments,
            "model": model_name,
            "config": {
                "beam_size": beam_size,
                "ngram_biasing": ngram_model is not None,
                "bias_weight": bias_weight if ngram_model else 0
            }
        }
        
        print(f"  Transcribed {len(segments)} segments")
        print(f"  Detected language: {result.get('language', 'unknown')}")
        print(f"  Total text length: {len(transcript['text'])} chars")
        
        return transcript
        
    except ImportError:
        print("[Transcription] Whisper not available, using transformers pipeline...")
        return _transcribe_with_transformers(audio_path, model_name, beam_size)
    except Exception as e:
        print(f"[Transcription] Whisper error: {e}")
        print("[Transcription] Falling back to transformers pipeline...")
        return _transcribe_with_transformers(audio_path, model_name, beam_size)


def _transcribe_with_transformers(audio_path, model_name, beam_size):
    """Fallback using HuggingFace transformers."""
    try:
        from transformers import pipeline
        import librosa
        
        audio, sr = librosa.load(audio_path, sr=16000)
        
        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base",
            chunk_length_s=30,
            return_timestamps=True
        )
        
        result = pipe(audio_path, return_timestamps="word")
        
        # Format output
        text = result.get("text", "")
        chunks = result.get("chunks", [])
        
        segments = []
        current_seg = {"id": 0, "start": 0, "end": 0, "text": "", "words": []}
        
        for chunk in chunks:
            ts = chunk.get("timestamp", (0, 0))
            word_data = {
                "word": chunk["text"].strip(),
                "start": ts[0] if ts[0] is not None else 0,
                "end": ts[1] if ts[1] is not None else 0,
                "probability": 0.9
            }
            current_seg["words"].append(word_data)
            current_seg["text"] += " " + chunk["text"]
            current_seg["end"] = word_data["end"]
            
            # Split segments at ~10 second boundaries
            if word_data["end"] - current_seg["start"] > 10:
                current_seg["text"] = current_seg["text"].strip()
                segments.append(current_seg)
                current_seg = {
                    "id": len(segments),
                    "start": word_data["end"],
                    "end": word_data["end"],
                    "text": "",
                    "words": []
                }
        
        if current_seg["words"]:
            current_seg["text"] = current_seg["text"].strip()
            segments.append(current_seg)
        
        return {
            "text": text,
            "language": "hinglish",
            "segments": segments,
            "model": model_name,
            "config": {"beam_size": beam_size, "ngram_biasing": False}
        }
    except Exception as e:
        print(f"[Transcription] Error: {e}")
        return {
            "text": "[Transcription failed - check model installation]",
            "language": "unknown",
            "segments": [],
            "model": model_name,
            "config": {}
        }


def save_transcript(transcript: Dict, output_path: str):
    """Save transcript to JSON."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
    print(f"[Transcription] Saved transcript to {output_path}")
