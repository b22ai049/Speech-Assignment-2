"""
Task 3.1: Voice Embedding Extraction (d-vector / x-vector)
Extracts speaker embeddings from reference voice recording.
"""

import torch
import numpy as np
import librosa
import soundfile as sf
import os
from typing import Optional, Tuple


class VoiceEmbeddingExtractor:
    """Extract speaker embeddings using ECAPA-TDNN or simple d-vector."""
    
    def __init__(self, sr=16000, embedding_dim=256, device=None):
        self.sr = sr
        self.embedding_dim = embedding_dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_type = None
    
    def _init_speechbrain(self):
        """Try loading SpeechBrain ECAPA-TDNN for x-vectors."""
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": self.device}
            )
            self.model_type = "x-vector"
            print("[Embedding] Loaded ECAPA-TDNN (x-vector)")
            return True
        except Exception as e:
            print(f"[Embedding] SpeechBrain not available: {e}")
            return False
    
    def _init_resemblyzer(self):
        """Try loading Resemblyzer for d-vectors."""
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav
            self.model = VoiceEncoder(device=self.device)
            self.model_type = "d-vector"
            print("[Embedding] Loaded Resemblyzer (d-vector)")
            return True
        except Exception as e:
            print(f"[Embedding] Resemblyzer not available: {e}")
            return False
    
    def init_model(self):
        """Initialize best available embedding model."""
        if self._init_speechbrain():
            return
        if self._init_resemblyzer():
            return
        print("[Embedding] Using mel-spectrogram based embedding (fallback)")
        self.model_type = "mel-dvector"
    
    def extract_mel_embedding(self, audio: np.ndarray) -> np.ndarray:
        """Fallback: extract embedding from mel-spectrogram statistics."""
        mel = librosa.feature.melspectrogram(y=audio, sr=self.sr, n_mels=80)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Statistical embedding: mean, std, delta-mean, delta-std per mel band
        mean_feat = np.mean(mel_db, axis=1)
        std_feat = np.std(mel_db, axis=1)
        delta = librosa.feature.delta(mel_db)
        delta_mean = np.mean(delta, axis=1)
        delta_std = np.std(delta, axis=1)
        
        # Spectral features
        centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=self.sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=self.sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
        
        embedding = np.concatenate([
            mean_feat, std_feat, delta_mean, delta_std,
            [centroid / 8000, rolloff / 8000, zcr]
        ])
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def extract(self, audio_path: str) -> np.ndarray:
        """
        Extract speaker embedding from audio file.
        
        Args:
            audio_path: Path to reference voice recording
            
        Returns:
            Speaker embedding vector (numpy array)
        """
        if self.model is None:
            self.init_model()
        
        print(f"[Embedding] Extracting from {audio_path}")
        audio, sr = librosa.load(audio_path, sr=self.sr, mono=True)
        duration = len(audio) / sr
        print(f"  Duration: {duration:.1f}s, Model: {self.model_type}")
        
        if self.model_type == "x-vector":
            signal_tensor = torch.FloatTensor(audio).unsqueeze(0).to(self.device)
            embedding = self.model.encode_batch(signal_tensor)
            embedding = embedding.squeeze().cpu().numpy()
        elif self.model_type == "d-vector":
            from resemblyzer import preprocess_wav
            wav = preprocess_wav(audio_path)
            embedding = self.model.embed_utterance(wav)
        else:
            embedding = self.extract_mel_embedding(audio)
        
        print(f"  Embedding shape: {embedding.shape}")
        return embedding
    
    def save_embedding(self, embedding: np.ndarray, output_path: str):
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        np.save(output_path, embedding)
        print(f"[Embedding] Saved to {output_path}")
    
    def load_embedding(self, path: str) -> np.ndarray:
        return np.load(path)


def generate_reference_voice(output_path: str, duration: float = 60.0,
                             sr: int = 22050):
    """
    Generate a synthetic reference voice if no recording exists.
    In production, the student would record their own voice.
    """
    print(f"[Voice Ref] Generating synthetic reference ({duration}s)...")
    t = np.linspace(0, duration, int(sr * duration))
    
    # Generate speech-like signal with formant-like structure
    f0_base = 120  # Male fundamental
    f0 = f0_base + 10 * np.sin(2 * np.pi * 0.5 * t)  # Slow F0 variation
    
    phase = np.cumsum(2 * np.pi * f0 / sr)
    signal = np.zeros_like(t)
    
    # Harmonics with formant-like envelope
    formants = [500, 1500, 2500, 3500]
    formant_bw = [100, 200, 300, 400]
    
    for h in range(1, 20):
        harmonic_freq = f0_base * h
        # Formant envelope
        amplitude = 0.0
        for fi, bw in zip(formants, formant_bw):
            amplitude += np.exp(-0.5 * ((harmonic_freq - fi) / bw) ** 2)
        amplitude = max(amplitude, 0.01) / (h ** 0.5)
        signal += amplitude * np.sin(h * phase)
    
    # Add some noise for naturalness
    signal += 0.02 * np.random.randn(len(signal))
    
    # Amplitude modulation (syllable-like rhythm)
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)  # ~3 Hz syllable rate
    signal = signal * env
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    sf.write(output_path, signal, sr)
    print(f"[Voice Ref] Saved to {output_path}")
    return output_path
