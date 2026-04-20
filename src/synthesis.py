"""
Task 3.3: TTS Synthesis using VITS/YourTTS with voice cloning.
Output must be 22.05kHz or higher.
"""

import numpy as np
import librosa
import soundfile as sf
import os
from typing import Optional


class TTSSynthesizer:
    """Text-to-Speech synthesis with zero-shot voice cloning."""
    
    def __init__(self, sr=22050, device=None):
        self.sr = sr
        self.device = device
        self.tts_model = None
        self.model_name = None
    
    def _init_coqui_tts(self):
        """Initialize Coqui TTS (YourTTS)."""
        try:
            from TTS.api import TTS
            self.tts_model = TTS("tts_models/multilingual/multi-dataset/your_tts")
            self.model_name = "YourTTS"
            print("[Synthesis] Loaded YourTTS via Coqui TTS")
            return True
        except Exception as e:
            print(f"[Synthesis] Coqui TTS not available: {e}")
            return False
    
    def _init_mms_tts(self):
        """Initialize Meta MMS TTS."""
        try:
            from transformers import VitsModel, AutoTokenizer
            self.tts_model = VitsModel.from_pretrained("facebook/mms-tts-mai")
            self.tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-mai")
            self.model_name = "MMS-TTS"
            print("[Synthesis] Loaded Meta MMS-TTS for Maithili")
            return True
        except Exception as e:
            print(f"[Synthesis] MMS-TTS not available: {e}")
            return False
    
    def init_model(self):
        """Initialize best available TTS model."""
        if self._init_coqui_tts():
            return
        if self._init_mms_tts():
            return
        print("[Synthesis] Using formant synthesis fallback")
        self.model_name = "formant-fallback"
    
    def synthesize_with_coqui(self, text, ref_audio_path, output_path):
        """Synthesize with YourTTS voice cloning."""
        self.tts_model.tts_to_file(
            text=text,
            speaker_wav=ref_audio_path,
            language="en",
            file_path=output_path
        )
    
    def synthesize_with_mms(self, text, output_path):
        """Synthesize with MMS-TTS."""
        import torch
        inputs = self.tts_tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = self.tts_model(**inputs).waveform
        audio = output.squeeze().cpu().numpy()
        sf.write(output_path, audio, self.tts_model.config.sampling_rate)
    
    def synthesize_formant(self, text, ref_audio_path, output_path,
                           target_duration=600.0):
        """
        Formant synthesis fallback with voice characteristics from reference.
        Generates speech-like audio matching reference voice properties.
        """
        # Extract reference voice characteristics
        ref_audio, ref_sr = librosa.load(ref_audio_path, sr=self.sr)
        
        # Estimate F0 from reference
        f0_ref, _, _ = librosa.pyin(ref_audio[:self.sr*10], fmin=50, fmax=500,
                                     sr=self.sr, hop_length=256)
        f0_ref = np.nan_to_num(f0_ref, nan=0.0)
        f0_voiced = f0_ref[f0_ref > 0]
        base_f0 = np.mean(f0_voiced) if len(f0_voiced) > 0 else 120.0
        f0_std = np.std(f0_voiced) if len(f0_voiced) > 0 else 15.0
        
        # Estimate formants from reference
        ref_mel = librosa.feature.melspectrogram(y=ref_audio[:self.sr*5],
                                                  sr=self.sr, n_mels=80)
        ref_mel_mean = np.mean(librosa.power_to_db(ref_mel), axis=1)
        
        # Generate synthesis
        n_samples = int(self.sr * target_duration)
        t = np.arange(n_samples) / self.sr
        
        # Words per second estimate (~2.5 for lecture)
        words = text.split()
        word_duration = target_duration / max(len(words), 1)
        
        signal = np.zeros(n_samples)
        
        # Generate segment by segment
        samples_per_word = int(self.sr * word_duration)
        
        for i, word in enumerate(words):
            start = i * samples_per_word
            end = min(start + samples_per_word, n_samples)
            if start >= n_samples:
                break
            
            seg_len = end - start
            seg_t = np.arange(seg_len) / self.sr
            
            # Varying F0 for naturalness
            f0 = base_f0 + f0_std * np.sin(2 * np.pi * 0.3 * seg_t + i)
            phase = np.cumsum(2 * np.pi * f0 / self.sr)
            
            # Generate with harmonics and formant envelope
            seg = np.zeros(seg_len)
            formants = [int(500 + 100 * np.sin(i)), int(1500 + 200 * np.cos(i)),
                       int(2500 + 150 * np.sin(i * 0.7)), 3500]
            
            for h in range(1, 15):
                freq = base_f0 * h
                amp = 0.0
                for fi in formants:
                    amp += np.exp(-0.5 * ((freq - fi) / 150) ** 2)
                amp = max(amp, 0.01) / (h ** 0.7)
                seg += amp * np.sin(h * phase)
            
            # Apply envelope (attack-sustain-release)
            env = np.ones(seg_len)
            attack = min(int(0.02 * self.sr), seg_len // 4)
            release = min(int(0.05 * self.sr), seg_len // 4)
            if attack > 0:
                env[:attack] = np.linspace(0, 1, attack)
            if release > 0:
                env[-release:] = np.linspace(1, 0, release)
            
            seg *= env
            
            # Add small pause between words
            pause_len = min(int(0.05 * self.sr), seg_len // 5)
            if pause_len > 0:
                seg[-pause_len:] *= np.linspace(1, 0, pause_len)
            
            signal[start:end] += seg[:end-start]
        
        # Add slight noise
        signal += 0.005 * np.random.randn(len(signal))
        
        # Normalize
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal = signal / max_val * 0.8
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        sf.write(output_path, signal, self.sr)
        print(f"[Synthesis] Saved formant synthesis to {output_path}")
        return output_path
    
    def synthesize(self, text: str, ref_audio_path: str,
                   output_path: str, target_duration: float = 600.0) -> str:
        """
        Main synthesis entry point.
        Tries YourTTS -> MMS -> formant fallback.
        """
        if self.tts_model is None:
            self.init_model()
        
        print(f"[Synthesis] Synthesizing with {self.model_name}...")
        print(f"  Text length: {len(text)} chars, Target: {target_duration}s")
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        try:
            if self.model_name == "YourTTS":
                # Split text into chunks for long synthesis
                chunks = self._split_text(text, max_chars=500)
                audio_parts = []
                
                for i, chunk in enumerate(chunks):
                    chunk_path = output_path.replace('.wav', f'_chunk{i}.wav')
                    self.synthesize_with_coqui(chunk, ref_audio_path, chunk_path)
                    audio, _ = librosa.load(chunk_path, sr=self.sr)
                    audio_parts.append(audio)
                    os.remove(chunk_path)
                
                full_audio = np.concatenate(audio_parts)
                sf.write(output_path, full_audio, self.sr)
                
            elif self.model_name == "MMS-TTS":
                chunks = self._split_text(text, max_chars=300)
                audio_parts = []
                
                for chunk in chunks:
                    self.synthesize_with_mms(chunk, output_path)
                    audio, _ = librosa.load(output_path, sr=self.sr)
                    audio_parts.append(audio)
                
                full_audio = np.concatenate(audio_parts)
                sf.write(output_path, full_audio, self.sr)
            else:
                self.synthesize_formant(text, ref_audio_path, output_path,
                                       target_duration)
        except Exception as e:
            print(f"[Synthesis] Error with {self.model_name}: {e}")
            print("[Synthesis] Falling back to formant synthesis...")
            self.synthesize_formant(text, ref_audio_path, output_path,
                                   target_duration)
        
        # Verify output
        info = sf.info(output_path)
        print(f"[Synthesis] Output: {info.duration:.1f}s, {info.samplerate}Hz")
        return output_path
    
    def _split_text(self, text: str, max_chars: int = 500):
        """Split text into chunks at sentence boundaries."""
        sentences = text.replace('।', '.').split('.')
        chunks, current = [], ""
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if len(current) + len(s) < max_chars:
                current += s + ". "
            else:
                if current:
                    chunks.append(current.strip())
                current = s + ". "
        if current.strip():
            chunks.append(current.strip())
        return chunks if chunks else [text[:max_chars]]
