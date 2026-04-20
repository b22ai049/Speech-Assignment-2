"""
Task 1.3: Denoising & Normalization
Spectral Subtraction for classroom noise and reverb handling.
"""

import numpy as np
import librosa
import soundfile as sf
import os


class SpectralSubtractionDenoiser:
    """Multi-band Spectral Subtraction denoiser.
    
    |S_hat(w)|^2 = max(|Y(w)|^2 - alpha(w)*|N_hat(w)|^2, beta*|N_hat(w)|^2)
    """
    
    def __init__(self, sr=16000, n_fft=512, hop_length=160, win_length=400,
                 noise_frames=30, oversubtraction=1.5, spectral_floor=0.002):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.noise_frames = noise_frames
        self.oversubtraction = oversubtraction
        self.spectral_floor = spectral_floor
        
    def estimate_noise_spectrum(self, stft_matrix):
        n = min(self.noise_frames, stft_matrix.shape[1])
        return np.mean(np.abs(stft_matrix[:, :n]) ** 2, axis=1)
    
    def compute_freq_alpha(self, n_bins):
        freqs = np.linspace(0, self.sr / 2, n_bins)
        alpha = np.ones(n_bins) * self.oversubtraction
        alpha[freqs < 300] = self.oversubtraction * 2.0
        alpha[freqs > 4000] = self.oversubtraction * 0.8
        return alpha
    
    def spectral_subtract(self, audio):
        stft = librosa.stft(audio, n_fft=self.n_fft,
                           hop_length=self.hop_length, win_length=self.win_length)
        mag = np.abs(stft)
        phase = np.angle(stft)
        power = mag ** 2
        noise_power = self.estimate_noise_spectrum(stft)
        alpha = self.compute_freq_alpha(stft.shape[0])
        
        clean_power = np.zeros_like(power)
        for t in range(power.shape[1]):
            subtracted = power[:, t] - alpha * noise_power
            floor = self.spectral_floor * noise_power
            clean_power[:, t] = np.maximum(subtracted, floor)
        
        clean_stft = np.sqrt(clean_power) * np.exp(1j * phase)
        return librosa.istft(clean_stft, hop_length=self.hop_length, win_length=self.win_length)
    
    def dereverberate(self, audio, decay=0.3):
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        mag = np.abs(stft)
        phase = np.angle(stft)
        rev = np.zeros_like(mag)
        rev[:, 0] = mag[:, 0]
        for t in range(1, mag.shape[1]):
            rev[:, t] = decay * rev[:, t-1] + (1 - decay) * mag[:, t]
        clean = np.maximum(mag - decay * rev, 0.01 * mag)
        return librosa.istft(clean * np.exp(1j * phase), hop_length=self.hop_length)
    
    def normalize(self, audio, target_db=-20.0):
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            gain = 10 ** ((target_db - 20 * np.log10(rms)) / 20)
            audio = audio * gain
        return np.clip(audio, -1.0, 1.0)
    
    def process(self, input_path, output_path):
        print(f"[Denoising] Loading {input_path}")
        audio, sr = librosa.load(input_path, sr=self.sr, mono=True)
        print(f"  Duration: {len(audio)/sr:.1f}s")
        
        print("  Step 1: Spectral subtraction...")
        audio = self.spectral_subtract(audio)
        print("  Step 2: Dereverberation...")
        audio = self.dereverberate(audio)
        print("  Step 3: Normalization...")
        audio = self.normalize(audio)
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        sf.write(output_path, audio, sr)
        print(f"  Saved to {output_path}")
        return output_path


def denoise_audio(input_path, output_path, config=None):
    cfg = config or {}
    denoiser = SpectralSubtractionDenoiser(
        sr=cfg.get('sample_rate', 16000),
        n_fft=cfg.get('n_fft', 512),
        noise_frames=cfg.get('noise_estimation_frames', 30),
        oversubtraction=cfg.get('oversubtraction_factor', 1.5),
        spectral_floor=cfg.get('spectral_floor', 0.002)
    )
    return denoiser.process(input_path, output_path)
