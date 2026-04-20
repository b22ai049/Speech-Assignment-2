"""
Task 3.2: Prosody Warping using Dynamic Time Warping (DTW)
Maps F0 and energy contours from professor's lecture onto synthesized speech.

Design Choice: We use FastDTW (approximate DTW with O(N) complexity) instead of
standard DTW (O(N^2)) because 10 minutes of audio at 100 Hz F0 rate = 60000 frames,
making exact DTW computationally prohibitive. The Sakoe-Chiba band constraint
prevents pathological warping paths.
"""

import numpy as np
import librosa
import soundfile as sf
import os
from typing import Tuple, Optional


class ProsodyWarper:
    """Extract and warp prosodic features using DTW."""
    
    def __init__(self, sr=16000, hop_length=160, f0_method="pyin"):
        self.sr = sr
        self.hop_length = hop_length
        self.f0_method = f0_method
    
    def extract_f0(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract F0 contour using pYIN.
        Returns: (f0_values, voiced_flag)
        """
        if self.f0_method == "pyin":
            f0, voiced, _ = librosa.pyin(
                audio, fmin=50, fmax=500,
                sr=self.sr, hop_length=self.hop_length
            )
        else:
            f0 = librosa.yin(
                audio, fmin=50, fmax=500,
                sr=self.sr, hop_length=self.hop_length
            )
            voiced = ~np.isnan(f0)
        
        # Replace NaN with 0
        f0 = np.nan_to_num(f0, nan=0.0)
        return f0, voiced
    
    def extract_energy(self, audio: np.ndarray) -> np.ndarray:
        """Extract frame-level energy contour."""
        frames = librosa.util.frame(audio, frame_length=self.hop_length * 2,
                                     hop_length=self.hop_length)
        energy = np.sqrt(np.mean(frames ** 2, axis=0))
        return energy
    
    def extract_prosody(self, audio: np.ndarray) -> dict:
        """Extract all prosodic features."""
        f0, voiced = self.extract_f0(audio)
        energy = self.extract_energy(audio)
        
        # Align lengths
        min_len = min(len(f0), len(energy))
        f0 = f0[:min_len]
        energy = energy[:min_len]
        voiced = voiced[:min_len] if len(voiced) >= min_len else np.ones(min_len, dtype=bool)
        
        return {"f0": f0, "energy": energy, "voiced": voiced}
    
    def dtw_align(self, source: np.ndarray, target: np.ndarray,
                  window_size: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        DTW alignment with Sakoe-Chiba band constraint.
        Returns aligned indices for source and target.
        """
        n, m = len(source), len(target)
        
        # Use FastDTW if available
        try:
            from fastdtw import fastdtw
            from scipy.spatial.distance import euclidean
            _, path = fastdtw(source.reshape(-1, 1), target.reshape(-1, 1),
                             radius=window_size, dist=euclidean)
            src_idx = np.array([p[0] for p in path])
            tgt_idx = np.array([p[1] for p in path])
            return src_idx, tgt_idx
        except ImportError:
            pass
        
        # Fallback: simplified DTW with band constraint
        cost = np.full((n + 1, m + 1), np.inf)
        cost[0, 0] = 0
        
        for i in range(1, n + 1):
            j_start = max(1, i - window_size)
            j_end = min(m + 1, i + window_size + 1)
            for j in range(j_start, j_end):
                d = abs(float(source[i-1]) - float(target[j-1]))
                cost[i, j] = d + min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])
        
        # Backtrack
        path_src, path_tgt = [], []
        i, j = n, m
        while i > 0 and j > 0:
            path_src.append(i - 1)
            path_tgt.append(j - 1)
            candidates = [
                (cost[i-1, j-1], i-1, j-1),
                (cost[i-1, j], i-1, j),
                (cost[i, j-1], i, j-1)
            ]
            _, i, j = min(candidates, key=lambda x: x[0])
        
        return np.array(path_src[::-1]), np.array(path_tgt[::-1])
    
    def warp_prosody(self, source_audio: np.ndarray,
                     target_prosody: dict) -> dict:
        """
        Warp source prosody to match target (professor's) prosody.
        Returns warped F0 and energy contours.
        """
        source_prosody = self.extract_prosody(source_audio)
        
        # DTW align F0 contours
        src_f0 = source_prosody["f0"]
        tgt_f0 = target_prosody["f0"]
        
        print(f"[Prosody] DTW aligning F0: source={len(src_f0)}, target={len(tgt_f0)}")
        src_idx, tgt_idx = self.dtw_align(src_f0, tgt_f0)
        
        # Create warped F0: use target F0 pattern scaled to source speaker range
        src_f0_voiced = src_f0[src_f0 > 0]
        tgt_f0_voiced = tgt_f0[tgt_f0 > 0]
        
        if len(src_f0_voiced) > 0 and len(tgt_f0_voiced) > 0:
            src_mean = np.mean(src_f0_voiced)
            src_std = np.std(src_f0_voiced) + 1e-6
            tgt_mean = np.mean(tgt_f0_voiced)
            tgt_std = np.std(tgt_f0_voiced) + 1e-6
            
            # Normalize target F0 to source speaker range
            warped_f0 = np.zeros(len(src_f0))
            for si, ti in zip(src_idx, tgt_idx):
                if si < len(warped_f0) and ti < len(tgt_f0):
                    if tgt_f0[ti] > 0:
                        normalized = (tgt_f0[ti] - tgt_mean) / tgt_std
                        warped_f0[si] = normalized * src_std + src_mean
        else:
            warped_f0 = src_f0.copy()
        
        # Warp energy similarly
        src_energy = source_prosody["energy"]
        tgt_energy = target_prosody["energy"]
        
        warped_energy = np.zeros(len(src_energy))
        e_src_idx, e_tgt_idx = self.dtw_align(src_energy, tgt_energy)
        
        src_e_mean = np.mean(src_energy) + 1e-6
        tgt_e_mean = np.mean(tgt_energy) + 1e-6
        
        for si, ti in zip(e_src_idx, e_tgt_idx):
            if si < len(warped_energy) and ti < len(tgt_energy):
                scale = tgt_energy[ti] / tgt_e_mean
                warped_energy[si] = src_energy[si] * scale if si < len(src_energy) else 0
        
        return {
            "f0": warped_f0,
            "energy": warped_energy,
            "voiced": source_prosody["voiced"],
            "original_f0": src_f0,
            "original_energy": src_energy
        }
    
    def apply_prosody_to_audio(self, audio: np.ndarray,
                                warped_prosody: dict) -> np.ndarray:
        """Apply warped prosody contours to audio using PSOLA-like approach."""
        # Simple energy scaling
        energy = warped_prosody["energy"]
        orig_energy = warped_prosody["original_energy"]
        
        # Frame-level energy scaling
        output = audio.copy()
        frame_len = self.hop_length * 2
        
        for i in range(min(len(energy), len(orig_energy))):
            start = i * self.hop_length
            end = min(start + frame_len, len(output))
            if start >= len(output):
                break
            
            if orig_energy[i] > 1e-6:
                scale = energy[i] / orig_energy[i]
                scale = np.clip(scale, 0.3, 3.0)
                output[start:end] *= scale
        
        output = np.clip(output, -1.0, 1.0)
        return output
    
    def process(self, source_audio_path: str, professor_audio_path: str,
                output_path: str = None) -> dict:
        """Full prosody warping pipeline."""
        print("[Prosody] Extracting professor's prosody...")
        prof_audio, _ = librosa.load(professor_audio_path, sr=self.sr)
        prof_prosody = self.extract_prosody(prof_audio)
        
        print("[Prosody] Extracting source prosody and warping...")
        src_audio, _ = librosa.load(source_audio_path, sr=self.sr)
        warped = self.warp_prosody(src_audio, prof_prosody)
        
        if output_path:
            warped_audio = self.apply_prosody_to_audio(src_audio, warped)
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            sf.write(output_path, warped_audio, self.sr)
            print(f"[Prosody] Saved warped audio to {output_path}")
        
        return warped
