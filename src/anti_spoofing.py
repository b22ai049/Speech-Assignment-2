"""
Task 4.1: Anti-Spoofing Classifier using LFCC features.
Classifies audio as Bona Fide (real) or Spoof (synthesized).
Evaluated using Equal Error Rate (EER).
"""

import numpy as np
import librosa
import os
import json
from typing import Tuple, List, Dict
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve


class LFCCExtractor:
    """Linear Frequency Cepstral Coefficients extractor."""
    
    def __init__(self, sr=16000, n_fft=512, hop_length=160,
                 n_filters=20, n_coeffs=20):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_filters = n_filters
        self.n_coeffs = n_coeffs
    
    def _linear_filterbank(self):
        """Create linear-spaced filterbank (unlike mel which is log-spaced)."""
        low_freq = 0
        high_freq = self.sr / 2
        freqs = np.linspace(low_freq, high_freq, self.n_filters + 2)
        bins = np.floor((self.n_fft + 1) * freqs / self.sr).astype(int)
        
        fbank = np.zeros((self.n_filters, self.n_fft // 2 + 1))
        for i in range(self.n_filters):
            for j in range(bins[i], bins[i+1]):
                fbank[i, j] = (j - bins[i]) / (bins[i+1] - bins[i])
            for j in range(bins[i+1], bins[i+2]):
                fbank[i, j] = (bins[i+2] - j) / (bins[i+2] - bins[i+1])
        
        return fbank
    
    def extract(self, audio: np.ndarray) -> np.ndarray:
        """Extract LFCC features from audio."""
        # Power spectrum
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        power = np.abs(stft) ** 2
        
        # Apply linear filterbank
        fbank = self._linear_filterbank()
        filtered = np.dot(fbank, power)
        filtered = np.maximum(filtered, 1e-10)
        
        # Log and DCT
        log_filtered = np.log(filtered)
        from scipy.fft import dct
        lfcc = dct(log_filtered, type=2, axis=0, norm='ortho')[:self.n_coeffs]
        
        # Add deltas
        delta = librosa.feature.delta(lfcc)
        delta2 = librosa.feature.delta(lfcc, order=2)
        
        features = np.concatenate([lfcc, delta, delta2], axis=0)
        return features.T  # (time, features)


class CQCCExtractor:
    """Constant-Q Cepstral Coefficients extractor."""
    
    def __init__(self, sr=16000, hop_length=160, n_coeffs=20):
        self.sr = sr
        self.hop_length = hop_length
        self.n_coeffs = n_coeffs
    
    def extract(self, audio: np.ndarray) -> np.ndarray:
        """Extract CQCC features."""
        cqt = np.abs(librosa.cqt(audio, sr=self.sr, hop_length=self.hop_length))
        cqt_power = cqt ** 2
        log_cqt = np.log(np.maximum(cqt_power, 1e-10))
        
        from scipy.fft import dct
        cqcc = dct(log_cqt, type=2, axis=0, norm='ortho')[:self.n_coeffs]
        
        delta = librosa.feature.delta(cqcc)
        delta2 = librosa.feature.delta(cqcc, order=2)
        
        return np.concatenate([cqcc, delta, delta2], axis=0).T


class AntiSpoofingClassifier:
    """GMM-based anti-spoofing countermeasure system."""
    
    def __init__(self, feature_type="lfcc", n_coeffs=20, sr=16000):
        self.sr = sr
        self.feature_type = feature_type
        
        if feature_type == "lfcc":
            self.extractor = LFCCExtractor(sr=sr, n_coeffs=n_coeffs)
        else:
            self.extractor = CQCCExtractor(sr=sr, n_coeffs=n_coeffs)
        
        self.gmm_bonafide = GaussianMixture(n_components=16, covariance_type='diag',
                                             max_iter=200, random_state=42)
        self.gmm_spoof = GaussianMixture(n_components=16, covariance_type='diag',
                                          max_iter=200, random_state=42)
        self.is_trained = False
    
    def extract_features(self, audio_path: str) -> np.ndarray:
        """Extract features from audio file."""
        audio, _ = librosa.load(audio_path, sr=self.sr, mono=True)
        return self.extractor.extract(audio)
    
    def extract_features_from_array(self, audio: np.ndarray) -> np.ndarray:
        return self.extractor.extract(audio)
    
    def train(self, bonafide_paths: List[str], spoof_paths: List[str]):
        """Train GMM models on bona fide and spoof data."""
        print("[Anti-Spoof] Training classifier...")
        
        # Extract features
        bf_features = []
        for path in bonafide_paths:
            feat = self.extract_features(path)
            bf_features.append(feat)
        bf_features = np.concatenate(bf_features, axis=0)
        
        sp_features = []
        for path in spoof_paths:
            feat = self.extract_features(path)
            sp_features.append(feat)
        sp_features = np.concatenate(sp_features, axis=0)
        
        print(f"  Bona fide: {bf_features.shape}, Spoof: {sp_features.shape}")
        
        self.gmm_bonafide.fit(bf_features)
        self.gmm_spoof.fit(sp_features)
        self.is_trained = True
        print("[Anti-Spoof] Training complete")
    
    def train_from_arrays(self, bonafide_audios: List[np.ndarray],
                          spoof_audios: List[np.ndarray]):
        """Train from audio arrays directly."""
        print("[Anti-Spoof] Training from audio arrays...")
        
        bf_features = np.concatenate(
            [self.extract_features_from_array(a) for a in bonafide_audios], axis=0)
        sp_features = np.concatenate(
            [self.extract_features_from_array(a) for a in spoof_audios], axis=0)
        
        self.gmm_bonafide.fit(bf_features)
        self.gmm_spoof.fit(sp_features)
        self.is_trained = True
    
    def score(self, audio_path: str) -> float:
        """
        Compute CM score. Positive = bona fide, negative = spoof.
        Score = log P(x|bonafide) - log P(x|spoof)
        """
        features = self.extract_features(audio_path)
        score_bf = np.mean(self.gmm_bonafide.score_samples(features))
        score_sp = np.mean(self.gmm_spoof.score_samples(features))
        return score_bf - score_sp
    
    def score_from_array(self, audio: np.ndarray) -> float:
        features = self.extract_features_from_array(audio)
        score_bf = np.mean(self.gmm_bonafide.score_samples(features))
        score_sp = np.mean(self.gmm_spoof.score_samples(features))
        return score_bf - score_sp
    
    def classify(self, audio_path: str) -> Tuple[str, float]:
        """Classify as bona fide or spoof."""
        s = self.score(audio_path)
        label = "bonafide" if s > 0 else "spoof"
        return label, s
    
    def compute_eer(self, bonafide_scores: List[float],
                    spoof_scores: List[float]) -> Tuple[float, float]:
        """
        Compute Equal Error Rate.
        EER is the point where FAR == FRR on the ROC curve.
        """
        labels = np.concatenate([
            np.ones(len(bonafide_scores)),
            np.zeros(len(spoof_scores))
        ])
        scores = np.concatenate([bonafide_scores, spoof_scores])
        
        fpr, tpr, thresholds = roc_curve(labels, scores)
        fnr = 1 - tpr
        
        # Find EER point
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        eer_threshold = thresholds[eer_idx]
        
        return float(eer), float(eer_threshold)
    
    def evaluate(self, bonafide_paths: List[str],
                 spoof_paths: List[str]) -> Dict:
        """Full evaluation with EER."""
        bf_scores = [self.score(p) for p in bonafide_paths]
        sp_scores = [self.score(p) for p in spoof_paths]
        
        eer, threshold = self.compute_eer(bf_scores, sp_scores)
        
        results = {
            "eer": eer,
            "threshold": threshold,
            "bonafide_scores": bf_scores,
            "spoof_scores": sp_scores,
            "feature_type": self.feature_type,
            "pass": eer < 0.10
        }
        
        print(f"[Anti-Spoof] EER: {eer:.4f} (threshold: {threshold:.4f})")
        print(f"  Pass criteria (EER < 0.10): {'PASS' if results['pass'] else 'FAIL'}")
        
        return results
