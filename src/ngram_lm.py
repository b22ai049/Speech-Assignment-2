"""
N-gram Language Model for Constrained Decoding (Task 1.2 support)
Trained on Speech Course Syllabus to prioritize technical terms.

Math: P(w_i | w_{i-n+1}...w_{i-1}) with Kneser-Ney smoothing
Logit bias: logit'_w = logit_w + lambda * log P_ngram(w | context)
"""

import re
import json
import math
import os
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional


class NGramLanguageModel:
    """N-gram LM with Kneser-Ney smoothing for technical term biasing."""
    
    def __init__(self, order=3, discount=0.75):
        self.order = order
        self.discount = discount
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = defaultdict(int)
        self.vocab = set()
        self.technical_terms = set()
    
    def tokenize(self, text: str) -> List[str]:
        text = text.lower().strip()
        text = re.sub(r'[^\w\s\-]', ' ', text)
        return text.split()
    
    def add_technical_terms(self, terms: List[str]):
        for t in terms:
            self.technical_terms.add(t.lower())
    
    def train(self, corpus_text: str):
        """Train on corpus text."""
        tokens = self.tokenize(corpus_text)
        self.vocab = set(tokens)
        
        for n in range(1, self.order + 1):
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i:i+n])
                context = ngram[:-1]
                word = ngram[-1]
                self.ngram_counts[context][word] += 1
                self.context_counts[context] += 1
        
        print(f"[N-gram LM] Trained: vocab={len(self.vocab)}, "
              f"technical_terms={len(self.technical_terms)}")
    
    def log_prob(self, word: str, context: Tuple[str, ...]) -> float:
        """Compute log probability with backoff."""
        word = word.lower()
        
        for n in range(len(context), -1, -1):
            ctx = context[-n:] if n > 0 else ()
            if ctx in self.ngram_counts and word in self.ngram_counts[ctx]:
                count = self.ngram_counts[ctx][word]
                total = self.context_counts[ctx]
                prob = max((count - self.discount), 0) / total
                if prob > 0:
                    return math.log(prob + 1e-10)
        
        # Uniform fallback
        return math.log(1.0 / max(len(self.vocab), 1))
    
    def compute_logit_bias(self, token_text: str, context_tokens: List[str],
                           bias_weight: float = 2.5) -> float:
        """
        Compute logit bias for a token.
        
        logit'_w = logit_w + lambda * log P(w|context)
        Additional boost for technical terms.
        """
        context = tuple(t.lower() for t in context_tokens[-(self.order-1):])
        lp = self.log_prob(token_text, context)
        bias = bias_weight * lp
        
        # Extra boost for technical terms
        if token_text.lower() in self.technical_terms:
            bias += bias_weight * 1.5
        
        return bias
    
    def get_bias_dict(self, context_tokens: List[str], tokenizer,
                      bias_weight: float = 2.5, top_k: int = 100) -> Dict[int, float]:
        """Get bias dict for Whisper logit processor."""
        biases = {}
        
        # Bias technical terms
        for term in self.technical_terms:
            token_ids = tokenizer.encode(" " + term)
            for tid in token_ids:
                decoded = tokenizer.decode([tid]).strip().lower()
                b = self.compute_logit_bias(decoded, context_tokens, bias_weight)
                if b > 0:
                    biases[tid] = b
        
        # Bias high-probability vocab words
        context = tuple(t.lower() for t in context_tokens[-(self.order-1):])
        if context in self.ngram_counts:
            for word, count in self.ngram_counts[context].most_common(top_k):
                token_ids = tokenizer.encode(" " + word)
                for tid in token_ids:
                    b = self.compute_logit_bias(word, context_tokens, bias_weight)
                    biases[tid] = max(biases.get(tid, 0), b)
        
        return biases
    
    def save(self, path):
        data = {
            "order": self.order,
            "discount": self.discount,
            "vocab": list(self.vocab),
            "technical_terms": list(self.technical_terms),
            "ngram_counts": {str(k): dict(v) for k, v in self.ngram_counts.items()},
            "context_counts": {str(k): v for k, v in self.context_counts.items()}
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        self.order = data["order"]
        self.discount = data["discount"]
        self.vocab = set(data["vocab"])
        self.technical_terms = set(data["technical_terms"])


# Speech course technical terms
SPEECH_TECHNICAL_TERMS = [
    "stochastic", "cepstrum", "cepstral", "mel", "frequency", "spectrogram",
    "fundamental", "formant", "phoneme", "phonology", "morpheme", "syntax",
    "prosody", "pitch", "intonation", "acoustic", "articulatory",
    "waveform", "sampling", "quantization", "aliasing", "nyquist",
    "fourier", "transform", "dft", "fft", "stft", "windowing",
    "hamming", "hanning", "gaussian", "filter", "bandpass", "lowpass",
    "highpass", "convolution", "autocorrelation", "lpc",
    "linear prediction", "mfcc", "delta", "acceleration",
    "hidden markov model", "hmm", "viterbi", "baum welch",
    "gaussian mixture model", "gmm", "expectation maximization",
    "neural network", "recurrent", "lstm", "gru", "attention",
    "transformer", "encoder", "decoder", "ctc", "connectionist",
    "beam search", "language model", "perplexity", "entropy",
    "word error rate", "phoneme error rate", "recognition",
    "synthesis", "vocoder", "griffin lim", "wavenet", "tacotron",
    "speaker", "verification", "identification", "diarization",
    "embedding", "ivector", "xvector", "dvector", "ecapa",
    "tdnn", "resnet", "conformer", "wav2vec", "hubert", "whisper",
    "self supervised", "pre training", "fine tuning", "transfer learning",
    "noise", "snr", "reverberation", "dereverberation", "beamforming",
    "microphone", "array", "signal", "processing", "digital",
    "analog", "amplitude", "phase", "magnitude", "power", "spectrum",
    "spectral", "temporal", "feature", "extraction", "classification",
    "regression", "segmentation", "detection", "enhancement",
    "separation", "source", "cocktail party", "masking",
    "voiced", "unvoiced", "fricative", "plosive", "nasal", "vowel",
    "consonant", "syllable", "utterance", "corpus", "dataset",
    "annotation", "transcription", "alignment", "forced alignment",
    "duration", "energy", "zero crossing rate", "zcr",
    "bark", "erb", "critical band", "auditory", "perception",
    "psychoacoustic", "loudness", "masking", "cochlea",
    "basilar membrane", "hair cell", "tonotopic",
    "parametric", "non parametric", "discriminative", "generative",
    "bayesian", "posterior", "prior", "likelihood", "evidence",
    "maximum likelihood", "maximum a posteriori", "map",
    "cross entropy", "softmax", "sigmoid", "relu", "dropout",
    "batch normalization", "layer normalization",
    "gradient", "backpropagation", "optimization", "adam", "sgd",
    "learning rate", "epoch", "batch size", "overfitting",
    "regularization", "data augmentation", "specaugment",
    "code switching", "multilingual", "cross lingual",
    "low resource", "zero shot", "few shot", "adaptation",
    "domain", "robust", "adversarial", "perturbation",
    "spoofing", "anti spoofing", "countermeasure",
    "equal error rate", "eer", "roc", "auc", "precision", "recall",
    "f1 score", "accuracy", "confusion matrix",
    "ipa", "grapheme", "phoneme", "g2p", "p2g",
    "text to speech", "tts", "speech to text", "stt", "asr",
    "automatic speech recognition",
]

SYLLABUS_CORPUS = """
Introduction to Speech Processing and Technology.
Digital Signal Processing fundamentals for speech.
Speech production mechanism: articulatory phonetics and acoustic phonetics.
Source-filter model of speech production.
The vocal tract transfer function and formant frequencies.
Sampling theorem, Nyquist rate, and aliasing in speech signals.
Quantization, pulse code modulation PCM.
Time domain analysis: energy, zero crossing rate ZCR, autocorrelation.
Short-time Fourier Transform STFT and spectrogram analysis.
Window functions: Hamming, Hanning, rectangular windows.
Discrete Fourier Transform DFT and Fast Fourier Transform FFT.
Filter banks: Mel scale, Bark scale, ERB scale.
Cepstral analysis and the cepstrum.
Mel Frequency Cepstral Coefficients MFCC extraction pipeline.
Delta and acceleration coefficients for dynamic features.
Linear Predictive Coding LPC and linear prediction analysis.
Line Spectral Pairs LSP and reflection coefficients.
Fundamental frequency F0 estimation: autocorrelation method, cepstral method.
Pitch tracking algorithms: RAPT, SWIPE, YIN, pYIN, CREPE.
Formant estimation and tracking.
Hidden Markov Models HMM for speech recognition.
Gaussian Mixture Models GMM for acoustic modeling.
Viterbi algorithm for decoding.
Baum-Welch algorithm for HMM training.
Expectation Maximization EM algorithm.
Deep Neural Networks DNN for acoustic modeling.
Recurrent Neural Networks: LSTM and GRU architectures.
Attention mechanisms and Transformer architecture.
Connectionist Temporal Classification CTC.
End-to-end speech recognition: Listen Attend and Spell LAS.
Wav2Vec 2.0 and self-supervised learning for speech.
HuBERT and data-driven speech representations.
Whisper: robust speech recognition via large-scale weak supervision.
Conformer architecture for speech recognition.
Language modeling: N-gram models, neural language models.
Beam search decoding and constrained decoding.
Word Error Rate WER and evaluation metrics.
Speaker recognition: verification and identification.
Speaker embeddings: i-vectors, d-vectors, x-vectors.
ECAPA-TDNN for speaker verification.
Speaker diarization: who spoke when.
Text-to-Speech synthesis: concatenative, parametric, neural TTS.
Tacotron and Tacotron 2 architecture.
WaveNet vocoder and autoregressive generation.
WaveRNN and WaveGlow for fast synthesis.
VITS: Variational Inference with adversarial learning.
Voice conversion and voice cloning.
Zero-shot voice cloning with speaker embeddings.
YourTTS and multilingual voice cloning.
Prosody modeling: F0, duration, energy contours.
Speech enhancement and noise reduction.
Spectral subtraction and Wiener filtering.
Deep learning for speech enhancement: Conv-TasNet, DCCRN.
Speech separation: deep clustering, permutation invariant training.
Robust speech recognition in noisy environments.
Code-switching in multilingual speech processing.
Language identification LID at utterance and frame level.
Cross-lingual and multilingual speech processing.
Low-resource language speech technology.
Adversarial attacks on speech systems.
Anti-spoofing and countermeasures for speaker verification.
LFCC and CQCC features for spoofing detection.
Equal Error Rate EER for evaluation.
Stochastic processes in speech modeling.
International Phonetic Alphabet IPA for phonetic transcription.
Grapheme-to-Phoneme G2P conversion.
Dynamic Time Warping DTW for speech alignment.
Mel-Cepstral Distortion MCD for speech quality evaluation.
"""


def build_ngram_model(order=3, bias_weight=2.5):
    """Build and return a trained N-gram LM."""
    model = NGramLanguageModel(order=order)
    model.add_technical_terms(SPEECH_TECHNICAL_TERMS)
    model.train(SYLLABUS_CORPUS)
    return model
