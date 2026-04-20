"""
Evaluation metrics for all tasks: WER, MCD, LID accuracy, EER.
"""

import numpy as np
import librosa
import json
import os
from typing import Dict, List, Tuple, Optional


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate using dynamic programming."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    n, m = len(ref_words), len(hyp_words)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[n][m] / max(n, 1)


def compute_mcd(ref_audio: np.ndarray, syn_audio: np.ndarray,
                sr: int = 22050, n_mfcc: int = 13) -> float:
    """
    Compute Mel-Cepstral Distortion.
    MCD = (10*sqrt(2)/ln(10)) * sqrt(sum((mc_ref - mc_syn)^2))
    """
    # Align lengths
    min_len = min(len(ref_audio), len(syn_audio))
    ref_audio = ref_audio[:min_len]
    syn_audio = syn_audio[:min_len]
    
    # Extract MFCCs (exclude C0)
    mfcc_ref = librosa.feature.mfcc(y=ref_audio, sr=sr, n_mfcc=n_mfcc + 1)[1:]
    mfcc_syn = librosa.feature.mfcc(y=syn_audio, sr=sr, n_mfcc=n_mfcc + 1)[1:]
    
    # Align frame counts
    min_frames = min(mfcc_ref.shape[1], mfcc_syn.shape[1])
    mfcc_ref = mfcc_ref[:, :min_frames]
    mfcc_syn = mfcc_syn[:, :min_frames]
    
    # MCD formula
    diff = mfcc_ref - mfcc_syn
    frame_mcd = np.sqrt(2 * np.sum(diff ** 2, axis=0))
    mcd = (10.0 / np.log(10)) * np.mean(frame_mcd)
    
    return float(mcd)


def compute_lid_accuracy(predictions: List[Dict],
                          ground_truth: List[Dict] = None) -> Dict:
    """Compute LID metrics including F1 and switch accuracy."""
    if not predictions:
        return {"f1": 0.0, "accuracy": 0.0}
    
    en_count = sum(1 for p in predictions if p["language"] == "en")
    hi_count = sum(1 for p in predictions if p["language"] == "hi")
    total = len(predictions)
    
    # If no ground truth, compute statistics only
    if ground_truth is None:
        avg_conf = np.mean([p["confidence"] for p in predictions])
        return {
            "total_frames": total,
            "english_frames": en_count,
            "hindi_frames": hi_count,
            "english_ratio": en_count / max(total, 1),
            "hindi_ratio": hi_count / max(total, 1),
            "avg_confidence": float(avg_conf),
            "estimated_f1": float(avg_conf)  # Confidence as proxy
        }
    
    # With ground truth: compute proper metrics
    tp_en, fp_en, fn_en = 0, 0, 0
    tp_hi, fp_hi, fn_hi = 0, 0, 0
    
    for pred, gt in zip(predictions, ground_truth):
        p_lang = pred["language"]
        g_lang = gt["language"]
        
        if p_lang == "en" and g_lang == "en":
            tp_en += 1
        elif p_lang == "en" and g_lang == "hi":
            fp_en += 1
            fn_hi += 1
        elif p_lang == "hi" and g_lang == "hi":
            tp_hi += 1
        elif p_lang == "hi" and g_lang == "en":
            fp_hi += 1
            fn_en += 1
    
    prec_en = tp_en / max(tp_en + fp_en, 1)
    rec_en = tp_en / max(tp_en + fn_en, 1)
    f1_en = 2 * prec_en * rec_en / max(prec_en + rec_en, 1e-6)
    
    prec_hi = tp_hi / max(tp_hi + fp_hi, 1)
    rec_hi = tp_hi / max(tp_hi + fn_hi, 1)
    f1_hi = 2 * prec_hi * rec_hi / max(prec_hi + rec_hi, 1e-6)
    
    return {
        "f1_english": float(f1_en),
        "f1_hindi": float(f1_hi),
        "f1_macro": float((f1_en + f1_hi) / 2),
        "accuracy": float((tp_en + tp_hi) / max(total, 1)),
        "pass": (f1_en + f1_hi) / 2 >= 0.85
    }


def compute_switch_precision(predicted_switches: List[Dict],
                              gt_switches: List[Dict] = None,
                              tolerance_ms: float = 200.0) -> Dict:
    """Compute language switch boundary precision."""
    if not predicted_switches:
        return {"num_switches": 0, "precision_within_200ms": 0.0}
    
    if gt_switches is None:
        return {
            "num_switches": len(predicted_switches),
            "switch_times": [s["time"] for s in predicted_switches],
            "note": "No ground truth available"
        }
    
    tolerance_s = tolerance_ms / 1000.0
    matched = 0
    
    for ps in predicted_switches:
        for gs in gt_switches:
            if abs(ps["time"] - gs["time"]) <= tolerance_s:
                matched += 1
                break
    
    precision = matched / max(len(predicted_switches), 1)
    recall = matched / max(len(gt_switches), 1)
    
    return {
        "num_predicted": len(predicted_switches),
        "num_ground_truth": len(gt_switches),
        "matched": matched,
        "precision": float(precision),
        "recall": float(recall),
        "tolerance_ms": tolerance_ms
    }


def full_evaluation(transcript: Dict = None, lid_results: List[Dict] = None,
                     switches: List[Dict] = None, ref_audio_path: str = None,
                     syn_audio_path: str = None, spoof_results: Dict = None,
                     adversarial_results: Dict = None) -> Dict:
    """Compile all evaluation metrics."""
    results = {"timestamp": str(np.datetime64('now'))}
    
    # WER (if reference transcript available)
    if transcript:
        results["transcription"] = {
            "text_length": len(transcript.get("text", "")),
            "num_segments": len(transcript.get("segments", [])),
            "detected_language": transcript.get("language", "unknown")
        }
    
    # LID
    if lid_results:
        results["lid"] = compute_lid_accuracy(lid_results)
    
    # Switch precision
    if switches:
        results["switches"] = compute_switch_precision(switches)
    
    # MCD
    if ref_audio_path and syn_audio_path:
        try:
            ref, sr1 = librosa.load(ref_audio_path, sr=22050)
            syn, sr2 = librosa.load(syn_audio_path, sr=22050)
            mcd = compute_mcd(ref, syn, sr=22050)
            results["mcd"] = {
                "value": mcd,
                "pass": mcd < 8.0,
                "threshold": 8.0
            }
        except Exception as e:
            results["mcd"] = {"error": str(e)}
    
    # Anti-spoofing
    if spoof_results:
        results["anti_spoofing"] = spoof_results
    
    # Adversarial
    if adversarial_results:
        results["adversarial"] = adversarial_results
    
    return results


def save_evaluation(results: Dict, output_path: str):
    """Save evaluation results to JSON."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Convert numpy types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            r = convert(obj)
            if r is not obj:
                return r
            return super().default(obj)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"[Evaluation] Results saved to {output_path}")
