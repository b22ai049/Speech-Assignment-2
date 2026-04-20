"""
Task 2.2: Semantic Translation to Maithili (LRL)
Uses parallel corpus + NLLB fallback for translation.
"""

import json
import os
import re
from typing import Dict, List, Optional


class MaithiliTranslator:
    """Translate English/Hindi text to Maithili using corpus + MT fallback."""
    
    def __init__(self, corpus: Dict[str, str] = None):
        self.corpus = corpus or {}
        self.nllb_model = None
        self.nllb_tokenizer = None
    
    def load_corpus(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            self.corpus = json.load(f)
        print(f"[Translation] Loaded {len(self.corpus)} entries")
    
    def _init_nllb(self):
        """Initialize NLLB model for fallback translation."""
        if self.nllb_model is not None:
            return True
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            model_name = "facebook/nllb-200-distilled-600M"
            print(f"[Translation] Loading NLLB model: {model_name}")
            self.nllb_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            return True
        except Exception as e:
            print(f"[Translation] NLLB not available: {e}")
            return False
    
    def translate_word(self, word: str) -> str:
        """Translate single word using corpus lookup."""
        w = word.lower().strip()
        if w in self.corpus:
            return self.corpus[w]
        # Try without common suffixes
        for suffix in ['ing', 'ed', 'tion', 'sion', 'ment', 'ness', 'ly', 's', 'es']:
            stem = w[:-len(suffix)] if w.endswith(suffix) else None
            if stem and stem in self.corpus:
                return self.corpus[stem]
        return None
    
    def translate_with_nllb(self, text: str, src_lang="hin_Deva",
                            tgt_lang="mai_Deva") -> str:
        """Translate using NLLB model."""
        if not self._init_nllb():
            return text
        try:
            self.nllb_tokenizer.src_lang = src_lang
            inputs = self.nllb_tokenizer(text, return_tensors="pt", max_length=512,
                                         truncation=True)
            tgt_id = self.nllb_tokenizer.convert_tokens_to_ids(tgt_lang)
            output = self.nllb_model.generate(
                **inputs, forced_bos_token_id=tgt_id, max_length=512
            )
            return self.nllb_tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            print(f"[Translation] NLLB error: {e}")
            return text
    
    def translate_text(self, text: str, use_nllb_fallback: bool = True) -> str:
        """
        Translate full text to Maithili.
        Strategy: corpus lookup per word, NLLB for uncovered phrases.
        """
        words = text.split()
        translated = []
        uncovered = []
        
        i = 0
        while i < len(words):
            # Try multi-word lookup (up to 3 words)
            found = False
            for n in range(min(3, len(words) - i), 0, -1):
                phrase = ' '.join(words[i:i+n]).lower()
                if phrase in self.corpus:
                    translated.append(self.corpus[phrase])
                    i += n
                    found = True
                    break
            
            if not found:
                result = self.translate_word(words[i])
                if result:
                    translated.append(result)
                else:
                    # Keep original (technical term or untranslatable)
                    translated.append(words[i])
                    uncovered.append(words[i])
                i += 1
        
        result_text = ' '.join(translated)
        
        # Try NLLB for remaining uncovered segments
        if uncovered and use_nllb_fallback:
            print(f"  {len(uncovered)} words not in corpus, trying NLLB...")
            result_text = self.translate_with_nllb(result_text, "hin_Deva", "mai_Deva")
        
        return result_text
    
    def translate_segments(self, segments: List[Dict],
                           use_nllb: bool = True) -> List[Dict]:
        """Translate transcript segments."""
        print(f"[Translation] Translating {len(segments)} segments to Maithili...")
        translated_segments = []
        
        for seg in segments:
            text = seg.get("text", "")
            translated_text = self.translate_text(text, use_nllb)
            translated_segments.append({
                **seg,
                "original_text": text,
                "text": translated_text,
                "target_language": "mai"
            })
        
        return translated_segments
    
    def save_translation(self, translated_text: str, output_path: str):
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(translated_text)
        print(f"[Translation] Saved to {output_path}")
