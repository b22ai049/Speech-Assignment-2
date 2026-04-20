"""
Task 2.1: IPA Unified Representation
Custom Hinglish G2P mapping for code-switched transcript to IPA.

Design Choice: Standard G2P tools (espeak, epitran) fail on code-switched text 
because they assume monolingual input. Our approach uses LID labels per word to 
route through language-specific G2P, with a custom bridging layer that handles:
1. Hindi words written in Roman script (transliterated Hinglish)
2. English technical terms borrowed into Hindi with modified phonology
3. Discourse markers that exist in both languages ("so", "like", "matlab")
"""

import re
import json
import os
from typing import Dict, List, Tuple, Optional


# ============ Devanagari to IPA mapping ============
DEVANAGARI_IPA = {
    'अ': 'ə', 'आ': 'aː', 'इ': 'ɪ', 'ई': 'iː', 'उ': 'ʊ', 'ऊ': 'uː',
    'ए': 'eː', 'ऐ': 'æː', 'ओ': 'oː', 'औ': 'ɔː', 'अं': 'əm', 'अः': 'əh',
    'क': 'k', 'ख': 'kʰ', 'ग': 'ɡ', 'घ': 'ɡʰ', 'ङ': 'ŋ',
    'च': 'tʃ', 'छ': 'tʃʰ', 'ज': 'dʒ', 'झ': 'dʒʰ', 'ञ': 'ɲ',
    'ट': 'ʈ', 'ठ': 'ʈʰ', 'ड': 'ɖ', 'ढ': 'ɖʰ', 'ण': 'ɳ',
    'त': 'tˑ', 'थ': 'tˑʰ', 'द': 'dˑ', 'ध': 'dˑʰ', 'न': 'n',
    'प': 'p', 'फ': 'pʰ', 'ब': 'b', 'भ': 'bʰ', 'म': 'm',
    'य': 'j', 'र': 'r', 'ल': 'l', 'व': 'ʋ',
    'श': 'ʃ', 'ष': 'ʂ', 'स': 's', 'ह': 'ɦ',
    'ा': 'aː', 'ि': 'ɪ', 'ी': 'iː', 'ु': 'ʊ', 'ू': 'uː',
    'े': 'eː', 'ै': 'æː', 'ो': 'oː', 'ौ': 'ɔː',
    'ं': 'ⁿ', 'ः': 'h', '्': '', 'ँ': '̃',
    'ड़': 'ɽ', 'ढ़': 'ɽʰ', 'क़': 'q', 'ख़': 'x', 'ग़': 'ɣ',
    'ज़': 'z', 'फ़': 'f',
}

# ============ Roman Hindi (transliterated) to IPA ============
ROMAN_HINDI_IPA = {
    'ka': 'kə', 'kha': 'kʰə', 'ga': 'ɡə', 'gha': 'ɡʰə',
    'cha': 'tʃə', 'ja': 'dʒə', 'jha': 'dʒʰə',
    'ta': 'ʈə', 'tha': 'ʈʰə', 'da': 'ɖə', 'dha': 'ɖʰə',
    'na': 'nə', 'pa': 'pə', 'pha': 'pʰə', 'ba': 'bə', 'bha': 'bʰə',
    'ma': 'mə', 'ya': 'jə', 'ra': 'rə', 'la': 'lə', 'va': 'ʋə',
    'sha': 'ʃə', 'sa': 'sə', 'ha': 'ɦə',
    'aa': 'aː', 'ee': 'iː', 'oo': 'uː', 'ai': 'æː', 'au': 'ɔː',
    'kk': 'kː', 'tt': 'ʈː', 'pp': 'pː', 'dd': 'ɖː',
}

# ============ English phoneme approximations ============
ENGLISH_IPA_SIMPLE = {
    'a': 'æ', 'e': 'ɛ', 'i': 'ɪ', 'o': 'ɒ', 'u': 'ʌ',
    'th': 'θ', 'sh': 'ʃ', 'ch': 'tʃ', 'ng': 'ŋ', 'ph': 'f',
    'wh': 'w', 'ck': 'k', 'gh': 'ɡ', 'dg': 'dʒ',
    'tion': 'ʃən', 'sion': 'ʒən', 'ous': 'əs', 'ture': 'tʃər',
    'ight': 'aɪt', 'ough': 'ɔː', 'eous': 'iːəs',
}

# Common Hinglish words that need special handling
HINGLISH_SPECIAL = {
    'matlab': 'mətləb', 'accha': 'ətʃːaː', 'theek': 'ʈʰiːk',
    'kya': 'kjɑː', 'hai': 'ɦæː', 'hain': 'ɦæːn',
    'nahi': 'nəɦiː', 'aur': 'ɔːr', 'lekin': 'leːkɪn',
    'toh': 'toː', 'yeh': 'jeː', 'woh': 'ʋoː',
    'kaise': 'kæːseː', 'kyunki': 'kjʊnkɪ', 'isliye': 'ɪslɪjeː',
    'samajh': 'səmədʒʰ', 'dekho': 'deːkʰoː', 'bolo': 'boːloː',
    'suno': 'sʊnoː', 'padho': 'pəɖʰoː', 'likho': 'lɪkʰoː',
    'hum': 'ɦəm', 'tum': 'tʊm', 'main': 'mæːn',
    'karo': 'kəroː', 'karna': 'kərnɑː', 'hota': 'ɦoːtɑː',
    'bahut': 'bəɦʊt', 'zyada': 'zjɑːdɑː', 'kam': 'kəm',
    'bada': 'bəɽɑː', 'chhota': 'tʃʰoːʈɑː',
    'mera': 'meːrɑː', 'tera': 'teːrɑː', 'uska': 'ʊskɑː',
    'isme': 'ɪsmeː', 'usme': 'ʊsmeː', 'jisme': 'dʒɪsmeː',
    'phir': 'pʰɪr', 'abhi': 'əbʰiː', 'pehle': 'peːɦleː',
    'baad': 'bɑːd', 'saath': 'sɑːtʰ', 'liye': 'lɪjeː',
}


class HinglishIPAConverter:
    """Convert code-switched Hinglish text to IPA."""
    
    def __init__(self):
        self.special_words = HINGLISH_SPECIAL.copy()
        self.dev_map = DEVANAGARI_IPA.copy()
        self.roman_hindi_map = ROMAN_HINDI_IPA.copy()
        self.eng_map = ENGLISH_IPA_SIMPLE.copy()
    
    def is_devanagari(self, text: str) -> bool:
        return any('\u0900' <= c <= '\u097F' for c in text)
    
    def devanagari_to_ipa(self, text: str) -> str:
        ipa = []
        i = 0
        while i < len(text):
            # Try 2-char sequences first
            if i + 1 < len(text):
                digraph = text[i:i+2]
                if digraph in self.dev_map:
                    ipa.append(self.dev_map[digraph])
                    i += 2
                    continue
            
            char = text[i]
            if char in self.dev_map:
                ipa.append(self.dev_map[char])
            elif char == ' ':
                ipa.append(' ')
            i += 1
        
        return ''.join(ipa)
    
    def english_to_ipa(self, word: str) -> str:
        """Simple rule-based English to IPA."""
        word_lower = word.lower()
        
        # Try epitran first if available
        try:
            import epitran
            epi = epitran.Epitran('eng-Latn')
            return epi.transliterate(word)
        except (ImportError, Exception):
            pass
        
        # Fallback: rule-based
        ipa = word_lower
        # Apply multi-char rules first (longest match)
        for pattern, replacement in sorted(self.eng_map.items(), 
                                          key=lambda x: -len(x[0])):
            ipa = ipa.replace(pattern, replacement)
        
        return ipa
    
    def hindi_roman_to_ipa(self, word: str) -> str:
        """Convert Romanized Hindi to IPA."""
        word_lower = word.lower()
        
        # Check special words first
        if word_lower in self.special_words:
            return self.special_words[word_lower]
        
        # Apply Roman Hindi mappings
        ipa = word_lower
        for pattern, replacement in sorted(self.roman_hindi_map.items(),
                                          key=lambda x: -len(x[0])):
            ipa = ipa.replace(pattern, replacement)
        
        return ipa
    
    def convert_word(self, word: str, lang: str = "auto") -> str:
        """Convert a single word to IPA."""
        word = word.strip()
        if not word:
            return ""
        
        # Check special Hinglish words
        if word.lower() in self.special_words:
            return self.special_words[word.lower()]
        
        # Devanagari script
        if self.is_devanagari(word):
            return self.devanagari_to_ipa(word)
        
        # Route based on language
        if lang == "hi":
            return self.hindi_roman_to_ipa(word)
        elif lang == "en":
            return self.english_to_ipa(word)
        else:
            # Auto-detect: if word has Hindi phonological patterns
            hindi_patterns = ['aa', 'ee', 'oo', 'bh', 'dh', 'gh', 'kh', 'ph', 'th', 'ch']
            if any(p in word.lower() for p in hindi_patterns):
                return self.hindi_roman_to_ipa(word)
            return self.english_to_ipa(word)
    
    def convert_transcript(self, segments: List[Dict],
                           lid_results: List[Dict] = None) -> str:
        """
        Convert entire transcript to IPA.
        
        Args:
            segments: Transcript segments with words and timestamps
            lid_results: LID frame results for language routing
        
        Returns:
            IPA string
        """
        ipa_parts = []
        
        for seg in segments:
            words = seg.get("words", [])
            if not words:
                # Fallback: split text
                words = [{"word": w, "start": seg.get("start", 0),
                          "end": seg.get("end", 0)} 
                         for w in seg.get("text", "").split()]
            
            for word_info in words:
                word = word_info.get("word", "").strip()
                if not word:
                    continue
                
                # Determine language from LID
                lang = "auto"
                if lid_results:
                    word_time = word_info.get("start", 0)
                    for lr in lid_results:
                        if lr["start_time"] <= word_time <= lr["end_time"]:
                            lang = lr["language"]
                            break
                
                ipa = self.convert_word(word, lang)
                ipa_parts.append(ipa)
        
        return ' '.join(ipa_parts)
    
    def save_ipa(self, ipa_text: str, output_path: str):
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(ipa_text)
        print(f"[IPA] Saved to {output_path}")
