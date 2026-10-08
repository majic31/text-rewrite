import logging
from typing import Dict
from .base import BaseFilter
from ..utils.algo_phoneme import get_phoneme_info, get_phoneme_seq
from ..utils.rag_fast import FastRAG

logger = logging.getLogger(__name__)

class FuzzyPhonemeFilter(BaseFilter):
    """
    Fuzzy Phoneme Filter using Numba JIT DP acceleration.
    Handles exact and fuzzy phoneme matches to correct ASR homophone errors.
    """
    def __init__(self, name: str = "FuzzyPhonemeFilter", hotwords: Dict[str, str] = None, threshold: float = 0.6):
        """
        :param threshold: The similarity threshold (0.0 to 1.0) for a phoneme sequence to match.
                          0.6 is a good default for allowing some ASR errors.
        """
        super().__init__(name=name)
        self.rag = FastRAG(threshold=threshold)
        self.hotword_replacements = {}
        # Cache hw_info to avoid recomputing
        self._hw_info_cache = {}
        
        # Optional mapping of original word -> specific threshold
        self.custom_thresholds = {}
        
        if hotwords:
            self.add_hotwords(hotwords)
            
    def add_hotwords(self, hotwords: Dict[str, str], custom_thresholds: Dict[str, float] = None):
        """
        Add hotwords dynamically.
        :param hotwords: Dictionary mapping the phonetic target (e.g., "张三") to the replacement string (e.g., "张三").
        :param custom_thresholds: Optional dictionary mapping target word to its specific threshold.
        """
        if custom_thresholds:
            self.custom_thresholds.update(custom_thresholds)
            
        hw_dict = {}
        for original, replacement in hotwords.items():
            phonemes = get_phoneme_seq(original)
            hw_dict[original] = phonemes
            self.hotword_replacements[original] = replacement
            self._hw_info_cache[original] = [p.info for p in phonemes]
        
        self.rag.add_hotwords(hw_dict)
        self.rag.numba_searcher.build_cache(self._hw_info_cache)

    def process(self, text: str) -> str:
        """
        Process text by converting to phonemes, searching for fuzzy matches, 
        mapping indices back, and replacing.
        """
        if not text:
            return text
            
        # 1. Get input phonemes with char_start/char_end
        input_phonemes = get_phoneme_info(text)
        if not input_phonemes:
            return text
            
        # 2. Get candidates from inverted index to avoid full scan
        candidates = self.rag.index.get_candidates(input_phonemes)
        if not candidates:
            return text
            
        input_info = [p.info for p in input_phonemes]
        
        # Find the absolute minimum threshold to query Numba, post-filter later
        min_thresh = self.rag.threshold
        if self.custom_thresholds:
            min_thresh = min(min_thresh, min(self.custom_thresholds.values()))
            
        # 3. Perform fine-grained Numba Substring DP search
        replacements = []
        
        # Pre-encode input_info ONCE to avoid O(N_candidates) encoding overhead
        inp_codes, inp_langs, _, inp_is_ws, inp_is_we = self.rag.numba_searcher.encode_input_vecs(input_info)
        
        candidate_keys = [hw_key for hw_key, _ in candidates]
        
        batch_res = self.rag.numba_searcher.search_batch_with_encoded_input(
            candidate_keys, inp_codes, inp_langs, inp_is_ws, inp_is_we, threshold=min_thresh
        )
        
        for hw_key, res_list in batch_res.items():
            for score, start_idx, end_idx in res_list:
                # Apply word-specific threshold
                if score >= self.custom_thresholds.get(hw_key, self.rag.threshold):
                    replacements.append({
                        'score': score,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'replacement': self.hotword_replacements[hw_key]
                    })
        
        if not replacements:
            return text
            
        # 4. Resolve overlaps (greedy: highest score, then longest span)
        replacements.sort(key=lambda x: (-x['score'], -(x['end_idx'] - x['start_idx'])))
        
        final_reps = []
        used_indices = set()
        for r in replacements:
            r_set = set(range(r['start_idx'], r['end_idx']))
            if not r_set.intersection(used_indices):
                final_reps.append(r)
                used_indices.update(r_set)
                
        # 5. Execute replacements from back to front to avoid index shifting issues
        final_reps.sort(key=lambda x: x['start_idx'], reverse=True)
        
        result_text = text
        for r in final_reps:
            # Map phoneme index to char index
            # end_idx is exclusive in phoneme array
            start_p = r['start_idx']
            end_p = r['end_idx'] - 1
            
            if start_p >= len(input_phonemes) or end_p >= len(input_phonemes) or start_p > end_p:
                continue
                
            char_start = input_phonemes[start_p].char_start
            char_end = input_phonemes[end_p].char_end
            
            result_text = result_text[:char_start] + r['replacement'] + result_text[char_end:]
            
        return result_text
