import logging
import re
from typing import List
from collections import defaultdict
from .base import BaseFilter
from .fuzzy_phoneme import FuzzyPhonemeFilter
logger = logging.getLogger(__name__)

class EntityAwareFuzzyFilter(BaseFilter):
    """
    SOTA Entity-Aware Fuzzy Phoneme Filter.
    Combines high-speed global phoneme matching with Jieba-based NER tagging
    to correct ASR homophone errors accurately and efficiently.
    """
    def __init__(self, rules: List[str] = None, use_hmm: bool = False, name: str = None):
        """
        :param rules: A list of rule strings. 
                      Format: `[tag]word1:threshold1|word2:replacement2:threshold2`
                      tag, replacement, and threshold are optional.
                      Example: "[nr]叶开:0.8|张三:0.7", "李四:0.7", "欧阳锋"
        :param use_hmm: Whether to enable Jieba's HMM feature. Enable this to discover unknown entities,
                        at the cost of some performance (~50ms latency vs 5ms when False).
        """
        super().__init__(name=name)
        self.use_hmm = use_hmm
        self.global_hotwords = {}
        self.global_thresholds = {}
        self.global_weights = {}
        self.tagged_hotwords = defaultdict(dict)
        self.tagged_weights = defaultdict(dict)
        
        self.global_filter = None
        self.tagged_filters = {}
        self.ner_filter = None
        
        if rules:
            self.add_rules(rules)
            
    def add_rules(self, rules: List[str]):
        tag_pattern = re.compile(r'^\[([^\]]+)\]')
        for rule in rules:
            self._parse_rule(rule, tag_pattern)
                
        self._rebuild_filters()
        self._setup_ner_filter()

    def _parse_rule(self, rule: str, tag_pattern: re.Pattern):
        rule = rule.strip()
        if not rule:
            return
            
        tag_match = tag_pattern.match(rule)
        tag = tag_match.group(1) if tag_match else None
        
        if tag_match:
            rule = rule[tag_match.end():]
            
        parts = [p.strip() for p in rule.split('|')]
        for part in parts:
            if not part:
                continue
                
            subparts = [sp.strip() for sp in part.split(':')]
            word = subparts[0]
            replacement = word
            weight = 1.0
            
            if len(subparts) == 2:
                try:
                    weight = float(subparts[1])
                except ValueError:
                    replacement = subparts[1]
            elif len(subparts) >= 3:
                replacement = subparts[1]
                try:
                    weight = float(subparts[2])
                except ValueError:
                    pass
            
            if tag:
                self.tagged_hotwords[tag][word] = replacement
                self.tagged_weights[tag][word] = weight
            else:
                self.global_hotwords[word] = replacement
                self.global_weights[word] = weight

    def _rebuild_filters(self):
        # Rebuild global filter
        if self.global_hotwords:
            self.global_filter = FuzzyPhonemeFilter(hotwords=self.global_hotwords, threshold=0.6)
            self.global_filter.custom_weights = self.global_weights
            
        # Rebuild tagged filters
        for tag, hw_dict in self.tagged_hotwords.items():
            f = FuzzyPhonemeFilter(hotwords=hw_dict, threshold=0.6)
            f.custom_weights = self.tagged_weights[tag]
            self.tagged_filters[tag] = f

    def _setup_ner_filter(self):
        # Init JiebaNERFilter if tagged rules exist
        if not self.tagged_filters:
            return

        import functools
        
        # Cache the processing of individual words to avoid redundant phoneme DP searches
        # Since tagged filters don't change state during processing, caching per tag is safe.
        word_cache = {}
        for tag in self.tagged_filters:
            word_cache[tag] = functools.lru_cache(maxsize=4096)(self.tagged_filters[tag].process)
            
        def jieba_callback(terms, text):
            new_text = text
            
            # Keep track of character offsets manually since jieba
            # splits continuously.
            offset = 0
            entities = []
            for term in terms:
                word = term.word
                tag_str = term.flag
                
                if tag_str in self.tagged_filters:
                    entities.append({
                        'word': word,
                        'tag': tag_str,
                        'start': offset,
                        'end': offset + len(word)
                    })
                offset += len(word)
            
            # Process from back to front to avoid index shifting after string replacement
            for ent in reversed(entities):
                word = ent['word']
                tag = ent['tag']
                start = ent['start']
                end = ent['end']
                
                # Fuzzy phoneme search on the isolated entity word (Cached)
                corrected = word_cache[tag](word)
                if corrected != word:
                    new_text = new_text[:start] + corrected + new_text[end:]
                    
            return new_text

        from .jieba_ner import JiebaNERFilter
        self.ner_filter = JiebaNERFilter(replace_callback=jieba_callback, use_hmm=self.use_hmm)
        self.ner_filter._load_model() # Pre-load it
        
        # Build prechecker
        all_tagged_hw = {}
        all_tagged_weights = {}
        for tag, hw_dict in self.tagged_hotwords.items():
            all_tagged_hw.update(hw_dict)
            all_tagged_weights.update(self.tagged_weights[tag])
            
        if all_tagged_hw:
            self.ner_prechecker = FuzzyPhonemeFilter(hotwords=all_tagged_hw, threshold=0.6)
            self.ner_prechecker.custom_weights = all_tagged_weights
        else:
            self.ner_prechecker = None

    def process(self, text: str) -> str:
        """
        Execute SOTA cascading replacement.
        1. Global Numba match (fastest, for long words or safe thresholds).
        2. Jieba NER constraint match (for tags like 'nr').
        """
        if not text:
            return text

            
        # Step 1: Global substitution
        if self.global_filter:
            text = self.global_filter.process(text)
            
        # Step 2: Tag-constrained substitution
        if self.ner_filter:
            # 预检：如果不可能有带标签的热词出现，直接跳过 jieba 分词
            skip_ner = False
            if hasattr(self, 'ner_prechecker') and self.ner_prechecker:
                from text_rewrite.utils.algo_phoneme import get_phoneme_info
                inp = get_phoneme_info(text)
                cands = self.ner_prechecker.rag.index.get_candidates(inp)
                if not cands:
                    skip_ner = True
            
            if not skip_ner:
                text = self.ner_filter.process(text)
            
        return text
