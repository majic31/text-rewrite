# coding: utf-8
"""
高性能 RAG 加速模块

使用以下技术优化检索性能：
1. Numba JIT 编译核心 DP 算法
2. 首音素倒排索引减少候选
3. 长度过滤跳过不可能的匹配
"""

import traceback
import numpy as np
from typing import List, Dict, Tuple, Set, Union
from collections import defaultdict
import time
import logging
from .algo_phoneme import Phoneme
from .algo_calc import SIMILAR_PHONEMES


import os

# 通过环境变量实现优雅切换：
# 如果环境里有 DISABLE_NUMBA=1，则主动关掉 Numba（配合 PyArmor 使用）
# 否则，默认开启 Numba（用于你日常的内网明文开发测试）
try:
    if os.environ.get("DISABLE_NUMBA", "0") == "1":
        raise ImportError("环境变量强制禁用了 Numba")
        
    from numba import jit, njit
    import numba
    HAS_NUMBA = True
    logging.debug("Numba 可用，使用 JIT 加速")
except ImportError:
    HAS_NUMBA = False
    logging.debug("Numba 不可用，使用纯 Python")


# =============================================================================
# Numba 加速版本
# =============================================================================

if HAS_NUMBA:
    @njit(cache=True)
    def _fuzzy_substring_distance_numba(main_codes: np.ndarray, sub_codes: np.ndarray) -> float:
        """
        Numba 加速的模糊子串距离计算
        
        使用整数编码代替字符串，大幅提升性能。
        """
        n = len(sub_codes)
        m = len(main_codes)
        
        if n == 0 or m == 0:
            return float(n)
        
        # DP 矩阵
        dp = np.zeros((n + 1, m + 1), dtype=np.float32)
        
        # 初始化第一列
        for i in range(1, n + 1):
            dp[i, 0] = float(i)
        
        # 填充 DP 矩阵
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                # 计算代价：相同=0，不同=1
                if sub_codes[i-1] == main_codes[j-1]:
                    cost = 0.0
                else:
                    cost = 1.0
                
                dp[i, j] = min(
                    dp[i-1, j] + 1.0,       # 删除
                    dp[i, j-1] + 1.0,       # 插入
                    dp[i-1, j-1] + cost     # 替换/匹配
                )
        
        # 找最小距离
        min_dist = dp[n, 1]
        for j in range(2, m + 1):
            if dp[n, j] < min_dist:
                min_dist = dp[n, j]
        
        return min_dist


    @njit(cache=True)
    def _fuzzy_search_constrained_numba(
        hw_codes: np.ndarray,    # (n,) int32  phoneme value 编码
        hw_langs: np.ndarray,    # (n,) int32  0=zh 1=en 2=other
        hw_is_tone: np.ndarray,  # (n,) int32  是否声调音素
        inp_codes: np.ndarray,   # (m,) int32
        inp_langs: np.ndarray,   # (m,) int32
        inp_is_ws: np.ndarray,   # (m,) int32  is_word_start
        inp_is_we: np.ndarray,   # (m,) int32  is_word_end
        zh_cost: np.ndarray,     # (N,N) float32  预计算 zh-zh 代价矩阵
        threshold: float,
    ):
        """
        Numba-加速的边界约束模糊子串搜索 DP。

        返回:
            out_scores (MAX_RES,) float64
            out_starts (MAX_RES,) int32  -- input 中 0-indexed 起始位置
            out_ends   (MAX_RES,) int32  -- DP 列索引（1-indexed，exclusive）
            n_res      int
        """
        n = len(hw_codes)
        m = len(inp_codes)
        MAX_RES = 32

        out_scores = np.zeros(MAX_RES, dtype=np.float64)
        out_starts = np.zeros(MAX_RES, dtype=np.int32)
        out_ends   = np.zeros(MAX_RES, dtype=np.int32)
        n_res = 0

        if n == 0 or m == 0:
            return out_scores, out_starts, out_ends, 0

        INF = 1e18
        mat_n = zh_cost.shape[0]

        # dp[i,j]: hw 前 i 个音素匹配到 input[j-1] 时的最小编辑距离
        dp = np.full((n + 1, m + 1), INF, dtype=np.float64)
        # sc[i,j]: 追踪匹配起始列（dp 列坐标，0-indexed）
        sc = np.zeros((n + 1, m + 1), dtype=np.int32)

        # 第 0 行：允许从词起始边界开始
        dp[0, 0] = 0.0
        sc[0, 0] = 0
        for j in range(1, m + 1):
            # inp_is_ws[j] 对应 input_info[j]（0-indexed），即 dp 列 j+1 起点
            # 与 Python 版对齐：`elif j < m and input_info[j][2]`
            if j < m and inp_is_ws[j] == 1:
                dp[0, j] = 0.0
                sc[0, j] = j

        # 填充 DP
        for i in range(1, n + 1):
            hc = hw_codes[i - 1]
            hl = hw_langs[i - 1]
            ht = hw_is_tone[i - 1]

            for j in range(1, m + 1):
                ic = inp_codes[j - 1]
                il = inp_langs[j - 1]

                # 计算替换代价
                if hl != il:
                    cost = 1.0
                elif hc == ic:
                    cost = 0.0
                elif hl == 0:  # zh-zh 且值不同
                    if ht == 1:  # hw 音素是声调，误识容忍
                        cost = 0.5
                    elif hc < mat_n and ic < mat_n:
                        cost = zh_cost[hc, ic]
                    else:
                        cost = 1.0
                else:  # en-en 或其他：精确匹配
                    cost = 1.0

                d_match = dp[i - 1, j - 1] + cost
                d_del   = dp[i - 1, j]     + 1.0
                d_ins   = dp[i,     j - 1] + 1.0

                best = d_match
                src  = 0  # 0=match/replace  1=del  2=ins
                if d_del < best:
                    best = d_del
                    src  = 1
                if d_ins < best:
                    best = d_ins
                    src  = 2

                dp[i, j] = best
                if src == 0:
                    sc[i, j] = sc[i - 1, j - 1]
                elif src == 1:
                    sc[i, j] = sc[i - 1, j]
                else:
                    sc[i, j] = sc[i, j - 1]

        # 收集词终止边界处的最优结果（每个终止位置保留最高分）
        best_sc_per_end  = np.full(m + 1, -1.0, dtype=np.float64)
        best_st_per_end  = np.zeros(m + 1,  dtype=np.int32)

        for j in range(1, m + 1):
            if inp_is_we[j - 1] == 0:
                continue
            dist = dp[n, j]
            if dist >= n * 0.8:
                continue
            score = 1.0 - dist / n
            if score < threshold:
                continue
            if score > best_sc_per_end[j]:
                best_sc_per_end[j] = score
                best_st_per_end[j] = sc[n, j]

        for j in range(1, m + 1):
            s = best_sc_per_end[j]
            if s < 0.0:
                continue
            if n_res >= MAX_RES:
                break
            out_scores[n_res] = s
            out_starts[n_res] = best_st_per_end[j]
            out_ends[n_res]   = j
            n_res += 1

        return out_scores, out_starts, out_ends, n_res


    @njit(cache=True)
    def _batch_search_constrained_numba(
        flat_codes: np.ndarray,   # 所有热词音素编码拼接 int32
        flat_langs: np.ndarray,   # int32
        flat_tone: np.ndarray,    # int32
        offsets: np.ndarray,      # (H+1,) int64  热词 h 的音素区间 [offsets[h], offsets[h+1])
        cand_ids: np.ndarray,     # (K,) int64  候选热词 id（按期望的输出顺序）
        inp_codes: np.ndarray,
        inp_langs: np.ndarray,
        inp_is_ws: np.ndarray,
        inp_is_we: np.ndarray,
        zh_cost: np.ndarray,
        threshold: float,
    ):
        """
        对一批候选热词逐个执行 _fuzzy_search_constrained_numba，
        一次 Numba 调用完成全部 DP，避免 K 次 Python 往返。

        返回 (out_cand, out_scores, out_starts, out_ends)，
        out_cand 是 cand_ids 中的下标；顺序为候选顺序、同一候选内按终止位置升序，
        与逐个调用的结果完全一致。
        """
        cap = 64
        o_k  = np.empty(cap, dtype=np.int64)
        o_sc = np.empty(cap, dtype=np.float64)
        o_st = np.empty(cap, dtype=np.int32)
        o_en = np.empty(cap, dtype=np.int32)
        cnt = 0
        for k in range(len(cand_ids)):
            h = cand_ids[k]
            a = offsets[h]
            b = offsets[h + 1]
            sc, st, en, nr = _fuzzy_search_constrained_numba(
                flat_codes[a:b], flat_langs[a:b], flat_tone[a:b],
                inp_codes, inp_langs, inp_is_ws, inp_is_we,
                zh_cost, threshold,
            )
            for r in range(nr):
                if cnt >= cap:
                    cap *= 2
                    n_k = np.empty(cap, dtype=np.int64)
                    n_sc = np.empty(cap, dtype=np.float64)
                    n_st = np.empty(cap, dtype=np.int32)
                    n_en = np.empty(cap, dtype=np.int32)
                    n_k[:cnt] = o_k[:cnt]
                    n_sc[:cnt] = o_sc[:cnt]
                    n_st[:cnt] = o_st[:cnt]
                    n_en[:cnt] = o_en[:cnt]
                    o_k, o_sc, o_st, o_en = n_k, n_sc, n_st, n_en
                o_k[cnt] = k
                o_sc[cnt] = sc[r]
                o_st[cnt] = st[r]
                o_en[cnt] = en[r]
                cnt += 1
        return o_k[:cnt], o_sc[:cnt], o_st[:cnt], o_en[:cnt]


# =============================================================================
# 音素编码器（字符串 -> 整数）
# =============================================================================

class PhonemeEncoder:
    """将音素字符串编码为整数，用于 Numba 加速"""
    
    def __init__(self):
        self.phoneme_to_code: Dict[str, int] = {}
        self.code_to_phoneme: Dict[int, str] = {}
        self.next_code = 1  # 0 保留
        
    def encode(self, phoneme: str) -> int:
        if phoneme not in self.phoneme_to_code:
            self.phoneme_to_code[phoneme] = self.next_code
            self.code_to_phoneme[self.next_code] = phoneme
            self.next_code += 1
        return self.phoneme_to_code[phoneme]
    
    def encode_sequence(self, phonemes: List[str]) -> np.ndarray:
        return np.array([self.encode(p) for p in phonemes], dtype=np.int32)


# =============================================================================
# 倒排索引
# =============================================================================

class PhonemeIndex:
    """
    多音素倒排索引
    
    按热词前几个音素分桶，检索时只匹配音素在输入中出现过的热词，减少计算量。
    - 中文：索引前两个音素（声母+韵母，即第一个字的完整拼音）
    - 英文：索引前两个音素（容错首音素识别错误，如 klaude -> Claude）
    """
    
    def __init__(self):
        self.encoder = PhonemeEncoder()
        # {首音节元组: [(热词原文, 音素编码数组), ...]}
        self.index_syl: Dict[Tuple[int, ...], List[Tuple[str, np.ndarray]]] = defaultdict(list)
        self.all_hotwords: List[Tuple[str, np.ndarray]] = []
        
    def _get_syllables(self, phonemes: List[Phoneme]) -> List[Tuple[str, ...]]:
        syls = []
        cur_syl = []
        for p in phonemes:
            if p.is_word_start:
                cur_syl = []
            if not p.is_tone:
                cur_syl.append(p.value)
            if p.is_word_end:
                if cur_syl:
                    syls.append(tuple(cur_syl))
                    cur_syl = []
        if cur_syl:
            syls.append(tuple(cur_syl))
        return syls


    def _expand_syllable(self, syl: Tuple[str, ...]) -> List[Tuple[int, ...]]:
        expanded = [[]]
        for v in syl:
            similars = {v}
            for s_set in SIMILAR_PHONEMES:
                if v in s_set:
                    similars.update(s_set)
            
            new_expanded = []
            for prefix in expanded:
                for sim in similars:
                    code = self.encoder.phoneme_to_code.get(sim)
                    if code is not None:
                        new_expanded.append(prefix + [code])
            expanded = new_expanded
            
        return [tuple(x) for x in expanded if x]
        
    def add(self, hotword: str, phonemes: List[Phoneme]):
        """添加热词到索引，内部自动决定索引哪些位置"""
        if not phonemes:
            return
        
        # 将音素对象编码为整数 ID 序列
        phoneme_strs = [p.value for p in phonemes]
        codes = self.encoder.encode_sequence(phoneme_strs)
        
        syls = self._get_syllables(phonemes)
        if not syls:
            return
            
        if len(syls) >= 2:
            key_syl = syls[0] + syls[1]
        else:
            key_syl = syls[0]
            
        key = tuple(self.encoder.encode(v) for v in key_syl)
        self.index_syl[key].append((hotword, codes))
            
        self.all_hotwords.append((hotword, codes))
        
    def get_candidates(self, input_phonemes: List[Phoneme]) -> List[Tuple[str, np.ndarray]]:
        """
        获取候选热词

        只返回索引音素在输入中出现过的热词

        Args:
            input_phonemes: 输入音素序列 (List[Phoneme])
        """
        candidates = []
        seen = set()
        
        # 获取输入中的所有音节
        input_syls = self._get_syllables(input_phonemes)
        # 1. 单音节检索（匹配长度为 1 的热词）
        for syl in input_syls:
            expanded_keys = self._expand_syllable(syl)
            for key in expanded_keys:
                for hw, codes in self.index_syl.get(key, []):
                    if hw not in seen:
                        candidates.append((hw, codes))
                        seen.add(hw)

        # 2. 双音节检索（匹配长度 >= 2 的热词）
        for i in range(len(input_syls) - 1):
            syl_bigram = input_syls[i] + input_syls[i+1]
            expanded_keys = self._expand_syllable(syl_bigram)
            for key in expanded_keys:
                for hw, codes in self.index_syl.get(key, []):
                    if hw not in seen:
                        candidates.append((hw, codes))
                        seen.add(hw)

        return candidates
    
    def encode_input(self, phonemes: List[Phoneme]) -> np.ndarray:
        """编码输入序列"""
        phoneme_strs = [p.value for p in phonemes]
        return self.encoder.encode_sequence(phoneme_strs)

    def get_hw_info_cache(self) -> Dict[str, List]:
        """返回热词的音素 info 列表缓存，供精筛层预编码使用。"""
        return {hw: [(self.encoder.phoneme_to_code.get(p.value, 0),) for p in phonemes]
                for hw, (_, phonemes) in {}
                .items()}  # placeholder, see FastRAG.add_hotwords for actual usage


# =============================================================================
# Numba 加速精筛（NumbaSubstringSearch）
# =============================================================================

class NumbaSubstringSearch:
    """
    Numba 加速的精细子串搜索。

    将音素 info 元组编码为 numpy 数组后调用 @njit DP 函数，
    与 algo_calc.fuzzy_substring_search_constrained 接口兼容。
    """

    def __init__(self, encoder: PhonemeEncoder):
        self.encoder = encoder
        self._zh_cost: np.ndarray = None
        self._cost_built_at: int = -1
        
        self._hw_keys = []
        self._hw_key_to_id = {}
        self._flat_codes = None
        self._flat_langs = None
        self._flat_tone = None
        self._offsets = None
        
        self._warmup_numba()

    def _warmup_numba(self):
        """预热 Numba JIT 编译器，避免首次检索时的数百毫秒延迟毛刺"""
        if not HAS_NUMBA:
            return
        try:
            dummy_1_i32 = np.array([0], dtype=np.int32)
            dummy_2_i64 = np.array([0, 1], dtype=np.int64)
            dummy_1_i64 = np.array([0], dtype=np.int64)
            dummy_mat = np.zeros((1, 1), dtype=np.float32)
            
            _batch_search_constrained_numba(
                dummy_1_i32, dummy_1_i32, dummy_1_i32,
                dummy_2_i64, dummy_1_i64,
                dummy_1_i32, dummy_1_i32, dummy_1_i32, dummy_1_i32,
                dummy_mat, 0.6
            )
        except Exception as e:
            logging.warning(f"Numba warmup failed (non-fatal): {traceback.format_exc()}")


    def _ensure_zh_cost(self):
        """构建或更新 zh-zh 代价矩阵（按需扩展）。"""
        n = self.encoder.next_code
        if self._zh_cost is not None and n <= self._cost_built_at:
            return

        mat = np.ones((n, n), dtype=np.float32)
        for i in range(n):
            mat[i, i] = 0.0

        for s_set in SIMILAR_PHONEMES:
            codes = [
                self.encoder.phoneme_to_code[v]
                for v in s_set
                if v in self.encoder.phoneme_to_code
            ]
            for c1 in codes:
                for c2 in codes:
                    if c1 != c2:
                        mat[c1, c2] = 0.5

        self._zh_cost = mat
        self._cost_built_at = n

    def _encode(self, info_tuples):
        """将 info 元组列表编码为 5 个 numpy 向量。"""
        n = len(info_tuples)
        codes   = np.zeros(n, dtype=np.int32)
        langs   = np.zeros(n, dtype=np.int32)
        is_tone = np.zeros(n, dtype=np.int32)
        is_ws   = np.zeros(n, dtype=np.int32)
        is_we   = np.zeros(n, dtype=np.int32)
        for i, t in enumerate(info_tuples):
            codes[i]   = self.encoder.encode(t[0])
            l = t[1]
            langs[i]   = 0 if l == 'zh' else (1 if l == 'en' else 2)
            is_ws[i]   = 1 if t[2] else 0
            is_we[i]   = 1 if t[3] else 0
            is_tone[i] = 1 if (len(t) > 4 and t[4]) else 0
        return codes, langs, is_tone, is_ws, is_we

    def encode_input_vecs(self, input_info) -> tuple:
        """将 input_info 编码为 5 个 numpy 向量，供外部缓存复用。"""
        return self._encode(input_info)

    def search_with_encoded_input(
        self,
        hw_info,
        inp_codes: np.ndarray,
        inp_langs: np.ndarray,
        inp_is_ws: np.ndarray,
        inp_is_we: np.ndarray,
        threshold: float = 0.6,
    ) -> List[Tuple[float, int, int]]:
        """
        精筛：input 已预编码，只对热词部分重新编码。
        避免在 for-hotword 循环中反复编码同一份 input，性能提升为 O(1) input 编码。
        """
        if not HAS_NUMBA:
            return None

        hw_codes, hw_langs, hw_is_tone, _, _ = self._encode(hw_info)

        self._ensure_zh_cost()
        max_code = int(max(
            hw_codes.max() if len(hw_codes) else 0,
            inp_codes.max() if len(inp_codes) else 0,
        ))
        if max_code >= self._zh_cost.shape[0]:
            self._ensure_zh_cost()

        scores, starts, ends, n_res = _fuzzy_search_constrained_numba(
            hw_codes, hw_langs, hw_is_tone,
            inp_codes, inp_langs, inp_is_ws, inp_is_we,
            self._zh_cost, threshold,
        )
        return [
            (float(scores[i]), int(starts[i]), int(ends[i]))
            for i in range(n_res)
        ]

    def build_cache(self, hw_info_dict: Dict[str, List]):
        """预先将所有热词编码成一维大数组，供批量查询"""
        self._hw_keys = list(hw_info_dict.keys())
        self._hw_key_to_id = {k: i for i, k in enumerate(self._hw_keys)}
        
        flat_codes = []
        flat_langs = []
        flat_tone = []
        offsets = [0]
        
        for k in self._hw_keys:
            hw_info = hw_info_dict[k]
            codes, langs, tone, _, _ = self._encode(hw_info)
            flat_codes.extend(codes)
            flat_langs.extend(langs)
            flat_tone.extend(tone)
            offsets.append(len(flat_codes))
            
        self._flat_codes = np.array(flat_codes, dtype=np.int32)
        self._flat_langs = np.array(flat_langs, dtype=np.int32)
        self._flat_tone = np.array(flat_tone, dtype=np.int32)
        self._offsets = np.array(offsets, dtype=np.int64)

    def search_batch_with_encoded_input(
        self,
        candidates_keys: List[str],
        inp_codes: np.ndarray,
        inp_langs: np.ndarray,
        inp_is_ws: np.ndarray,
        inp_is_we: np.ndarray,
        threshold: float = 0.6,
    ) -> Dict[str, List[Tuple[float, int, int]]]:
        """批量对多个候选热词进行 DP，仅有一次 Python 到 Numba 的调用开销"""
        if not HAS_NUMBA:
            return {}
            
        cand_ids = []
        valid_keys = []
        for k in candidates_keys:
            if k in self._hw_key_to_id:
                cand_ids.append(self._hw_key_to_id[k])
                valid_keys.append(k)
                
        if not cand_ids:
            return {}
            
        cand_ids = np.array(cand_ids, dtype=np.int64)
        
        self._ensure_zh_cost()
        max_code = int(max(
            self._flat_codes.max() if len(self._flat_codes) else 0,
            inp_codes.max() if len(inp_codes) else 0,
        ))
        if max_code >= self._zh_cost.shape[0]:
            self._ensure_zh_cost()
            
        o_k, o_sc, o_st, o_en = _batch_search_constrained_numba(
            self._flat_codes, self._flat_langs, self._flat_tone, self._offsets, cand_ids,
            inp_codes, inp_langs, inp_is_ws, inp_is_we,
            self._zh_cost, threshold,
        )
        
        res = defaultdict(list)
        for i in range(len(o_k)):
            k_idx = o_k[i]
            hw_key = valid_keys[k_idx]
            res[hw_key].append((float(o_sc[i]), int(o_st[i]), int(o_en[i])))
            
        return res

    def search(
        self, hw_info, input_info, threshold: float = 0.6
    ) -> List[Tuple[float, int, int]]:
        """
        执行 Numba 精细搜索（保留原接口，内部复用 search_with_encoded_input）。
        若 Numba 不可用则返回 None。
        """
        if not HAS_NUMBA:
            return None

        inp_codes, inp_langs, _, inp_is_ws, inp_is_we = self._encode(input_info)
        return self.search_with_encoded_input(
            hw_info, inp_codes, inp_langs, inp_is_ws, inp_is_we, threshold
        )


# =============================================================================
# 高性能 RAG 检索器
# =============================================================================

class FastRAG:
    """
    高性能 RAG 检索器
    
    特点：
    1. Numba JIT 加速核心算法
    2. 首音素倒排索引减少候选
    3. 长度过滤跳过不可能匹配
    """
    
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        self.index = PhonemeIndex()
        self.hotword_count = 0
        # 精筛加速器（共享同一个 PhonemeEncoder）
        self.numba_searcher: NumbaSubstringSearch = NumbaSubstringSearch(self.index.encoder)
        
    def add_hotwords(self, hotwords: Dict[str, List[Phoneme]]):
        """
        批量添加热词
        
        Args:
            hotwords: {热词原文: 音素序列} (key: str, value: List[Phoneme])
        """
        for hw, phonemes in hotwords.items():
            if phonemes:
                self.index.add(hw, phonemes)
                self.hotword_count += 1
                
    def search(self, input_phonemes: List[Phoneme], top_k: int = 10) -> List[Tuple[str, float]]:
        """
        检索相关热词（高层编排）
        """
        if not input_phonemes: return []

        # DEBUG
        logging.debug(f"[DEBUG] FastRAG.search: input_phonemes type={type(input_phonemes)}, len={len(input_phonemes)}")
        if input_phonemes:
            logging.debug(f"[DEBUG] FastRAG.search: input_phonemes[0] type={type(input_phonemes[0])}, value={input_phonemes[0]}")

        # 1. 编码输入并获取候选
        input_codes = self.index.encode_input(input_phonemes)
        candidates = self.index.get_candidates(input_phonemes)

        # 2. 遍历打分与过滤
        results = self._score_candidates(input_codes, candidates)

        # 3. 排序并截断
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _score_candidates(self, input_codes: np.ndarray, candidates: List[Tuple[str, np.ndarray]]) -> List[Tuple[str, float]]:
        """对候选列表进行相似度计算与阈值过滤"""
        results = []
        input_len = len(input_codes)
        
        for hw, hw_codes in candidates:
            hw_len = len(hw_codes)
            
            # 长度过滤：热词太长或太短都不可能匹配
            if hw_len > input_len + 3: continue
            
            # 计算距离
            if HAS_NUMBA:
                min_dist = _fuzzy_substring_distance_numba(input_codes, hw_codes)
            else:
                min_dist = self._python_distance(input_codes, hw_codes)
            
            # 计算分数 (1 - 归一化距离)
            score = 1.0 - (min_dist / hw_len)
            if score >= self.threshold:
                results.append((hw, round(score, 3)))
        return results

    def compute_score(self, input_phonemes: List[str], hotword_phonemes: List[str]) -> float:
        """
        计算单个热词的精确分数 (用于重排序)
        """
        input_codes = self.index.encode_input(input_phonemes)
        hw_codes = self.index.encode_input(hotword_phonemes)
        
        hw_len = len(hw_codes)
        if hw_len == 0:
            return 0.0
            
        if HAS_NUMBA:
            min_dist = _fuzzy_substring_distance_numba(input_codes, hw_codes)
        else:
            min_dist = self._python_distance(input_codes, hw_codes)
            
        return max(0.0, 1.0 - (min_dist / hw_len))

    
    def _python_distance(self, main_codes: np.ndarray, sub_codes: np.ndarray) -> float:
        """纯 Python 版本（Numba 不可用时）"""
        n = len(sub_codes)
        m = len(main_codes)
        
        if n == 0 or m == 0:
            return float(n)
        
        dp = [[0.0] * (m + 1) for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            dp[i][0] = float(i)
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0.0 if sub_codes[i-1] == main_codes[j-1] else 1.0
                dp[i][j] = min(
                    dp[i-1][j] + 1.0,
                    dp[i][j-1] + 1.0,
                    dp[i-1][j-1] + cost
                )
        
        return min(dp[n][j] for j in range(1, m + 1))


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    import random
    from .algo_phoneme import get_phoneme_seq
    
    logging.basicConfig(level=logging.INFO)
    
    print(f"\n=== 高性能 RAG 测试 ===")
    print(f"Numba 可用: {HAS_NUMBA}")
    
    # 生成测试数据
    chinese_chars = '的一是不了在人有我他这个们中来上大为和国地到以说时要就出会可也你对生能而子那得于着下自之年过发后作里如等'
    
    print("\n生成 1000 个热词...")
    hotwords = {}
    for i in range(1000):
        length = random.randint(2, 4)
        word = ''.join(random.choice(chinese_chars) for _ in range(length))
        phonemes = get_phoneme_seq(word)
        hotwords[word] = phonemes
    
    # 创建 FastRAG
    print("构建索引...")
    start = time.time()
    rag = FastRAG(threshold=0.8)
    rag.add_hotwords(hotwords)
    print(f"  索引构建耗时: {time.time() - start:.3f}s")
    
    # 生成输入
    input_text = ''.join(random.choice(chinese_chars) for _ in range(100))
    input_phonemes = get_phoneme_seq(input_text)
    print(f"\n输入: {input_text[:50]}... ({len(input_text)}字, {len(input_phonemes)}音素)")
    
    # 预热 Numba
    if HAS_NUMBA:
        print("\n预热 Numba JIT...")
        _ = rag.search(input_phonemes[:10], top_k=3)
    
    # 测试性能
    print("\n测试检索性能...")
    start = time.time()
    results = rag.search(input_phonemes, top_k=10)
    elapsed = time.time() - start
    
    print(f"  检索耗时: {elapsed:.3f}s")
    print(f"  热词总数: {rag.hotword_count}")
    print(f"  候选数量: {len(rag.index.get_candidates(input_phonemes))}")
    print(f"  结果: {results[:5]}")
