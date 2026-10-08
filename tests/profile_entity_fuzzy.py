"""
Profile EntityAwareFuzzyFilter: 统计候选数 K 及各阶段耗时。
用法: python tests/profile_entity_fuzzy.py
"""
import time
import random
import sys
import os
import logging
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jieba
from text_rewrite.filters.entity_fuzzy import EntityAwareFuzzyFilter
from text_rewrite.utils.algo_phoneme import get_phoneme_info

logging.basicConfig(level=logging.WARNING)

CHARS = '的一是不了在人有我他这个们中来上大为和国地到以说时要就出会可也你对生能而子那得于着下自之年过发后作里如等'
SHORT = "昨天通知了一下展伞和里死，但是他也开心了，最后通知了也开。"


def build_filter(num_fuzzy):
    random.seed(43)
    rules = [f"{''.join(random.choice(CHARS) for _ in range(2))}:0.7" for _ in range(num_fuzzy)]
    rules += ["张三:0.7", "[nr]叶开:0.8", "李四:0.7"]
    jieba.add_word('也开', tag='nr')
    return EntityAwareFuzzyFilter(rules=rules)


def build_long_text():
    random.seed(7)
    s = ""
    for _ in range(50):
        s += SHORT + "".join(random.choice(CHARS) for _ in range(10))
    return s


def profile_global(f, text, iters):
    """手工拆解 FuzzyPhonemeFilter.process 的各阶段"""
    g = f.global_filter
    rag = g.rag
    ns = rag.numba_searcher
    t = defaultdict(float)
    K = 0
    n_hits = 0
    for _ in range(iters):
        t0 = time.perf_counter()
        inp = get_phoneme_info(text)
        t1 = time.perf_counter()
        cands = rag.index.get_candidates(inp)
        t2 = time.perf_counter()
        info = [p.info for p in inp]
        codes, langs, _, ws, we = ns.encode_input_vecs(info)
        t3 = time.perf_counter()
        enc_t = 0.0
        dp_t = 0.0
        for hw_key, _ in cands:
            hw_info = g._hw_info_cache.get(hw_key)
            if not hw_info:
                continue
            a = time.perf_counter()
            hc, hl, ht, _, _ = ns._encode(hw_info)
            ns._ensure_zh_cost()
            b = time.perf_counter()
            from text_rewrite.utils.rag_fast import _fuzzy_search_constrained_numba
            r = _fuzzy_search_constrained_numba(hc, hl, ht, codes, langs, ws, we, ns._zh_cost, 0.6)
            c = time.perf_counter()
            enc_t += b - a
            dp_t += c - b
            n_hits += r[3]
        t4 = time.perf_counter()
        t['phoneme'] += t1 - t0
        t['candidates'] += t2 - t1
        t['enc_input'] += t3 - t2
        t['enc_hw'] += enc_t
        t['numba_dp'] += dp_t
        t['loop_overhead'] += (t4 - t3) - enc_t - dp_t
        K = len(cands)
    return {k: v * 1000 / iters for k, v in t.items()}, K, len(inp), n_hits // iters


def main():
    long_text = build_long_text()
    for n in [500, 3000, 6000, 10000]:
        f = build_filter(n)
        f.process(SHORT)  # warmup (Numba JIT + jieba)
        total_hw = len(f.global_filter._hw_info_cache)
        for label, text, iters in [("短文本", SHORT, 20), ("长文本", long_text, 3)]:
            t0 = time.perf_counter()
            for _ in range(iters):
                f.process(text)
            e2e = (time.perf_counter() - t0) * 1000 / iters

            t0 = time.perf_counter()
            for _ in range(iters):
                f.global_filter.process(text)
            g_ms = (time.perf_counter() - t0) * 1000 / iters

            stages, K, m, hits = profile_global(f, text, iters)
            print(f"\n[{n} 热词 | {label} {len(text)}字 / {m}音素] 端到端 {e2e:.1f}ms "
                  f"(全局路 {g_ms:.1f}ms, NER路 {e2e - g_ms:.1f}ms)")
            print(f"  候选 K = {K} / {total_hw} ({K / total_hw:.0%}), DP命中 {hits}")
            for k, v in stages.items():
                print(f"  {k:<14}{v:8.2f} ms")


if __name__ == "__main__":
    main()
