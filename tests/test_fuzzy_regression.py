# tests/test_fuzzy_regression.py
"""
模糊音素纠错的回归测试，覆盖 v0.2.1 code review 中发现的 3 个问题：
  1. 热词侧与输入侧音素化不一致（英文 / 多音字 / 轻声）
  2. 同一热词在长文本中最多替换 32 次
  3. DISABLE_NUMBA=1 时模糊纠错完全失效
"""
import os
import subprocess
import sys

import pytest

from text_rewrite.filters import FuzzyPhonemeFilter
from text_rewrite.utils.algo_phoneme import get_phoneme_info

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- 1. 音素化一致性
@pytest.mark.parametrize("hotword, text, expected", [
    ("Claude", "I use claude today", "I use Claude today"),
    ("GitHub", "push to github now", "push to GitHub now"),
    ("重庆", "我去冲庆玩", "我去重庆玩"),        # 多音字：词组注音 chong2
    ("叶开", "业开来了", "叶开来了"),
    ("撒贝宁", "今天萨贝宁来了", "今天撒贝宁来了"),
])
def test_fuzzy_basic_cases(hotword, text, expected):
    assert FuzzyPhonemeFilter({hotword: hotword}).process(text) == expected


def test_hotword_and_input_phonemes_identical():
    """热词与同样文本作为输入时，音素序列必须一致（含轻声、多音字）。"""
    f = FuzzyPhonemeFilter()
    for word in ["好吗", "我的", "重庆", "GitHub", "iPhone15"]:
        f.add_hotwords({word: word})
        hw_vals = [t[0] for t in f._hw_info_cache[word]]
        inp_vals = [p.value for p in get_phoneme_info(word)]
        assert hw_vals == inp_vals, word


def test_exact_hotword_scores_one():
    """热词原样出现在输入中时应完全匹配（score == 1.0），且不改变文本。"""
    for word in ["好吗", "重庆", "Claude"]:
        f = FuzzyPhonemeFilter({word: word})
        inp = get_phoneme_info(word)
        enc = f.rag.numba_searcher.encode_input_vecs([p.info for p in inp])
        res = f.rag.numba_searcher.search_batch_with_encoded_input(
            [word], enc[0], enc[1], enc[3], enc[4], threshold=0.6)
        assert max(s for s, _, _ in res[word]) == pytest.approx(1.0), word
        assert f.process(word) == word


# ---------------------------------------------------------------- 2. 长文本不截断
@pytest.mark.parametrize("repeat", [1, 32, 33, 100])
def test_no_truncation_on_long_text(repeat):
    f = FuzzyPhonemeFilter({"叶开": "叶开"})
    out = f.process("，".join(["业开来了"] * repeat))
    assert out.count("叶开") == repeat
    assert "业开" not in out


# ---------------------------------------------------------------- 3. 无 Numba 降级
_FALLBACK_SCRIPT = r"""
import json, logging
logging.disable(logging.CRITICAL)
from text_rewrite.utils import rag_fast
from text_rewrite.filters import FuzzyPhonemeFilter
cases = [("叶开", "业开来了，" * 40), ("重庆", "我去冲庆玩"), ("Claude", "I use claude"), ("张三", "展伞和里死")]
print(json.dumps({"has_numba": rag_fast.HAS_NUMBA,
                  "out": [FuzzyPhonemeFilter({h: h}).process(t) for h, t in cases]}, ensure_ascii=False))
"""


def _run(disable):
    env = dict(os.environ, DISABLE_NUMBA="1" if disable else "0")
    r = subprocess.run([sys.executable, "-c", _FALLBACK_SCRIPT], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    import json
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_disable_numba_matches_jit():
    """纯 Python 降级路径必须与 Numba JIT 路径输出完全一致。"""
    py, jit = _run(disable=True), _run(disable=False)
    assert py["has_numba"] is False
    assert py["out"] == jit["out"]
    assert py["out"][0].count("叶开") == 40
