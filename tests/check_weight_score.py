import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from text_rewrite.filters.fuzzy_phoneme import FuzzyPhonemeFilter
from text_rewrite.utils.algo_phoneme import get_phoneme_info

def analyze_match(target_word: str, test_text: str, weight: float = 1.0):
    print(f"\n{'='*50}")
    print(f"🎯 目标词汇 (Target): '{target_word}'")
    print(f"📝 测试文本 (Input) : '{test_text}'")
    print(f"⚖️  配置权重 (Weight): {weight}")
    
    # 引擎底层基础分为 0.6
    base_threshold = 0.6
    effective_threshold = base_threshold / weight if weight > 0 else 1.0
    print(f"📏 对应的及格线门槛 : 相似度必须 >= {effective_threshold:.4f} 才能触发替换")
    
    # 初始化底层引擎
    f = FuzzyPhonemeFilter(hotwords={target_word: target_word})
    
    # 准备拼音特征
    input_phonemes = get_phoneme_info(test_text)
    input_info = [p.info for p in input_phonemes]
    
    # 调用底层 Numba 引擎，阈值设为 0.0，强行把所有的分数都吐出来
    inp_codes, inp_langs, _, inp_is_ws, inp_is_we = f.rag.numba_searcher.encode_input_vecs(input_info)
    batch_res = f.rag.numba_searcher.search_batch_with_encoded_input(
        [target_word], inp_codes, inp_langs, inp_is_ws, inp_is_we, threshold=0.0
    )
    
    matches = batch_res.get(target_word, [])
    if not matches:
        print("\n❌ 引擎结论: 读音相差十万八千里，没有任何沾边的发音片段！")
        return
        
    print("\n🔍 引擎扫描到的疑似片段及得分:")
    
    # 为了避免打印太多干扰项，只取最高分的一个展示
    matches.sort(key=lambda x: x[0], reverse=True)
    best_match = matches[0]
    
    score, start_idx, end_idx = best_match
    
    # 还原匹配到的文本
    if start_idx < len(input_phonemes) and (end_idx - 1) < len(input_phonemes):
        char_start = input_phonemes[start_idx].char_start
        char_end = input_phonemes[end_idx - 1].char_end
        matched_text = test_text[char_start:char_end]
    else:
        matched_text = "未知片段"
        
    print(f"   ► 锁定片段: '{matched_text}'")
    print(f"   ► 实际得分: {score:.4f}")
    
    if score >= effective_threshold:
        print(f"✅ 引擎结论: 实际得分 {score:.4f} >= 及格线 {effective_threshold:.4f}，成功触发替换！")
    else:
        print(f"❌ 引擎结论: 实际得分 {score:.4f} < 及格线 {effective_threshold:.4f}，未达到替换门槛！")
    print(f"{'='*50}")

if __name__ == "__main__":
    # 场景 1：权重设为 0.6 (要求极其严格的完全匹配)
    analyze_match(target_word="李四", test_text="里死", weight=0.6)
    
    # 场景 2：权重设为 0.7 (稍微放宽容错度)
    analyze_match(target_word="李四", test_text="里死", weight=0.7)
    
    # 场景 3：完全同音同调的完美匹配
    analyze_match(target_word="李四", test_text="理似", weight=0.6)
    
    # 场景 4：发音差异很大的情况 (测试张三和展伞)
    analyze_match(target_word="张三", test_text="展伞", weight=0.7)
    analyze_match(target_word="张三", test_text="展伞", weight=1.0)

    analyze_match(target_word="安康", test_text="安刊", weight=0.7)
