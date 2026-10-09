import time
import logging
import sys
import os
import jieba
# 确保能找到 text_rewrite 包
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from text_rewrite.pipeline import Pipeline
from text_rewrite.filters.regex import RegexFilter
from text_rewrite.filters.hotword import HotwordFilter
from text_rewrite.filters.entity_fuzzy import EntityAwareFuzzyFilter

logging.basicConfig(level=logging.WARNING)

def run_demo():
    print("=== TextRewrite 完整流水线综合测试演示 ===\n")

    # ---------------------------------------------------------
    # 第一步：构建各类过滤器
    # ---------------------------------------------------------
    
    # 1. 正则过滤器 (用于前置清洗)
    print("[1/3] 正在初始化 RegexFilter...")
    regex_rules = {
        r"\b(那个|呃|嗯|啊|哦)\b": "",  # 去除无意义的口水词
        # r"\s+": ""                      # 去除所有多余空格
        r"try to": "尝试去"              # 正则匹配并替换为想要的文字
    }
    regex_filter = RegexFilter(rules=regex_rules)
    
    # 2. 精确匹配过滤器 (模拟工业级 10000 大词表，微秒级匹配)
    print("[2/3] 正在初始化 HotwordFilter (FlashText)...")
    exact_hotwords = {"极速替换词": "【精确命中】"}
    exact_filter = HotwordFilter(hotwords=exact_hotwords)
    
    # 3. 实体感知模糊过滤器 (SOTA 引擎)
    print("[3/3] 正在初始化 EntityAwareFuzzyFilter (Numba + Jieba)...")
    # 工业界防呆技巧：对于 Jieba 识别不出的特殊人名错别字，直接强行喂入词典
    jieba.add_word('也开', tag='nr')
    
    fuzzy_rules = [
        "张三:0.7|李四:0.7",             # 极简写法：目标词是“张三”，默认任何发音相似的词（如“展伞”）都会被替换为“张三”
        "[nr]叶开:0.8",         # 局部约束：必须被 Jieba 识别为人名(nr)才触发纠错
        # "李四:0.7"              # 极简写法：任何像“里死”的发音都会变成“李四”
    ]
    entity_filter = EntityAwareFuzzyFilter(rules=fuzzy_rules)
    
    # ---------------------------------------------------------
    # 第二步：组装 Pipeline 引擎
    # ---------------------------------------------------------
    print("\n组装 Pipeline 中...")
    pipeline = Pipeline()
    pipeline.add_filter(regex_filter)  # 最先清洗
    pipeline.add_filter(exact_filter)  # 其次精确拦截
    pipeline.add_filter(entity_filter) # 最后模糊兜底
    
    # ---------------------------------------------------------
    # 第三步：测试各种边界 Case
    # ---------------------------------------------------------
    test_cases = [
        "那个 昨天通知了一下展伞和里死。",                     # Case 1: 包含正则口水词 + 全局音素纠错
        "但是他也开心了，最后通知了也开。",                     # Case 2: 验证防误杀 ("也开心"不能被纠正)，而单独人名 "也开" 应该被纠正(假设被错误识别)
        "如果出现 极速替换词 ，那么瞬间就会被改掉。",            # Case 3: 包含空格 + 精确大词表替换
        "i try to test"
    ]
    
    print("\n" + "="*50)
    print("开始处理测试句子：")
    print("="*50)
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n[Case {i}]")
        print(f"输入文本: {text}")
        
        t0 = time.perf_counter()
        result = pipeline.process(text)
        t1 = time.perf_counter()
        
        print(f"输出结果: {result}")
        print(f"处理耗时: {(t1-t0)*1000:.2f} ms")

def get_phoneme_info(text):
    import jieba.posseg as pseg
    words = pseg.cut(text)
    for w in words:
        print(f"词语: {w.word} \t 词性: {w.flag}")

if __name__ == "__main__":
    run_demo()
    get_phoneme_info('最后通知了夜凯。')
