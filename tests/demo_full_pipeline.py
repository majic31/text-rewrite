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
        "张三:0.7|李四:0.6",             # 极简写法：目标词是“张三”，默认任何发音相似的词（如“展伞”）都会被替换为“张三”
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


def demo2():
    from text_rewrite.pipeline import Pipeline
    from text_rewrite.filters.regex import RegexFilter
    from text_rewrite.filters.hotword import HotwordFilter
    from text_rewrite.filters.entity_fuzzy import EntityAwareFuzzyFilter

    # 1. 配置正则过滤器（清洗语气词）
    regex_rules = {
        r"\b(嗯|啊|哦|那个)\b": ""
    }
    regex_filter = RegexFilter(rules=regex_rules)

    # 2. 配置精确热词过滤器（处理 10万级 安全大词表）
    exact_filter = HotwordFilter(hotwords={"确定性长词": "替换词"})

    # 3. 配置实体感知模糊过滤器 (EntityAwareFuzzyFilter)
    # 语法: [标签]原词:权重 | 原词2:替换词:权重
    fuzzy_rules = [
        "张三:0.9",             # 无标签：全局极简配置（较高权重，发音类似张三的词，如展伞，都会被纠正为张三）
        "[nr]叶开:1.0",         # 有标签：严格约束（必须是人名，且发音相似才纠正，适中权重）
        "李四:0.7"              # 全局极简配置（里死 -> 李四，常规权重）
    ]

    f_rules_notag = ['叶开:0.75']    # 这里做了个没有词性的干预

    # 这里显示了开启hmm模式下的用法，但真实环境中，即使开启hmm后，也很难保证词性一定被识别出来，性能反而会有5-6倍的损耗，所以不太建议开启（默认也是关闭的）
    # 不开启hmm的情况下，热词还是尽量选择3个字以上的，并且阈值尽量调成0.6-0.8的样子.
    # 如果热词都是短词（2个字的），则建议开启hmm，并将asr误识别的词加入到jieba词典，避免误杀。
    entity_filter_hmm = EntityAwareFuzzyFilter(rules=fuzzy_rules, use_hmm=True)
    entity_filter_default = EntityAwareFuzzyFilter(rules=fuzzy_rules, use_hmm=False)
    entity_filter_no_tag = EntityAwareFuzzyFilter(rules=f_rules_notag)
    # 4. 组装 Pipeline 引擎（顺序即执行顺序）
    p_hmm = Pipeline()
    p_hmm.add_filter(regex_filter)
    p_hmm.add_filter(exact_filter)
    p_hmm.add_filter(entity_filter_hmm)

    p_default = Pipeline()
    p_default.add_filter(entity_filter_default)

    p_notag = Pipeline()
    p_notag.add_filter(entity_filter_no_tag)

    # 5. 真实流式调用
    text = "那个，昨天通知了一下展伞和里死，但是他也开心了，最后通知了夜凯。"
    result_hmm = p_hmm.process(text)
    result_default = p_default.process(text)
    result_notag = p_notag.process(text)
    # 开了hmm后，词性是准确的，“也开”不会被替换，结果为：昨天通知了一下张三和李四，但是他也开心了，最后通知了叶开。
    print(f'hmm result: {result_hmm}') 
    # 没有开hmm后，未能猜出“夜凯”的人名词性，所以【叶开】没有被替换；但无标签的【张三】、【李四】依然被成功替换。结果为：那个，昨天通知了一下张三和李四，但是他也开心了，最后通知了夜凯。
    print(f'default result: {result_default}')
    # 没有词性后，可能会误替换。结果为：那个，昨天通知了一下展伞和里死，但是他叶开心了，最后通知了叶开。
    print(f'notag result: {result_notag}')

def demo_english():
    print("\n=== English Fuzzy Correction Demo ===")
    from text_rewrite.pipeline import Pipeline
    from text_rewrite.filters.entity_fuzzy import EntityAwareFuzzyFilter
    
    # 英文模糊纠错：天然支持拼写错误、大小写无关、以及外语无空格粘连纠错
    # 对于带有特定业务外语的 ASR，可以通过添加 [eng] 词性来限定
    fuzzy_rules = [
        "Claude:1.0",         # 正常权重，默认 threshold 0.6
        "[eng]Apple:0.8",      # 仅在 Jieba 识别出 eng 词性时触发纠错
        "iphone 15:1.0"
    ]
    
    entity_filter = EntityAwareFuzzyFilter(rules=fuzzy_rules, use_hmm=False)
    p = Pipeline()
    p.add_filter(entity_filter)
    
    test_cases = [
        "I use cloude everyday",      # 拼写错误 (o -> a)，大小写脱敏
        "this is my aple watch",       # 无空格外语连读
        "this is my i phone 15 watch",       # iphone
        "claode is great",            # 中间拼写错误
        "klaude is great",            # 中间拼写错误
        "klaode is great",            # 中间拼写错误
    ]
    
    for text in test_cases:
        result = p.process(text)
        print(f"输入: {text:<25} -> 输出: {result}")

def get_phoneme_info(text):
    # pyrefly: ignore [missing-import]
    import jieba.posseg as pseg
    words = pseg.cut(text, HMM=False)
    for w in words:
        print(f"词语: {w.word} \t 词性: {w.flag}")


if __name__ == "__main__":
    # run_demo()
    demo2()
    demo_english()
    # get_phoneme_info('那个，昨天通知了一下展伞和里死，但是他也开心了，最后通知了夜凯。')
