import time
import logging
import sys
import os
import jieba
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from text_rewrite.pipeline import Pipeline
from text_rewrite.filters.entity_fuzzy import EntityAwareFuzzyFilter

logging.basicConfig(level=logging.WARNING)

def run_benchmark(num_exact=9000, num_fuzzy=500):
    print(f"\n=== SOTA EntityAwareFuzzyFilter 性能压测 (精确词:{num_exact}, 模糊词:{num_fuzzy}) ===")
    
    import random
    random.seed(43)
    chars = '的一是不了在人有我他这个们中来上大为和国地到以说时要就出会可也你对生能而子那得于着下自之年过发后作里如等'
    
    exact_hotwords = {}
    for i in range(num_exact):
        w = ''.join(random.choice(chars) for _ in range(3))
        exact_hotwords[w] = w
        
    fuzzy_rules = []
    for i in range(num_fuzzy):
        w = ''.join(random.choice(chars) for _ in range(2))
        fuzzy_rules.append(f"{w}:0.7")
        
    # 加入我们的目标测试用例
    fuzzy_rules.append("张三:0.7")
    fuzzy_rules.append("[nr]叶开:0.8")
    fuzzy_rules.append("李四:0.7")
    
    print("正在初始化规则引擎...")
    jieba.add_word('也开', tag='nr')
    from text_rewrite.filters.hotword import HotwordFilter
    t0 = time.perf_counter()
    exact_filter = HotwordFilter(hotwords=exact_hotwords)
    entity_filter = EntityAwareFuzzyFilter(rules=fuzzy_rules)
    
    pipeline = Pipeline()
    if num_exact > 0:
        pipeline.add_filter(exact_filter)
    pipeline.add_filter(entity_filter)
    t1 = time.perf_counter()
    print(f"-> 解析并构造 {num_exact + num_fuzzy} 条混合规则引擎耗时: {(t1-t0)*1000:.2f} ms")
    
    text = "昨天通知了一下展伞和里死，但是他也开心了，最后通知了也开。"
    
    res = pipeline.process(text)
    print(f"原始短文本: {text}")
    print(f"处理结果: {res}")
    
    # 短文本吞吐量压测
    iterations = 50
    start = time.perf_counter()
    for _ in range(iterations):
        pipeline.process(text)
    end = time.perf_counter()
    print(f"-> 短文本(30字) 端到端平均处理耗时: {(end - start) * 1000 / iterations:.2f} ms")
    
    # 生成一篇 500 字的长文本
    long_text = ""
    for _ in range(50):
        long_text += "昨天通知了一下展伞和里死，但是他也开心了，最后通知了也开。"
        long_text += "".join(random.choice(chars) for _ in range(10))
        
    print(f"长文本长度: {len(long_text)} 字")
    
    # res_long = pipeline.process(long_text) # warmup
    
    iterations_long = 20
    start_long = time.perf_counter()
    for _ in range(iterations_long):
        pipeline.process(long_text)
    end_long = time.perf_counter()
    
    avg_time_long_ms = (end_long - start_long) * 1000 / iterations_long
    print(f"-> 千字长文本端到端平均处理耗时: {avg_time_long_ms:.2f} ms")

if __name__ == "__main__":
    run_benchmark(num_exact=0, num_fuzzy=500)
    run_benchmark(num_exact=0, num_fuzzy=500)
    run_benchmark(num_exact=0, num_fuzzy=3000)
    run_benchmark(num_exact=0, num_fuzzy=3000)
    run_benchmark(num_exact=0, num_fuzzy=6000)
    run_benchmark(num_exact=0, num_fuzzy=6000)
    run_benchmark(num_exact=0, num_fuzzy=10000)
    run_benchmark(num_exact=1000, num_fuzzy=6000)
    run_benchmark(num_exact=6000, num_fuzzy=6000)

