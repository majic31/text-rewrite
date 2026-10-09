import sys
import time
import random

sys.path.append("/Users/majie/project/asr_gateway")
sys.path.append("/Users/majie/project/text-rewrite")

try:
    from utils.hotword.hot_phoneme import PhonemeCorrector
except ImportError as e:
    print(f"错误: 无法导入旧版 asr_gateway 引擎: {e}")
    sys.exit(1)

try:
    from text_rewrite.pipeline import Pipeline
    from text_rewrite.filters.entity_fuzzy import EntityAwareFuzzyFilter
except ImportError as e:
    print(f"错误: 无法导入新版 text-rewrite 引擎: {e}")
    sys.exit(1)


def generate_hotwords(num=1000):
    words = []
    chars = "天地玄黄宇宙洪荒日月盈昃辰宿列张金木水火土"
    for _ in range(num):
        words.append("".join(random.choice(chars) for _ in range(random.randint(2, 4))))
    return words


def run_benchmark():
    num_hotwords = 1000
    print(f"生成 {num_hotwords} 个随机模糊热词用于测试...")
    hotwords = generate_hotwords(num_hotwords)
    
    print("正在初始化旧版 PhonemeCorrector...")
    old_corrector = PhonemeCorrector(threshold=0.7)
    hotword_text = "\n".join(hotwords + ["张三", "李四", "叶开"])
    old_corrector.update_hotwords(hotword_text)
    
    print("正在初始化新版 text-rewrite Pipeline...")
    pipeline = Pipeline()
    fuzzy_rules = [f"{hw}:0.7" for hw in hotwords]
    fuzzy_rules.extend(["张三:0.7", "李四:0.7", "[nr]叶开:0.8"])
    entity_filter = EntityAwareFuzzyFilter(rules=fuzzy_rules)
    pipeline.add_filter(entity_filter)
    
    short_text = "昨天通知了展伞和里死，最后通知了夜凯。"
    print("预热 Numba JIT (请稍候)...")
    old_corrector.correct(short_text)
    pipeline.process(short_text)
    
    long_text = short_text * 50  
    iterations = 50
    
    print("开始性能测试...")
    # 旧版短文
    t0 = time.perf_counter()
    for _ in range(iterations):
        old_corrector.correct(short_text)
    old_short_ms = (time.perf_counter() - t0) * 1000 / iterations
    
    # 旧版长文
    t0 = time.perf_counter()
    for _ in range(iterations):
        old_corrector.correct(long_text)
    old_long_ms = (time.perf_counter() - t0) * 1000 / iterations
    
    # 新版短文
    t0 = time.perf_counter()
    for _ in range(iterations):
        pipeline.process(short_text)
    new_short_ms = (time.perf_counter() - t0) * 1000 / iterations
    
    # 新版长文
    t0 = time.perf_counter()
    for _ in range(iterations):
        pipeline.process(long_text)
    new_long_ms = (time.perf_counter() - t0) * 1000 / iterations
    
    print("\n" + "="*60)
    print(f"基准测试: 词库容量 {num_hotwords}，迭代 {iterations} 次取平均")
    print("="*60)
    print(f"【旧版引擎】 asr_gateway - PhonemeCorrector")
    print(f"  短文本 (~20字) : {old_short_ms:8.2f} ms / 句")
    print(f"  长文本 (~1000字): {old_long_ms:8.2f} ms / 句")
    print("-"*60)
    print(f"【新版引擎】 text-rewrite - EntityAwareFuzzyFilter")
    print(f"  短文本 (~20字) : {new_short_ms:8.2f} ms / 句")
    print(f"  长文本 (~1000字): {new_long_ms:8.2f} ms / 句")
    print("="*60)
    
    print(f"\n🚀 性能飞跃 (旧版耗时 / 新版耗时):")
    print(f"  短文本加速: {old_short_ms / new_short_ms:.1f} 倍")
    print(f"  长文本加速: {old_long_ms / new_long_ms:.1f} 倍\n")

if __name__ == "__main__":
    run_benchmark()
