import time
import logging
import sys
import os
import random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from text_rewrite.pipeline import Pipeline
from text_rewrite.filters.entity_fuzzy import EntityAwareFuzzyFilter

logging.basicConfig(level=logging.WARNING)

def generate_rules(prefix: str, count: int) -> list:
    """生成模拟的随机热词规则"""
    chars = '的一是不了在人有我他这个们中来上大为和国地到以说时要就出会可也你对生能而子那得于着下自之年过发后作里如等'
    rules = []
    for _ in range(count):
        w = ''.join(random.choice(chars) for _ in range(2))
        rules.append(f"{w}:0.7")
    return rules

def main():
    print("=== 全局静态热词 + 租户动态热词 架构演示 ===")
    
    # ---------------------------------------------------------
    # 1. 模拟服务器启动阶段 (只需执行一次)
    # ---------------------------------------------------------
    print("\n[服务器启动] 加载包含 10000 条规则的全局静态词库...")
    t0 = time.perf_counter()
    static_rules = generate_rules("static", 10000)
    # 故意加一条冲突规则，测试动态词覆盖静态词的能力
    static_rules.append("张三:李四:0.7") 
    
    global_static_filter = EntityAwareFuzzyFilter(rules=static_rules)
    t1 = time.perf_counter()
    print(f" -> 静态词库加载完成，耗时: {(t1 - t0) * 1000:.2f} ms")


    # ---------------------------------------------------------
    # 2. 模拟运行时的请求级别缓存
    # ---------------------------------------------------------
    tenant_pipelines = {}

    def get_pipeline_for_tenant(tenant_id: str, dynamic_rules: list) -> Pipeline:
        if tenant_id not in tenant_pipelines:
            print(f"\n[冷启动] 正在为租户 {tenant_id} 编译动态热词引擎 (500词)...")
            t_start = time.perf_counter()
            
            # 由于 get_phoneme_seq 加了 LRU Cache，相同动态词的第二次编译几乎是 0 毫秒！
            dynamic_filter = EntityAwareFuzzyFilter(rules=dynamic_rules)
            
            # 组装 Pipeline (责任链模式)
            # 【重要】动态词排在前面，优先拦截并执行替换！
            pipeline = Pipeline()
            pipeline.add_filter(dynamic_filter)
            pipeline.add_filter(global_static_filter)
            
            tenant_pipelines[tenant_id] = pipeline
            t_end = time.perf_counter()
            print(f" -> 动态引擎编译及 Pipeline 组装完成，耗时: {(t_end - t_start) * 1000:.2f} ms")
            
        return tenant_pipelines[tenant_id]


    # ---------------------------------------------------------
    # 3. 模拟实际业务请求
    # ---------------------------------------------------------
    print("\n[模拟请求] 租户 A 发起请求...")
    tenant_a_dynamic_rules = generate_rules("dynamic_a", 500)
    # 动态热词优先：把张三替换成王五，这会截胡静态词库的张三->李四规则
    tenant_a_dynamic_rules.append("张三:王五:0.7") 

    # 第一次请求：会触发冷启动编译
    text = "昨天通知了一下章三，让他明天过来。"
    print(f"\n原文本: {text}")
    
    pipeline_a = get_pipeline_for_tenant("tenant_a", tenant_a_dynamic_rules)
    
    t0 = time.perf_counter()
    res1 = pipeline_a.process(text)
    t1 = time.perf_counter()
    print(f"租户 A 处理结果: {res1} (处理耗时: {(t1 - t0) * 1000:.2f} ms)")
    print(" -> (注: '章三' 被动态热词规则替换为了 '王五'，而不是静态规则的 '李四')")

    # ---------------------------------------------------------
    # 第二次请求：命中内存级别缓存，零延迟
    print("\n[模拟请求] 租户 A 再次发起请求...")
    t0 = time.perf_counter()
    pipeline_a_cached = get_pipeline_for_tenant("tenant_a", tenant_a_dynamic_rules)
    res2 = pipeline_a_cached.process("刚才遇到张伞了。")
    t1 = time.perf_counter()
    print(f"租户 A 缓存处理结果: {res2} (全链路耗时: {(t1 - t0) * 1000:.2f} ms)")


if __name__ == "__main__":
    main()
