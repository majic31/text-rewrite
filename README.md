# text-rewrite 🚀

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**text-rewrite** 是一个专为工业级 ASR（语音识别）后处理设计的**高性能文本纠错与过滤引擎**。它将极致的计算效率与高精度的语义约束相结合，完美解决了传统 ASR 热词匹配中常见的“跨词误杀”、“谐音错认”以及“高并发性能瓶颈”问题。

## ✨ 核心特性

- **极致性能 ($O(N)$ 线性扩展)**：针对万级甚至十万级热词进行专项优化。处理近 2000 字的长文本只需不足 600 毫秒（常规短句 <10ms），绝不会拖垮流式并发服务器。
- **Entity-Aware 实体约束引擎**：针对容易误杀的极短词（如人名“叶开” vs “也开心”），创新性引入基于 Jieba 词性标注的轻量级 NER 引擎。**内建极速音素预检机制与无 HMM 模式**，将千字长文本的 NER 解析时间从 400ms 暴砍至 5ms。
- **双音节倒排索引 (Bigram Syllable Index)**：创新性地构建了基于“首位双音节”自适应哈希倒排池。将 10000 个热词在 2000 字长文本上的 DP 候选空间极致压缩了 98.8%，打破长文本下 $K=100\%$ 的魔咒。
- **Numba Batch DP 加速**：底层基于 Numba JIT 编译的“批量化动态规划（DP）”核心引擎，彻底消除跨语言调度开销，支持纯发音级别的纠错（完美包容平翘舌、前后鼻音、形近音误差）。
- **微秒级 FlashText 精确匹配**：对于全局安全大词表，底层自动退化为 Aho-Corasick 自动机，做到微秒级无感替换。
- **Pipeline 乐高式组装**：提供高度可扩展的链式过滤器架构，正则清洗、精准替换、模糊纠错一气呵成。

## 📦 安装依赖

该项目核心依赖于高效计算与轻量级 NLP 组件：

```bash
pip install -r requirements.txt
```
> **要求**: `numba>=0.56.0`, `numpy>=1.21.0`, `pypinyin>=0.49.0`, `jieba>=0.42.1`

## 🧩 核心过滤器 (Filters) 概览与最佳实践

在真实的工业级落地场景中，最常用的“黄金三剑客”是以下三个过滤器的组合：

1. **`RegexFilter` (打头阵：规则清洗)**
   - **作用**：干脏活累活。负责前置格式规整。
   - **场景**：将全角符号转半角、去除多余空格、清理语气词（“呃”、“啊”、“那个”），以及执行如 `四S -> 4S` 这种高度规律性的文本格式化。

2. **`HotwordFilter` (中坚力量：业务绝对权威)**
   - **作用**：基于 FlashText 的极速精确替换，零误杀，快、准、狠。
   - **场景**：用于承载几万到几十万量级的“黑白名单 / 品牌库 / 敏感词库”（如“蔚来”、“极氪”）。用它拦截掉绝大部分必须 100% 准确的词，避免增加下游模糊匹配的开销和误杀率。

3. **`EntityAwareFuzzyFilter` (最后兜底：长尾智能纠错)**
   - **作用**：收拾残局。
   - **场景**：经过前面精确词表的拦截，剩下的“长尾错别字”（如用户口音导致的“魏来”、“及克”）由它出马，通过发音的编辑距离计算把错别字捞回来。
   - **语法**：`[标签]原词:权重 | 原词2:替换词:权重`
     - 强行模糊拦截：`张三:1.5` (权重 1.5 极高，即使发音偏差较大如“展伞”也会被强行纠正为“张三”)
     - 严格约束拦截：`李四:0.6` (权重 0.6 极低，只有在拼音和声调100%完美吻合时才允许纠正，防止误杀)
     - 短实体约束：`[nr]叶开:0.8` (`nr`为人名，只有 Jieba 认为是人名时才启动音素级检索，且权重适中)
   - **权重 (Weight) 机制**：底层的发音相似度基础门槛为 `0.6`，实际生效门槛 = `0.6 / 权重`。
     - **数值越大**：越容易被强行修改（容忍口音和 ASR 识别偏差）。
     - **数值越小**：要求发音越精确，设置为 `0.6` 时代表需要 `1.0`（即 100% 完全 match，包括声调）才能触发修改。

> **注**：除上述三者外，内部模块如 `FuzzyPhonemeFilter`（底层的 Numba DP 发音匹配引擎）和 `JiebaNERFilter`（底层 NER 引擎）主要作为组件被 `EntityAwareFuzzyFilter` 自动编排调用，在常规业务中通常无需直接操作。

### 🛠 权重诊断与排错工具 (Diagnostic Tool)
在实际业务落地时，如果你发现某个错别字**没有按预期被纠正**，或者某个正常词**被意外误杀**，可以使用内置的诊断脚本来查看底层的真实打分情况，从而精准调优权重：

```bash
python tests/check_weight_score.py
```

该工具会直接输出底层 Numba 引擎的真实打分逻辑，帮助你快速理解权重的运作方式：
- **场景 A（权重 0.6，极其严格）**：试图把“里死(li3 si3)”纠正为“李四(li3 si4)”。因为声调不同，底层真实得分为 `0.9167`。但权重 `0.6` 对应的及格线是 `1.0000` (要求 100% 完美匹配)。因此 `0.9167 < 1.0000`，引擎判定差距过大，**拒绝修改**。
- **场景 B（权重 0.7，适当放宽）**：同样测试“里死”纠正为“李四”。权重提高到 `0.7` 后，及格线降为 `0.8571`。此时 `0.9167 >= 0.8571`，引擎判定符合容错范围，**成功触发修改**。

你可以随时修改该脚本中的 `analyze_match(target_word="你的词", test_text="测试文本", weight=0.7)` 参数，针对你的业务专有名词进行沙盒测试和精细化调参！

## 🚀 快速开始 (Quick Start)

下面是一个将“正则清洗 -> 精确大词表拦截 -> 高危实体模糊纠错”串联起来的完整 Demo：

```python
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
# 语法: [标签]目标词|阈值 (省略替换词会自动以目标词作为原词和替换词)
fuzzy_rules = [
    "张三|0.7",             # 无标签：全局极简配置（发音类似于张三的词，如展伞，都会被纠正为张三）
    "[nr]叶开|0.8",         # 有标签：严格约束（必须是人名，且发音相似度>=0.8 才纠正）
    "李四|0.7"              # 全局极简配置（里死 -> 李四）
]
entity_filter = EntityAwareFuzzyFilter(rules=fuzzy_rules)

# 4. 组装 Pipeline 引擎（顺序即执行顺序）
pipeline = Pipeline()
pipeline.add_filter(regex_filter)
pipeline.add_filter(exact_filter)
pipeline.add_filter(entity_filter)

# 5. 真实流式调用
text = "那个，昨天通知了一下展伞和里死，但是他也开心了，最后通知了页开。"
result = pipeline.process(text)

print(result) 
# 输出: "，昨天通知了一下张三和李四，但是他也开心了，最后通知了夜凯。"
```

> **注意**：由于初始化 Numba 引擎以及 JIT 预热需要少量时间，建议在服务启动时**单例初始化** `Pipeline` 实例，在后续流式请求中复用该实例调用 `.process(text)`。

## 📊 性能压测 (Benchmark)

您可以运行自带的压测脚本，体验在注入 **10000** 条工业级比例热词配置下的强悍性能：
```bash
python tests/bench_entity_fuzzy.py
```
*   **短句 (约 30 字)**: 端到端平均耗时 **~ 1 ms**
*   **长文 (约 2000 字)**: 端到端平均耗时 **~ 150 ms** (呈现完美的 $O(N)$ 线性扩展，不受十万级词表拖累)

---
*Built with ❤️ for High-Performance NLP Engineering.*
