# tests/test_basic.py
def test_import():
    from text_rewrite.pipeline import Pipeline
    from text_rewrite.filters.entity_fuzzy import EntityAwareFuzzyFilter
    
    # 只要不报错，就说明依赖和环境没问题！
    assert True
