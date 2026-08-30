"""Grounded-answer flag: prompt hardening switch for the generation module."""

from __future__ import annotations

from rag_modules.domain_config import get_domain
from rag_modules.generation_integration import (
    GROUNDING_CLAUSE,
    GenerationIntegrationModule,
    apply_grounding,
)


def test_apply_grounding_appends_clause_only_when_enabled() -> None:
    base = "你是一位专业的烹饪助手。用户问题: {question}\n相关食谱信息:\n{context}"

    assert apply_grounding(base, enabled=False) == base
    grounded = apply_grounding(base, enabled=True)
    assert grounded.startswith(base)
    assert "忠实性硬约束" in grounded
    assert "{question}" in grounded and "{context}" in grounded  # 占位符不受影响


def test_grounded_clause_covers_the_measured_failure_modes() -> None:
    # RAGAS 归因出的三类编造：时长估算、尺寸/用量估算、评价性描述
    assert "时长" in GROUNDING_CLAUSE and "毫升" in GROUNDING_CLAUSE
    assert "口感" in GROUNDING_CLAUSE


def test_generation_module_wires_grounded_flag_into_answer_prompts() -> None:
    module = GenerationIntegrationModule(grounded_answer=True, llm=object())
    grounded = apply_grounding(get_domain().step_by_step_prompt, module.grounded_answer)

    # 原模板要求"大概所需时间"（语料缺失时的编造源头），硬约束必须覆盖它
    assert "大概所需时间" in grounded
    assert "严禁自行估算" in grounded
