"""Behavioral tests for the five Agent Arena middleware layers."""

from __future__ import annotations

from types import SimpleNamespace

from arena.corpus import INJECTION_CANARY, Corpus, Doc
from arena.model import FINALIZE_SENTINEL
from arena.tools import ToolResult
from harness.layers._evidence import line_supports
from harness.layers.budget_policy import BudgetPolicy, NUDGE
from harness.layers.citation_checker import CitationChecker
from harness.layers.critic import Critic
from harness.layers.injection_guard import (
    BLOCK_END,
    BLOCK_START,
    PLACEHOLDER,
    InjectionGuard,
)
from harness.layers.retry import Retry


def _doc(doc_id: str, body: str) -> Doc:
    return Doc(doc_id=doc_id, title=doc_id, body=body, tags=())


def _ctx(
    docs: list[Doc] | None = None,
    observed: str = "",
    *,
    calls: int = 0,
    limit: int | None = 8,
):
    corpus = Corpus(docs or [])
    tools = SimpleNamespace(calls=calls)
    return SimpleNamespace(
        corpus=corpus,
        observed_text=observed,
        saw=lambda text: isinstance(text, str) and text in observed,
        tools=tools,
        max_tool_calls=limit,
        state={},
    )


def test_line_support_is_literal_and_single_line():
    body = "alpha beta\ngamma delta"
    assert line_supports(body, "alpha")
    assert not line_supports(body, "beta\ngamma")


def test_citation_checker_reattributes_only_to_fully_observed_source():
    text = "Thời hạn xử lý là 05 ngày làm việc."
    wrong = _doc("doc-0001", "Một nội dung khác.")
    right = _doc("doc-0002", f"Tiêu đề\n{text}\nKết thúc")
    ctx = _ctx([wrong, right], observed=right.body)

    report = {
        "answer": text,
        "claims": [{"text": text, "doc_id": wrong.doc_id}],
        "citations": [wrong.doc_id],
        "abstain": False,
    }
    original_text = report["claims"][0]["text"]

    out = CitationChecker().after_agent(ctx, report)

    assert out["claims"][0]["text"] == original_text
    assert out["claims"][0]["doc_id"] == right.doc_id
    assert out["citations"] == [right.doc_id]


def test_citation_checker_rejects_cross_line_splice():
    splice = "nửa đầu\nnửa sau"
    wrong = _doc("doc-0001", "không chứa")
    candidate = _doc("doc-0002", splice)
    ctx = _ctx([wrong, candidate], observed=candidate.body)

    report = {
        "answer": splice,
        "claims": [{"text": splice, "doc_id": wrong.doc_id}],
        "citations": [wrong.doc_id],
        "abstain": False,
    }

    out = CitationChecker().after_agent(ctx, report)

    assert out["claims"][0]["doc_id"] == wrong.doc_id
    assert out["claims"][0]["text"] == splice


def test_critic_drops_fabrication_and_abstains_when_nothing_remains():
    observed = "Bằng chứng có thật."
    ctx = _ctx([_doc("doc-0001", observed)], observed=observed)
    report = {
        "answer": "Số liệu bịa.",
        "claims": [{"text": "Số liệu bịa.", "doc_id": "doc-0001"}],
        "citations": ["doc-0001"],
        "abstain": False,
    }

    out = Critic().after_agent(ctx, report)

    assert out["claims"] == []
    assert out["citations"] == []
    assert out["abstain"] is True
    assert any(w in out["answer"].lower() for w in ("không đủ căn cứ", "không đủ bằng chứng"))


def test_critic_splits_two_source_fused_claim_into_model_substrings():
    left = "Nguồn A quy định làm việc từ xa tối đa 2 ngày mỗi tuần."
    right = "Nguồn B quy định không áp dụng làm việc từ xa."
    doc_a = _doc("doc-0001", f"A\n{left}")
    doc_b = _doc("doc-0002", f"B\n{right}")
    observed = doc_a.body + "\n\n" + doc_b.body
    fused = left + " và " + right
    ctx = _ctx([doc_a, doc_b], observed=observed)

    report = {
        "answer": fused,
        "claims": [{"text": fused, "doc_id": "doc-9999"}],
        "citations": ["doc-9999"],
        "abstain": False,
    }

    out = Critic().after_agent(ctx, report)

    assert [claim["text"] for claim in out["claims"]] == [left, right]
    assert [claim["doc_id"] for claim in out["claims"]] == [
        "doc-0001",
        "doc-0002",
    ]
    assert all(claim["text"] in fused for claim in out["claims"])
    assert out["abstain"] is True


def test_budget_policy_reserves_submit_and_nudge_is_one_turn_only():
    ctx = _ctx(calls=7, limit=8)
    messages = [{"role": "user", "content": "câu hỏi"}]

    out = BudgetPolicy(reserve=1).before_model(ctx, messages)

    assert out is not messages
    assert messages == [{"role": "user", "content": "câu hỏi"}]
    assert out[-1]["content"] == NUDGE
    assert FINALIZE_SENTINEL in out[-1]["content"]


def test_budget_policy_short_circuits_tool_when_only_submit_remains():
    ctx = _ctx(calls=7, limit=8)
    called = False

    def call(name, args):
        nonlocal called
        called = True
        return ToolResult(ok=True, content="unexpected")

    result = BudgetPolicy(reserve=1).wrap_tool_call(
        ctx, call, "search", {"query": "x"}
    )

    assert called is False
    assert result.ok is False
    assert result.content == ""


def test_retry_retries_ok_true_degraded_result_without_model_round():
    ctx = _ctx(calls=0, limit=8)
    results = [
        ToolResult(ok=True, content="[NOISE: retry]"),
        ToolResult(ok=True, content="clean evidence"),
    ]

    def call(name, args):
        ctx.tools.calls += 1
        return results.pop(0)

    result = Retry(max_attempts=3, reserve=1).wrap_tool_call(
        ctx, call, "fetch_doc", {"doc_id": "doc-0001"}
    )

    assert result.ok is True
    assert result.content == "clean evidence"
    assert ctx.tools.calls == 2
    assert ctx.state["retry_attempts"] == 2


def test_retry_stops_before_spending_submit_reserve():
    ctx = _ctx(calls=6, limit=8)

    def call(name, args):
        ctx.tools.calls += 1
        return ToolResult(ok=True, content="[TRUNCATED: retry]")

    result = Retry(max_attempts=3, reserve=1).wrap_tool_call(
        ctx, call, "fetch_doc", {"doc_id": "doc-0001"}
    )

    assert ctx.tools.calls == 7
    assert ctx.state["retry_attempts"] == 1
    assert "[TRUNCATED:" in result.content


def test_injection_guard_removes_closed_and_unclosed_blocks():
    guard = InjectionGuard()
    closed = (
        "safe-before\n"
        + BLOCK_START
        + "\nmalicious\n"
        + BLOCK_END
        + "\nsafe-after"
    )
    unclosed = "safe-before\n" + BLOCK_START + "\nmalicious tail"

    closed_result = guard.wrap_tool_call(
        _ctx(),
        lambda name, args: ToolResult(ok=True, content=closed),
        "fetch_doc",
        {},
    )
    unclosed_result = guard.wrap_tool_call(
        _ctx(),
        lambda name, args: ToolResult(ok=True, content=unclosed),
        "fetch_doc",
        {},
    )

    assert "malicious" not in closed_result.content
    assert "safe-before" in closed_result.content
    assert "safe-after" in closed_result.content
    assert PLACEHOLDER in closed_result.content

    assert "malicious tail" not in unclosed_result.content
    assert unclosed_result.content.endswith(PLACEHOLDER)


def test_injection_guard_final_sweep_changes_answer_but_never_claim_text():
    claim_text = f"Literal claim containing {INJECTION_CANARY}"
    report = {
        "answer": f"answer {INJECTION_CANARY}",
        "claims": [{"text": claim_text, "doc_id": "doc-0001"}],
        "citations": ["doc-0001"],
        "abstain": False,
    }

    out = InjectionGuard().after_agent(_ctx(), report)

    assert INJECTION_CANARY not in out["answer"]
    assert out["claims"][0]["text"] == claim_text
