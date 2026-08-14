"""LỚP `citation_checker` — bài giảng Day 16, §10 (Fact-Checking & Citation Alignment).

NHIỆM VỤ: mô hình hay trích nhầm sang một tài liệu trông có vẻ uy tín (lookalike, outdated).
Nhiệm vụ của bạn là kiểm tra doc_id của từng claim và gán lại chính xác doc_id của tài liệu
thực tế quan sát được có chứa nguyên văn claim text trên một dòng duy nhất, không làm thay
đổi claim text của model.
"""

from __future__ import annotations

from harness.layers._evidence import (
    citations_from_claims,
    find_observed_source,
    line_supports,
)
from harness.middleware import Middleware


class CitationChecker(Middleware):
    """Gán lại doc_id đúng cho claim dựa trên tài liệu quan sát được chứa nguyên văn dòng đó."""

    name = "citation_checker"

    def after_agent(self, ctx, report):
        claims = report.get("claims")
        if not isinstance(claims, list) or not claims or ctx.corpus is None:
            return report

        for claim in claims:
            if not isinstance(claim, dict):
                continue

            text = claim.get("text")
            if not isinstance(text, str) or not text:
                continue

            current = ctx.corpus.get(claim.get("doc_id"))
            if current is not None and line_supports(current.body, text):
                continue

            # Câu không xuất hiện trong quan sát là bịa -> để Critic xử lý
            if not ctx.saw(text):
                continue

            # Chỉ gán lại cho tài liệu mà toàn bộ body đã được quan sát
            source = find_observed_source(ctx, text)
            if source is not None:
                claim["doc_id"] = source

        report["citations"] = citations_from_claims(claims)
        return report
