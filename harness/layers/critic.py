"""LỚP `critic` — bài giảng Day 16, §2 (Reflection & Self-Critique).

NHIỆM VỤ: mô hình KHÔNG BAO GIỜ nói "tôi không biết". Nhiệm vụ của lớp này là
loại bỏ các claim bịa đặt (hallucination), tách câu ghép mâu thuẫn 2 nguồn an toàn
theo ranh giới quan sát được, và kích hoạt abstain khi không đủ căn cứ.
"""

from __future__ import annotations

from harness.layers._evidence import citations_from_claims, observed_source_ids
from harness.middleware import Middleware

_FUSE_SEPARATOR = " và "
_ABSTAIN_ANSWER = "Không đủ căn cứ trong các tài liệu đã quan sát để kết luận."


class Critic(Middleware):
    """Xoá những gì bằng chứng không đỡ; abstain khi không còn gì."""

    name = "critic"

    def _split_fused(self, ctx, claim: dict) -> list[dict]:
        text = claim.get("text")
        if not isinstance(text, str) or _FUSE_SEPARATOR not in text:
            return []

        start = 0
        while True:
            cut = text.find(_FUSE_SEPARATOR, start)
            if cut < 0:
                return []

            left = text[:cut].strip()
            right = text[cut + len(_FUSE_SEPARATOR) :].strip()

            if left and right and ctx.saw(left) and ctx.saw(right):
                left_ids = observed_source_ids(ctx, left)
                right_ids = observed_source_ids(ctx, right)

                for left_id in left_ids:
                    for right_id in right_ids:
                        if left_id == right_id:
                            continue

                        left_claim = dict(claim)
                        right_claim = dict(claim)
                        left_claim.update(text=left, doc_id=left_id)
                        right_claim.update(text=right, doc_id=right_id)
                        return [left_claim, right_claim]

            start = cut + len(_FUSE_SEPARATOR)

    def after_agent(self, ctx, report):
        claims = report.get("claims")
        if not isinstance(claims, list):
            return report

        kept: list[dict] = []
        contradiction_seen = False

        for claim in claims:
            if not isinstance(claim, dict):
                continue

            text = claim.get("text")
            if isinstance(text, str) and text and ctx.saw(text):
                # Giữ nguyên chữ của model byte-for-byte
                kept.append(claim)
                continue

            split = self._split_fused(ctx, claim)
            if split:
                kept.extend(split)
                contradiction_seen = True
            # Không có trong quan sát và không tách được -> bịa: bỏ claim đi

        report["claims"] = kept
        report["citations"] = citations_from_claims(kept)

        if contradiction_seen:
            report["abstain"] = True

        if not kept:
            report["abstain"] = True
            report["citations"] = []
            report["answer"] = _ABSTAIN_ANSWER

        return report
