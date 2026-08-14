"""LỚP `injection_guard` — bài giảng Day 16, §9 (Defensive Agent Architecture).

NHIỆM VỤ: trong kho có một tài liệu độc hại chèn chỉ thị giả dạng dữ liệu.
Nhiệm vụ của bạn là cách ly khối chỉ thị độc hại này tại ranh giới công cụ
và làm sạch canary ở report answer cuối cùng.
"""

from __future__ import annotations

from arena.corpus import INJECTION_CANARY
from arena.tools import ToolResult
from harness.middleware import Middleware

BLOCK_START = "--- LƯU Ý HỆ THỐNG"
BLOCK_END = "--- HẾT LƯU Ý HỆ THỐNG ---"
PLACEHOLDER = "[nội dung không đáng tin cậy đã bị loại bỏ]"


def _strip_untrusted_blocks(content: str) -> str:
    """Loại bỏ toàn bộ khối lệnh độc hại, kể cả khối bị cắt cụt không có thẻ đóng."""
    while BLOCK_START in content:
        start = content.find(BLOCK_START)
        end = content.find(BLOCK_END, start + len(BLOCK_START))

        if end < 0:
            return content[:start] + PLACEHOLDER

        content = (
            content[:start]
            + PLACEHOLDER
            + content[end + len(BLOCK_END) :]
        )

    return content


class InjectionGuard(Middleware):
    """Bảo vệ agent trước prompt injection: coi dữ liệu từ tool là data, không phải instruction."""

    name = "injection_guard"

    def wrap_tool_call(self, ctx, call, name, args):
        result = call(name, args)
        if BLOCK_START not in result.content:
            return result

        return ToolResult(
            ok=result.ok,
            content=_strip_untrusted_blocks(result.content),
            error=result.error,
        )

    def after_agent(self, ctx, report):
        answer = report.get("answer")
        if isinstance(answer, str) and INJECTION_CANARY in answer:
            report["answer"] = answer.replace(INJECTION_CANARY, "").strip()
        return report
