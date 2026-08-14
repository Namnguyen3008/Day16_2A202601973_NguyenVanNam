"""LỚP `retry` — bài giảng Day 16, §6 (Transient Errors & Robustness).

NHIỆM VỤ: các lượt gọi tool có thể bị lỗi, timeout, rác (NOISE) hoặc bị cắt cụt (TRUNCATED).
Nhiệm vụ của bạn là bắt các kết quả suy biến (kể cả khi ok=True) và tự động thử lại
ngay bên dưới tầng model trong phạm vi ngân sách cho phép.
"""

from __future__ import annotations

from arena.model import is_degraded
from harness.middleware import Middleware

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RESERVE = 1


class Retry(Middleware):
    """Thử lại lượt gọi tool giống hệt nhau khi kết quả lỗi hoặc suy biến."""

    name = "retry"

    def __init__(
        self,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        reserve: int = DEFAULT_RESERVE,
    ) -> None:
        self.max_attempts = max(1, int(max_attempts))
        self.reserve = max(0, int(reserve))

    def _budget_allows_retry(self, ctx) -> bool:
        limit = ctx.max_tool_calls
        return limit is None or ctx.tools.calls < limit - self.reserve

    def wrap_tool_call(self, ctx, call, name, args):
        attempts = 1
        result = call(name, args)

        while (
            attempts < self.max_attempts
            and ((not result.ok) or is_degraded(result.content))
            and self._budget_allows_retry(ctx)
        ):
            result = call(name, args)
            attempts += 1

        ctx.state["retry_attempts"] = attempts
        ctx.state["retry_last_attempts"] = attempts
        ctx.state["retry_extra_attempts"] = (
            int(ctx.state.get("retry_extra_attempts", 0)) + attempts - 1
        )
        return result
