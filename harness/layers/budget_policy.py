"""LỚP `budget_policy` — bài giảng Day 16, §3 (Resource Budgets & Early Stopping).

NHIỆM VỤ: mô hình hay gọi tool vượt quá ngân sách brief cho phép.
Nhiệm vụ của bạn là bảo lưu 1 lượt gọi cho lệnh `submit`, bơm tín hiệu `FINALIZE_SENTINEL`
vào tin nhắn nhắc model ra FINAL, và chặn các lượt gọi tool vượt hạn mức.
"""

from __future__ import annotations

from arena.model import FINALIZE_SENTINEL
from arena.tools import ToolResult
from harness.middleware import Middleware

DEFAULT_RESERVE = 1

NUDGE = (
    "Ngân sách công cụ đã hết. Hãy trả lời ngay bằng bằng chứng đang có, "
    f"không gọi thêm công cụ nào nữa. {FINALIZE_SENTINEL}"
)


class BudgetPolicy(Middleware):
    """Bắt buộc ra FINAL và chặn gọi tool khi chỉ còn phần ngân sách dành riêng cho submit."""

    name = "budget_policy"

    def __init__(self, reserve: int = DEFAULT_RESERVE) -> None:
        self.reserve = max(0, int(reserve))

    def _spent(self, ctx) -> bool:
        limit = ctx.max_tool_calls
        return limit is not None and ctx.tools.calls >= limit - self.reserve

    def before_model(self, ctx, messages):
        if not self._spent(ctx):
            return messages

        # Trả về list mới để không làm biến đổi lịch sử tin nhắn gốc
        return messages + [{"role": "user", "content": NUDGE}]

    def wrap_tool_call(self, ctx, call, name, args):
        if not self._spent(ctx):
            return call(name, args)

        # Trả về ToolResult thất bại mà không raise error để agent tiếp tục đến FINAL và submit
        return ToolResult(
            ok=False,
            content="",
            error=(
                "tool budget exhausted; "
                f"reserving {self.reserve} call(s) for submit"
            ),
        )
