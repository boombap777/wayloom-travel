"""The one prompt format shared by SFT construction and base/adapter evaluation."""

from __future__ import annotations

from typing import Any


def build_messages(system_prompt: str, today: str, user_input: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": f"当前日期：{today}\n用户请求：{user_input}"},
    ]


def render_generation_prompt(tokenizer: Any, system_prompt: str, today: str, user_input: str) -> str:
    return tokenizer.apply_chat_template(
        build_messages(system_prompt, today, user_input),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

