from __future__ import annotations

import json

from reef_client.sse import SSEAccumulator, synthesize_sse_events


def frame(chunk: dict[str, object]) -> bytes:
    return f"data: {json.dumps(chunk)}\n\n".encode()


def chunk(delta: dict[str, object], **choice: object) -> dict[str, object]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [{"index": 0, "delta": delta, "finish_reason": None, **choice}],
    }


def accumulated(*chunks: dict[str, object]) -> dict[str, object]:
    accumulator = SSEAccumulator()
    for item in chunks:
        accumulator.feed(frame(item))
    accumulator.feed(b"data: [DONE]\n\n")
    accumulator.finish()
    assert accumulator.completion is not None
    return accumulator.completion


def test_the_accumulator_adds_up_a_text_field_it_has_no_rule_for() -> None:
    completion = accumulated(
        chunk({"role": "assistant", "reasoning": "think ", "content": None}),
        chunk({"reasoning": "first", "content": ""}),
        chunk({"reasoning": None, "content": "answer"}),
    )

    assert completion["choices"][0]["message"] == {"role": "assistant", "content": "answer", "reasoning": "think first"}


def test_the_accumulator_keeps_the_last_value_of_another_field() -> None:
    completion = accumulated(
        chunk({"role": "assistant", "provider_specific_fields": {"a": 1}}),
        chunk({"content": "x", "provider_specific_fields": {"a": 2}}),
    )

    assert completion["choices"][0]["message"]["provider_specific_fields"] == {"a": 2}


def test_the_accumulator_adds_up_log_probs_and_keeps_chunk_and_choice_fields() -> None:
    first = {**chunk({"role": "assistant", "content": "a"}, logprobs={"content": [{"token": "a"}]}), "provider": "p1"}
    second = {**chunk({"content": "b"}, logprobs={"content": [{"token": "b"}]}, stop_reason=7), "provider": "p2"}

    completion = accumulated(first, second)

    choice = completion["choices"][0]
    assert choice["logprobs"] == {"content": [{"token": "a"}, {"token": "b"}]}
    assert choice["stop_reason"] == 7
    assert completion["provider"] == "p2"


def test_a_synthesized_stream_and_the_accumulator_keep_every_message_field() -> None:
    message = {
        "role": "assistant",
        "content": "next",
        "reasoning": "think",
        "reasoning_content": "think too",
        "provider_specific_fields": {"refusal": None},
        "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}],
    }
    payload = {
        "id": "chatcmpl-1",
        "created": 1,
        "model": "m",
        "choices": [{"index": 0, "message": {**message, "refusal": None}, "finish_reason": "tool_calls"}],
    }

    accumulator = SSEAccumulator()
    accumulator.feed("".join(synthesize_sse_events(payload)).encode())
    accumulator.finish()

    assert accumulator.completion is not None
    assert accumulator.completion["choices"][0]["message"] == message
