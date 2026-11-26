from __future__ import annotations

import json

import pytest

from reef_client.sse import SSEAccumulator, synthesize_sse_events


def completion(message: dict[str, object]) -> dict[str, object]:
    return {
        "id": "chatcmpl-1",
        "created": 1,
        "model": "model",
        "choices": [{"index": 0, "message": {"role": "assistant", **message}, "finish_reason": "tool_calls"}],
    }


def deltas(events: list[str]) -> list[dict[str, object]]:
    chunks = [json.loads(event[len("data: ") :]) for event in events if event != "data: [DONE]\n\n"]
    return [choice["delta"] for chunk in chunks for choice in chunk["choices"]]


CALL = {"id": "call-1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
def test_a_synthesized_stream_carries_the_reasoning_before_the_content(field: str) -> None:
    events = synthesize_sse_events(completion({"content": "next", field: "think first", "tool_calls": [CALL]}))

    sent = deltas(events)

    assert {field: "think first"} in sent
    assert sent.index({field: "think first"}) < sent.index({"content": "next"})


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
def test_the_accumulator_rebuilds_the_reasoning_of_a_stream(field: str) -> None:
    message = {"content": "", field: "think first", "tool_calls": [CALL]}
    accumulator = SSEAccumulator()

    accumulator.feed("".join(synthesize_sse_events(completion(message))).encode())
    accumulator.finish()

    assert accumulator.completion is not None
    assert accumulator.completion["choices"][0]["message"] == {"role": "assistant", **message}


def test_a_message_without_reasoning_gets_no_reasoning_delta() -> None:
    sent = deltas(synthesize_sse_events(completion({"content": "hello", "reasoning": None})))

    assert sent == [{"role": "assistant"}, {"content": "hello"}, {}]
