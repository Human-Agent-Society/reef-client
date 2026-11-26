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


def event_frames(*events: dict[str, object]) -> bytes:
    return b"".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events)


def accumulated_events(*events: dict[str, object]) -> dict[str, object] | None:
    accumulator = SSEAccumulator()
    payload = event_frames(*events)
    # Awkward slices split frames and multi-byte text.
    for offset in range(0, len(payload), 5):
        accumulator.feed(payload[offset : offset + 5])
    accumulator.finish()
    return accumulator.completion


def test_the_accumulator_adds_up_a_choices_token_ids() -> None:
    completion = accumulated(
        {**chunk({"role": "assistant", "content": "a"}, token_ids=[1]), "prompt_token_ids": [7, 8]},
        chunk({"content": "b"}, token_ids=[2, 3]),
    )

    assert completion["choices"][0]["token_ids"] == [1, 2, 3]
    assert completion["prompt_token_ids"] == [7, 8]


def test_a_responses_stream_gives_the_response_of_its_last_event_that_holds_one() -> None:
    final = {"id": "resp_1", "object": "response", "status": "completed", "output": [{"type": "message"}]}

    completion = accumulated_events(
        {"type": "response.created", "response": {"id": "resp_1", "status": "in_progress", "output": []}},
        {"type": "response.output_text.delta", "item_id": "m1", "delta": "hi"},
        {"type": "response.completed", "response": final},
    )

    assert completion == final


def test_a_messages_stream_gives_the_message_that_its_events_build() -> None:
    completion = accumulated_events(
        {
            "type": "message_start",
            "message": {"id": "msg_1", "type": "message", "role": "assistant", "content": [],
                        "stop_reason": None, "usage": {"input_tokens": 9, "output_tokens": 0}},
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "add ✓ "}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "them"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "ping"},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "add", "input": {}}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"a\": "}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "2}"}},
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": 5},
            "token_ids": [4, 5],
        },
        {"type": "message_stop"},
    )

    assert completion == {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "add ✓ them", "signature": "sig"},
            {"type": "tool_use", "id": "toolu_1", "name": "add", "input": {"a": 2}},
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 9, "output_tokens": 5},
        "token_ids": [4, 5],
    }


def test_an_error_event_is_the_completion_of_a_stream_that_gave_nothing_else() -> None:
    error = {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}

    assert accumulated_events(error) == error
