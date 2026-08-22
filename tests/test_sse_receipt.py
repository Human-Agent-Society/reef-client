from __future__ import annotations

from reef_client.sse import SSEAccumulator, synthesize_sse_events


def test_synthesized_receipt_is_emitted_immediately_before_done() -> None:
    payload = {
        "id": "chatcmpl-1",
        "created": 1,
        "model": "model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }
        ],
    }

    events = synthesize_sse_events(payload, agent_record_id="record-1")

    assert '"reef": {"agent_record_id": "record-1"}' in events[-2]
    assert events[-1] == "data: [DONE]\n\n"


def test_accumulator_reads_terminal_receipt_metadata() -> None:
    events = synthesize_sse_events(
        {
            "id": "chatcmpl-1",
            "created": 1,
            "model": "model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ],
        },
        agent_record_id="record-1",
    )
    accumulator = SSEAccumulator()

    accumulator.feed("".join(events).encode())
    accumulator.finish()

    assert accumulator.agent_record_id == "record-1"
    assert accumulator.done is True
