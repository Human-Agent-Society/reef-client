"""OpenAI-style SSE helpers, shared by the serve mode and graders.

Two directions, both stdlib-only:

* ``synthesize_sse_events`` re-emits one buffered chat completion as
  ``chat.completion.chunk`` events. Training capture forces non-streaming
  upstream (the complete response is needed for trajectory capture), but
  streaming clients must still receive a protocol-valid stream.
* ``SSEAccumulator`` does the inverse: it consumes a live
  ``chat.completion.chunk`` stream and rebuilds the buffered completion,
  while retaining Reef's terminal ``agent_record_id`` receipt, so a
  passthrough proxy can capture trajectories without delaying the
  client-facing stream.

Both directions carry every field of a message, not a list of known ones: a field that one of them dropped, such
as a reasoning model's ``reasoning``, never reached the agent, which then sent the model a history without it.
"""

from __future__ import annotations

import codecs
import json
from typing import Any


#: The message fields that carry a model's reasoning text: vLLM writes ``reasoning``, and others ``reasoning_content``.
#: Both directions carry them as they carry any other field; the name stays for callers of 0.4.1.
REASONING_FIELDS = ("reasoning", "reasoning_content")
#: The delta fields that ``SSEAccumulator`` adds up by their own rules; it keeps every other field as it comes.
MESSAGE_FIELDS = ("role", "content", "tool_calls")
#: The chunk fields that ``SSEAccumulator`` reads by their own rules; it keeps every other one, the last value wins.
CHUNK_FIELDS = ("id", "object", "created", "model", "choices", "usage", "reef")
#: The choice fields that ``SSEAccumulator`` reads by their own rules.
CHOICE_FIELDS = ("index", "delta", "finish_reason")


def synthesize_sse_events(
    payload: dict[str, Any],
    *,
    include_usage: bool = False,
    agent_record_id: str | None = None,
) -> list[str]:
    """Re-emit one buffered chat completion as OpenAI stream events.

    Content and tool calls each arrive as a single delta rather than
    token-by-token; client SDKs treat delta granularity as an implementation
    detail, so one large delta is protocol-valid. Every other field of the
    message that holds a value, such as ``reasoning`` or ``reasoning_content``,
    is a delta of its own before the content, under its own name.
    """
    base = {
        "id": payload.get("id", "chatcmpl-reef-client"),
        "object": "chat.completion.chunk",
        "created": payload.get("created", 0),
        "model": payload.get("model", ""),
    }
    events: list[str] = []

    def emit(delta: dict[str, Any], index: int, finish_reason: str | None = None) -> None:
        chunk = {**base, "choices": [{"index": index, "delta": delta, "finish_reason": finish_reason}]}
        events.append(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n")

    for index, choice in enumerate(payload.get("choices", [])):
        message = choice.get("message") or {}
        emit({"role": message.get("role", "assistant")}, index)
        # Every field the stream has no rule for, under the upstream's own name: an agent that never receives a
        # reasoning model's thinking cannot send it back, and the model then reads a history of its own empty turns.
        for field, value in message.items():
            if field not in MESSAGE_FIELDS and value is not None and value != "":
                emit({field: value}, index)
        content = message.get("content")
        if content:
            emit({"content": content}, index)
        for call_index, call in enumerate(message.get("tool_calls") or []):
            function = call.get("function") or {}
            emit(
                {
                    "tool_calls": [
                        {
                            "index": call_index,
                            "id": call.get("id"),
                            "type": call.get("type", "function"),
                            "function": {
                                "name": function.get("name"),
                                "arguments": function.get("arguments", ""),
                            },
                        }
                    ]
                },
                index,
            )
        emit({}, index, finish_reason=choice.get("finish_reason", "stop"))

    if include_usage and payload.get("usage") is not None:
        events.append(
            f"data: {json.dumps({**base, 'choices': [], 'usage': payload['usage']}, ensure_ascii=False)}\n\n"
        )
    if agent_record_id is not None:
        events.append(
            f"data: {json.dumps({**base, 'choices': [], 'reef': {'agent_record_id': agent_record_id}}, ensure_ascii=False)}\n\n"
        )
    events.append("data: [DONE]\n\n")
    return events


class SSEAccumulator:
    """Rebuild a buffered chat completion from a chunk stream.

    Feed raw response bytes as they arrive; ``completion`` then returns a
    dict shaped like a non-streaming chat completion. Delta concatenation
    follows OpenAI semantics: content and tool-call arguments accumulate by
    (choice index, tool-call index), scalar fields keep their first value.
    Every other delta field is kept under its own name: text adds up, another
    value replaces the one before it. A choice's ``logprobs.content`` adds up,
    and each other field of a chunk or of a choice keeps its last value.
    """

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._buffer = ""
        self.done = False
        self.id = ""
        self.model = ""
        self.created = 0
        self.usage: dict[str, Any] | None = None
        self.agent_record_id: str | None = None
        self._extra: dict[str, Any] = {}
        self._choices: dict[int, dict[str, Any]] = {}

    def feed(self, data: bytes) -> None:
        self._buffer += self._decoder.decode(data)
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._handle_line(line.strip())

    def finish(self) -> None:
        """Flush any trailing bytes once the response is exhausted."""
        tail = self._buffer + self._decoder.decode(b"", final=True)
        self._buffer = ""
        if tail.strip():
            self._handle_line(tail.strip())

    def _handle_line(self, line: str) -> None:
        if not line or line.startswith(":") or not line.startswith("data:"):
            return  # blank separator, keepalive comment, or event:/id: field
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            self.done = True
            return
        try:
            chunk = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # A malformed frame costs its own content, never the stream: the
            # relay is mid-flight to the client and the turn is still worth
            # capturing from the frames that do parse.
            return
        if isinstance(chunk, dict):
            self._apply_chunk(chunk)

    def _apply_chunk(self, chunk: dict[str, Any]) -> None:
        self.id = chunk.get("id") or self.id
        self.model = chunk.get("model") or self.model
        self.created = chunk.get("created") or self.created
        if chunk.get("usage") is not None:
            self.usage = chunk["usage"]
        reef = chunk.get("reef")
        if isinstance(reef, dict) and isinstance(reef.get("agent_record_id"), str):
            self.agent_record_id = reef["agent_record_id"]
        for field, value in chunk.items():
            if field not in CHUNK_FIELDS and value is not None:
                self._extra[field] = value
        for choice in chunk.get("choices") or []:
            index = choice.get("index", 0)
            slot = self._choices.setdefault(
                index,
                {"role": None, "content": [], "tool_calls": {}, "finish_reason": None, "fields": {}, "choice": {}},
            )
            delta = choice.get("delta") or {}
            if delta.get("role"):
                slot["role"] = delta["role"]
            if delta.get("content"):
                slot["content"].append(delta["content"])
            for field, value in delta.items():
                if field in MESSAGE_FIELDS or value is None:
                    continue
                kept = slot["fields"].get(field)
                if isinstance(value, str) and (kept is None or isinstance(kept, str)):
                    slot["fields"][field] = (kept or "") + value
                elif value != "":
                    slot["fields"][field] = value
            for field, value in choice.items():
                if field in CHOICE_FIELDS or value is None:
                    continue
                kept = slot["choice"].get(field)
                if (
                    field == "logprobs"
                    and isinstance(value, dict)
                    and isinstance(kept, dict)
                    and isinstance(value.get("content"), list)
                    and isinstance(kept.get("content"), list)
                ):
                    slot["choice"][field] = {**kept, **value, "content": kept["content"] + value["content"]}
                else:
                    slot["choice"][field] = value
            for call in delta.get("tool_calls") or []:
                acc = slot["tool_calls"].setdefault(
                    call.get("index", 0), {"id": None, "type": None, "name": [], "arguments": []}
                )
                if call.get("id"):
                    acc["id"] = call["id"]
                if call.get("type"):
                    acc["type"] = call["type"]
                function = call.get("function") or {}
                if function.get("name"):
                    acc["name"].append(function["name"])
                if function.get("arguments"):
                    acc["arguments"].append(function["arguments"])
            if choice.get("finish_reason"):
                slot["finish_reason"] = choice["finish_reason"]

    @property
    def completion(self) -> dict[str, Any] | None:
        """The accumulated completion, or None if no chunk arrived yet."""
        if not self._choices and self.usage is None:
            return None
        choices = []
        for index in sorted(self._choices):
            slot = self._choices[index]
            message: dict[str, Any] = {
                "role": slot["role"] or "assistant",
                "content": "".join(slot["content"]),
            }
            message.update({field: value for field, value in slot["fields"].items() if value != ""})
            if slot["tool_calls"]:
                message["tool_calls"] = [
                    {
                        "id": acc["id"],
                        "type": acc["type"] or "function",
                        "function": {
                            "name": "".join(acc["name"]),
                            "arguments": "".join(acc["arguments"]),
                        },
                    }
                    for _, acc in sorted(slot["tool_calls"].items())
                ]
            choices.append(
                {"index": index, "message": message, "finish_reason": slot["finish_reason"], **slot["choice"]}
            )
        completion: dict[str, Any] = {
            **self._extra,
            "id": self.id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": choices,
        }
        if self.usage is not None:
            completion["usage"] = self.usage
        return completion
