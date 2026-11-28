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
The accumulator reads the three streams that a model call can answer with: Chat Completions chunks, OpenAI
Responses events and Anthropic Messages events.
"""

from __future__ import annotations

import codecs
import io
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
#: The event types of an Anthropic Messages stream.
MESSAGE_EVENTS = (
    "message_start",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
    "message_delta",
    "message_stop",
    "ping",
)


def added_up(kept: Any, value: Any) -> Any:
    """``value`` added to ``kept``, the value so far of one field of a stream: text and lists add up, and any other
    value replaces ``kept``. Purpose: the accumulator keeps a field that it has no rule for as the stream sent it."""
    if isinstance(value, str) and (kept is None or isinstance(kept, str)):
        return (kept or "") + value
    if isinstance(value, list) and isinstance(kept, list):
        return kept + value
    return value


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
    is a delta of its own before the content, under its own name. Every other
    field of a choice, such as ``logprobs`` or ``token_ids``, rides on the
    choice's closing chunk, and every other field of the completion, such as
    ``prompt_token_ids`` or a private ``training`` block, on a closing chunk of
    its own: ``SSEAccumulator`` rebuilds the whole completion from the stream.
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
        extra = {
            field: value for field, value in choice.items() if field not in ("index", "message", "finish_reason")
        }
        chunk = {
            **base,
            "choices": [
                {"index": index, "delta": {}, "finish_reason": choice.get("finish_reason", "stop"), **extra}
            ],
        }
        events.append(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n")

    rest = {field: value for field, value in payload.items() if field not in CHUNK_FIELDS}
    if rest:
        events.append(f"data: {json.dumps({**base, 'choices': [], **rest}, ensure_ascii=False)}\n\n")
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
    value replaces the one before it. Each other field of a choice adds up the
    same way, lists too (``token_ids``), and ``logprobs.content`` adds up; each
    other field of a chunk keeps its last value.

    An OpenAI Responses stream gives the response that its last event holding one
    carries (``response.completed``, ``response.incomplete`` or
    ``response.failed``). An Anthropic Messages stream gives the message that its
    events build: ``message_start`` gives the message, each content block adds up
    its deltas (``input_json_delta`` text becomes the block's ``input`` when the
    block stops), and ``message_delta`` sets its delta, adds its ``usage`` and sets
    each other field it holds. An ``error`` event of either stream is the
    completion when the stream gave nothing else. ``stream_failed`` records an explicit protocol error
    even when the stream gave part of an answer first.
    """

    def __init__(self) -> None:
        self._decoder = io.IncrementalNewlineDecoder(codecs.getincrementaldecoder("utf-8")(), translate=True)
        self._buffer = ""
        self._event = ""
        self._data: list[str] = []
        self.stream_failed = False
        self.done = False
        self.id = ""
        self.model = ""
        self.created = 0
        self.usage: dict[str, Any] | None = None
        self.agent_record_id: str | None = None
        self._extra: dict[str, Any] = {}
        self._response: dict[str, Any] | None = None
        self._message: dict[str, Any] | None = None
        self._blocks: dict[int, dict[str, Any]] = {}
        self._partial_json: dict[int, str] = {}
        self._error: dict[str, Any] | None = None
        self._choices: dict[int, dict[str, Any]] = {}

    def feed(self, data: bytes) -> None:
        self._buffer += self._decoder.decode(data)
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._handle_line(line)

    def finish(self) -> None:
        """Flush any trailing bytes once the response is exhausted."""
        tail = self._buffer + self._decoder.decode(b"", final=True)
        self._buffer = ""
        for line in tail.split("\n"):
            self._handle_line(line)
        self._dispatch()

    def _handle_line(self, line: str) -> None:
        if not line:
            self._dispatch()
        elif line.startswith("event:"):
            self._event = line[6:].removeprefix(" ")
        elif line.startswith("data:"):
            self._data.append(line[5:].removeprefix(" "))

    def _dispatch(self) -> None:
        event, lines = self._event, self._data
        self._event, self._data = "", []
        if not lines:
            return
        if event in ("error", "response.failed"):
            self.stream_failed = True
        data = "\n".join(lines).strip()
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
        kind = chunk.get("type")
        if kind in ("error", "response.failed") or (isinstance(chunk.get("error"), dict) and not chunk.get("choices")):
            self.stream_failed = True
        if kind == "error":
            self._error = chunk
            return
        if isinstance(kind, str) and kind.startswith("response."):
            if isinstance(chunk.get("response"), dict):
                self._response = chunk["response"]
            return
        if kind in MESSAGE_EVENTS:
            self._apply_message_event(chunk)
            return
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
                if value != "" or field in slot["fields"]:
                    slot["fields"][field] = added_up(slot["fields"].get(field), value)
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
                    slot["choice"][field] = added_up(kept, value)
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

    def _apply_message_event(self, event: dict[str, Any]) -> None:
        kind = event["type"]
        if kind == "message_start" and isinstance(event.get("message"), dict):
            self._message = dict(event["message"])
            for index, block in enumerate(self._message.get("content") or []):
                if isinstance(block, dict):
                    self._blocks[index] = dict(block)
            return
        index = event.get("index")
        if kind == "content_block_start" and isinstance(index, int) and isinstance(event.get("content_block"), dict):
            self._blocks[index] = dict(event["content_block"])
            return
        if kind == "content_block_delta" and isinstance(index, int) and isinstance(event.get("delta"), dict):
            block = self._blocks.setdefault(index, {})
            for field, value in event["delta"].items():
                if field == "type" or value is None:
                    continue
                if field == "partial_json" and isinstance(value, str):
                    self._partial_json[index] = self._partial_json.get(index, "") + value
                elif field == "citation":
                    block.setdefault("citations", []).append(value)
                else:
                    block[field] = added_up(block.get(field), value)
            return
        if kind == "content_block_stop" and isinstance(index, int) and index in self._partial_json:
            text = self._partial_json.pop(index)
            try:
                self._blocks.setdefault(index, {})["input"] = json.loads(text) if text else {}
            except json.JSONDecodeError:
                # The tool's arguments as the model wrote them: a capture keeps what the stream sent.
                self._blocks[index]["input"] = text
            return
        if kind == "message_delta":
            message = self._message if self._message is not None else {}
            self._message = message
            if isinstance(event.get("delta"), dict):
                message.update({field: value for field, value in event["delta"].items()})
            if isinstance(event.get("usage"), dict):
                usage = message.get("usage") if isinstance(message.get("usage"), dict) else {}
                message["usage"] = {**usage, **event["usage"]}
            for field, value in event.items():
                if field not in ("type", "delta", "usage") and value is not None:
                    message[field] = value

    @property
    def completion(self) -> dict[str, Any] | None:
        """The accumulated completion, or None if no chunk arrived yet: a chat completion, an OpenAI response or
        an Anthropic message, as the stream was."""
        if self._response is not None:
            return self._response
        if self._message is not None:
            for index, text in self._partial_json.items():
                # A block that never stopped: keep the arguments as they came.
                self._blocks.setdefault(index, {}).setdefault("input", text)
            return {**self._message, "content": [self._blocks[index] for index in sorted(self._blocks)]}
        if not self._choices and self.usage is None:
            return self._error
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
