"""Stdlib-only client that external harnesses can copy or import."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any


class ReefClientError(RuntimeError):
    """The service answered >= 400. ``content_type`` is the bare mime type
    (parameters like charset stripped), mirroring aiohttp's ``.content_type``."""

    def __init__(self, status: int, body: str, content_type: str | None = None) -> None:
        super().__init__(f"Reef service returned {status}: {body[:400]}")
        self.status = status
        self.body = body
        self.content_type = content_type.split(";")[0].strip() if content_type else None


class ReefClient:
    def __init__(
        self,
        service_url: str,
        *,
        token: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self.service_url = service_url.rstrip("/")
        self.token = token
        self.timeout_s = timeout_s

    def inference(
        self,
        scenario: str,
        path: str,
        payload: Mapping[str, Any],
        *,
        recipe: str | None = None,
        artifact_version: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        return self._post(
            path,
            scenario,
            payload,
            recipe=recipe,
            artifact_version=artifact_version,
            extra_headers=extra_headers,
        )

    def inference_with_record(
        self,
        scenario: str,
        path: str,
        payload: Mapping[str, Any],
        *,
        recipe: str | None = None,
        artifact_version: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> tuple[dict[str, Any], str]:
        body, headers = self.post(
            path,
            scenario,
            payload,
            recipe=recipe,
            artifact_version=artifact_version,
            extra_headers=extra_headers,
        )
        agent_data_id = headers.get("x-reef-agent-data-id")
        if not agent_data_id:
            raise ReefClientError(502, "inference response is missing x-reef-agent-data-id")
        return body, agent_data_id

    def report(
        self,
        scenario: str,
        payload: Mapping[str, Any] | Any,
        *,
        references: Sequence[str] | None = None,
        recipe: str | None = None,
        artifact_version: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """POST a report: a plain wire dict, or a typed signal instance.

        Any object with a ``to_report(references=...)`` method (a recipe's
        declared ``reef.schemas.ReportSchema`` dataclass) is serialized through it,
        so the report body is built — and validated — by the contract's own
        code before the request leaves the process. ``references`` are the
        inference ``x-reef-agent-data-id`` receipts this report grades; with
        a plain dict they merge into the payload (the dict's own
        ``references`` key wins).
        """
        to_report = getattr(payload, "to_report", None)
        if callable(to_report):
            payload = to_report(references=tuple(references or ()))
        elif references is not None:
            payload = {"references": list(references), **payload}
        return self._post(
            "/reef/report",
            scenario,
            payload,
            recipe=recipe,
            artifact_version=artifact_version,
            extra_headers=extra_headers,
        )

    def event(
        self,
        scenario: str,
        payload: Mapping[str, Any],
        *,
        recipe: str | None = None,
        artifact_version: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        return self._post(
            "/reef/event",
            scenario,
            payload,
            recipe=recipe,
            artifact_version=artifact_version,
            extra_headers=extra_headers,
        )

    def post(
        self,
        path: str,
        scenario: str,
        payload: Mapping[str, Any],
        *,
        recipe: str | None = None,
        artifact_version: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> tuple[dict[str, Any], Mapping[str, str]]:
        """POST an arbitrary protocol path; return (parsed body, response headers).

        This is the low-level entry point behind ``inference``/``report``/
        ``event``, exposed for callers that need the response headers (e.g. a
        forwarding proxy capturing the ``x-reef-agent-data-id`` receipt, where
        a missing receipt is tolerated rather than an error) or that mirror
        upstream behavior. Response header names are lowercased.
        """
        return self._post_with_headers(
            path,
            scenario,
            payload,
            recipe=recipe,
            artifact_version=artifact_version,
            extra_headers=extra_headers,
        )

    def get(
        self,
        path: str,
        *,
        extra_headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """GET an arbitrary protocol path and return the parsed JSON body.

        Read-side counterpart of ``post`` (e.g.
        ``/reef/scenarios/{scenario}/versions``). On HTTP errors the raised
        ``ReefClientError`` carries ``.status``, so callers can distinguish a
        not-yet-created scenario (404) from a failed probe.
        """
        headers: dict[str, str] = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        covered = {name.lower() for name in headers}
        headers.update({name: value for name, value in (extra_headers or {}).items() if name.lower() not in covered})
        request = urllib.request.Request(f"{self.service_url}{path}", headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = response.read().decode()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise ReefClientError(exc.code, body, exc.headers.get("Content-Type")) from exc
        return json.loads(body) if body else {}

    def _post(
        self,
        path: str,
        scenario: str,
        payload: Mapping[str, Any],
        *,
        recipe: str | None = None,
        artifact_version: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        body, _ = self._post_with_headers(
            path,
            scenario,
            payload,
            recipe=recipe,
            artifact_version=artifact_version,
            extra_headers=extra_headers,
        )
        return body

    def _post_with_headers(
        self,
        path: str,
        scenario: str,
        payload: Mapping[str, Any],
        *,
        recipe: str | None = None,
        artifact_version: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> tuple[dict[str, Any], Mapping[str, str]]:
        # Explicit arguments own the protocol headers; extra_headers only fill
        # in what the arguments do not cover (scenario/recipe overrides must
        # therefore be resolved by the caller before dispatch).
        headers = {
            "Content-Type": "application/json",
            "x-reef-scenario": scenario,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if recipe:
            headers["x-reef-recipe"] = recipe
        if artifact_version:
            headers["x-reef-artifact-version"] = artifact_version
        covered = {name.lower() for name in headers}
        headers.update({name: value for name, value in (extra_headers or {}).items() if name.lower() not in covered})
        request = urllib.request.Request(
            f"{self.service_url}{path}",
            data=json.dumps(dict(payload)).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = response.read().decode()
                response_headers = dict(getattr(response, "headers", {}).items())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise ReefClientError(exc.code, body, exc.headers.get("Content-Type")) from exc
        return (json.loads(body) if body else {}), {key.lower(): value for key, value in response_headers.items()}
