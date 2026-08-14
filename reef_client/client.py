"""Stdlib-only client that external harnesses can copy or import."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any


class ReefClientError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"Reef service returned {status}: {body[:400]}")
        self.status = status
        self.body = body


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
    ) -> dict[str, Any]:
        return self._post(
            path,
            scenario,
            payload,
            recipe=recipe,
            artifact_version=artifact_version,
        )

    def inference_with_record(
        self,
        scenario: str,
        path: str,
        payload: Mapping[str, Any],
        *,
        recipe: str | None = None,
        artifact_version: str | None = None,
    ) -> tuple[dict[str, Any], str]:
        body, headers = self._post_with_headers(
            path,
            scenario,
            payload,
            recipe=recipe,
            artifact_version=artifact_version,
        )
        agent_data_id = headers.get("x-reef-agent-data-id")
        if not agent_data_id:
            raise ReefClientError(502, "inference response is missing x-reef-agent-data-id")
        return body, agent_data_id

    def report(
        self,
        scenario: str,
        payload: Mapping[str, Any],
        *,
        recipe: str | None = None,
        artifact_version: str | None = None,
    ) -> dict[str, Any]:
        return self._post(
            "/reef/report",
            scenario,
            payload,
            recipe=recipe,
            artifact_version=artifact_version,
        )

    def event(
        self,
        scenario: str,
        payload: Mapping[str, Any],
        *,
        recipe: str | None = None,
        artifact_version: str | None = None,
    ) -> dict[str, Any]:
        return self._post(
            "/reef/event",
            scenario,
            payload,
            recipe=recipe,
            artifact_version=artifact_version,
        )

    def _post(
        self,
        path: str,
        scenario: str,
        payload: Mapping[str, Any],
        *,
        recipe: str | None = None,
        artifact_version: str | None = None,
    ) -> dict[str, Any]:
        body, _ = self._post_with_headers(
            path,
            scenario,
            payload,
            recipe=recipe,
            artifact_version=artifact_version,
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
    ) -> tuple[dict[str, Any], Mapping[str, str]]:
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
            raise ReefClientError(exc.code, body) from exc
        return (json.loads(body) if body else {}), {key.lower(): value for key, value in response_headers.items()}
