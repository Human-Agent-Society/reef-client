"""Stdlib-only SDK for harnesses talking to a Reef service.

Components: ``client`` (the ask/observe/report loop) and ``skill`` (fetch
and sync the served skill file). Everything is standard library only, so
external harnesses can copy or import the package with no dependency on the
``reef`` package.
"""

from reef_client.client import ReefClient, ReefClientError
from reef_client.skill import SkillNotServedError, fetch_skill, sync_skill

__all__ = [
    "ReefClient",
    "ReefClientError",
    "SkillNotServedError",
    "fetch_skill",
    "sync_skill",
]
