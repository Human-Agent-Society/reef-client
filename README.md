# reef_client

Stdlib-only client SDK for the [Reef](https://github.com/Human-Agent-Society/reef) protocol. `reef_client` is a client of
Reef's HTTP wire protocol, not of the `reef` package: it never imports `reef`
and has no dependencies, so external harnesses can install it or copy it into
environments that cannot take the `reef` wheel's dependencies.

The distribution is `reef-client` ([`pyproject.toml`](pyproject.toml)); the
import package is the inner [`reef_client/`](reef_client) directory:

- [`client.py`](reef_client/client.py) is `ReefClient`, the
  ask/observe/report loop every example shares (see the
  [Reef docs](https://github.com/Human-Agent-Society/reef/blob/main/docs/reference/reef-client.md)),
  plus the harness update channel reads: `harness_versions` (the version
  catalog with gate metrics) and `harness_pull` (write one served tree, head
  or version-addressed, into a directory).
- [`skill.py`](reef_client/skill.py) fetches and syncs the scenario's served
  skill file (`fetch_skill`/`sync_skill`);
- [`sse.py`](reef_client/sse.py) converts between buffered chat completions
  and OpenAI-style SSE chunk streams in both directions
  (`synthesize_sse_events` / `SSEAccumulator`).
- [`serve.py`](reef_client/serve.py) is serve mode: a local sidecar that
  agents treat as their model endpoint (`python -m reef_client.serve`),
  owning session stamping, trajectory capture, receipt collection, and SSE
  passthrough/synthesis so reef-oblivious agents need no SDK. See the
  [reference](https://github.com/Human-Agent-Society/reef/blob/main/docs/reference/reef-client.md#serve-mode-a-local-proxy-for-reef-oblivious-agents).

The client and skill helpers are re-exported at the package root:
`from reef_client import ReefClient, fetch_skill, sync_skill`. Import the
`sse`/`serve` submodules directly.

## Install

```bash
pip install reef-client
```

For development:

```bash
git clone https://github.com/Human-Agent-Society/reef-client && cd reef-client
pip install -e .
```

The package is standard library only — copying the inner `reef_client/`
directory into your harness works too.
