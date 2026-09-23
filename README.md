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

## Import existing records

Run the importer on the machine holding your data, pointing it at a running
Reef service that supports `POST /reef/records/batch`:

```bash
reef-client import records.jsonl --url https://reef.example.com --scenario my-agent
# Also works without a console-script installation:
python -m reef_client import records.jsonl --url https://reef.example.com --scenario my-agent
```

No `reef-infra`, training libraries, or third-party HTTP client is required.
Authentication is read from `REEF_TOKEN` when set.

Each UTF-8 JSONL line is an existing record envelope. IDs must stay stable and
reports must follow their referenced inferences:

```jsonl
{"agent_record_id":"sample-1","request_type":"inference","payload":{"messages":[{"role":"user","content":"2+2?"}],"response":{"choices":[{"message":{"role":"assistant","content":"4"}}]}}}
{"agent_record_id":"score-1","request_type":"report","payload":{"references":["sample-1"],"score":1}}
```

The selected server recipe determines which inference payload and feedback fields
it needs. The importer preserves IDs and payloads; it does not run inference or
convert arbitrary dataset schemas.

- Reads incrementally, limiting each request by `--batch-size` (default 128)
  and `--max-batch-bytes` (default 524288). Requests must stay below 1 MiB and
  contain at most 1000 records. Each source line must fit the byte limit.
- Reuses the HTTP connection and retries transient failures with backoff
  (`--retries`, default 3). It sends unchanged record IDs on retry.
- Saves acknowledged byte offsets to `FILE.reef-import.json`, or `--progress PATH`.
  Re-run the same command after interruption. A full bounded-memory checksum pass
  verifies the source on each invocation before seeking to saved progress.
- Binds the checkpoint to the file checksum, service URL and scenario, and locks
  it against concurrent importers. Keep the source immutable. Checkpoints contain
  no token or record payloads. POSIX file locking requires Linux/macOS.
- Stops on permanent failures without advancing the failed batch's checkpoint.
  Previously acknowledged batches remain stored. Lost responses can be retried
  safely because the server deduplicates stable record IDs.

Python callers can use `reef_client.record_import.import_records_file(Path(...),
url=..., scenario=...)`. It returns the cumulative number of acknowledged source
rows, including existing records; it does not count completed training samples.
After import, continue sending chat/report traffic to the same scenario.
Import completion means records were acknowledged, not that training has finished.
