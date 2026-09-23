"""Import tooling works in the standalone, standard-library-only client."""

import fcntl
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from reef_client.record_import import import_records_file, record_batches


def imported(record_id, payload=None):
    return {
        "agent_record_id": record_id,
        "request_type": "inference",
        "payload": payload or {"text": record_id},
    }


def write_records(path, count):
    path.write_bytes(
        b"".join(
            json.dumps(imported(str(index))).encode() + b"\n" for index in range(count)
        )
    )


def test_batches_bound_bytes_count_and_preserve_resume_offsets():
    rows = [
        json.dumps(
            imported(str(index), {"text": "汉字" * index}), ensure_ascii=False
        ).encode()
        for index in range(8)
    ]
    data = b"\n\n".join(rows) + b"\n"
    source = io.BytesIO(data)
    batches = list(record_batches(source, batch_size=3, max_bytes=400))
    assert sum(count for _, _, count in batches) == 8
    assert all(len(body) <= 400 and count <= 3 for body, _, count in batches)
    _, first_offset, _ = batches[0]
    source.seek(first_offset)
    resumed = list(record_batches(source, batch_size=3, max_bytes=400))
    assert resumed == batches[1:]
    ids = [
        item["agent_record_id"]
        for body, _, _ in batches
        for item in json.loads(body)["records"]
    ]
    assert ids == [str(index) for index in range(8)]
    with pytest.raises(ValueError, match="exceeds"):
        list(record_batches(io.BytesIO(b"x" * 401), batch_size=3, max_bytes=400))
    with pytest.raises(ValueError, match="invalid JSONL"):
        list(record_batches(io.BytesIO(b"not json\n"), batch_size=3, max_bytes=400))


def test_import_rejects_concurrent_checkpoint_use(tmp_path):
    source = tmp_path / "records.jsonl"
    write_records(source, 1)
    progress = tmp_path / "progress.json"
    with progress.with_name(progress.name + ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="another importer"):
            import_records_file(
                source,
                url="http://localhost:8900",
                scenario="s",
                progress_path=progress,
            )


def test_import_command_works_without_site_packages(tmp_path):
    # Copying the dependency-free package is a supported harness deployment.
    import shutil

    root = Path(__file__).resolve().parents[1]
    shutil.copytree(
        root / "reef_client",
        tmp_path / "reef_client",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    result = subprocess.run(
        [sys.executable, "-S", "-m", "reef_client", "import", "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "reef-client import" in result.stdout
    assert "JSONL" in result.stdout
    assert "--progress" in result.stdout
    assert "--max-batch-bytes" in result.stdout


def test_standalone_command_uploads_and_resumes_without_server_dependencies(tmp_path):
    import os
    import shutil
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    root = Path(__file__).resolve().parents[1]
    shutil.copytree(
        root / "reef_client",
        tmp_path / "reef_client",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    source = tmp_path / "records.jsonl"
    write_records(source, 5)
    received = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            assert self.path == "/reef/records/batch"
            assert self.headers["Authorization"] == "Bearer fixture-token"
            assert self.headers["x-reef-scenario"] == "s"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(body["records"])
            payload = json.dumps(
                {
                    "records": [
                        {
                            "agent_record_id": item["agent_record_id"],
                            "request_type": item["request_type"],
                            "scenario": "s",
                        }
                        for item in body["records"]
                    ]
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        worker = threading.Thread(target=server.serve_forever)
        worker.start()
        try:
            command = [
                sys.executable,
                "-S",
                "-m",
                "reef_client",
                "import",
                str(source),
                "--url",
                f"http://127.0.0.1:{server.server_port}",
                "--scenario",
                "s",
                "--batch-size",
                "2",
            ]
            for _ in range(2):
                result = subprocess.run(
                    command,
                    cwd=tmp_path,
                    text=True,
                    capture_output=True,
                    env={**os.environ, "REEF_TOKEN": "fixture-token"},
                    timeout=10,
                )
                assert result.returncode == 0, result.stderr
                assert "5 records acknowledged" in result.stdout
                assert [len(batch) for batch in received] == [2, 2, 1]
        finally:
            server.shutdown()
            worker.join(timeout=5)
