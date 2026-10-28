"""Commands run on the user machine against a remote Reef service."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "import":
        from reef_client.record_import import main as import_main

        import_main(arguments[1:])
        return
    if arguments and arguments[0] == "harness":
        from reef_client.harness import main as harness_main

        harness_main(arguments[1:])
        return
    parser = argparse.ArgumentParser(prog="reef-client", description=__doc__)
    parser.add_argument(
        "command",
        choices=("import", "harness"),
        help="import: upload a records JSONL file with resumable batches; "
        "harness: sync an agent's harness with Reef and run the agent on its release",
    )
    parser.parse_args(arguments)


if __name__ == "__main__":
    main()
