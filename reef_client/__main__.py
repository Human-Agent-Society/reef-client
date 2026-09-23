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
    parser = argparse.ArgumentParser(prog="reef-client", description=__doc__)
    parser.add_argument(
        "command",
        choices=("import",),
        help="import a records JSONL file with resumable batch uploads",
    )
    parser.parse_args(arguments)


if __name__ == "__main__":
    main()
