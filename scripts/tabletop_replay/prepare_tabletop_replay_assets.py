"""Copy the tabletop inputs into assets/TabletopReplay (own copy; tasks do not share assets).

Run from the repository root with Isaac Sim's interpreter::

    ./app/python.sh scripts/tabletop_replay/prepare_tabletop_replay_assets.py

Same sources, file list and checks as
``scripts/tabletop_pickplace/prepare_tabletop_pickplace_assets.py``; only the target differs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.tabletop_pickplace.prepare_tabletop_pickplace_assets import SOURCE, prepare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=Path("assets/TabletopReplay"))
    parser.add_argument("--source", type=Path, default=SOURCE)
    args = parser.parse_args()
    prepare(args.target, source=args.source, task="tabletop_replay")


if __name__ == "__main__":
    main()
