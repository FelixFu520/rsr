"""Collect, independently validate and publish tabletop replay episodes.

Run: ./app/python.sh scripts/tabletop_replay/collect_tabletop_replay.py
Uses the same clock workaround and publication checks as the pick-and-place collector.
"""

from scripts.tabletop_pickplace.collect_tabletop_pickplace import main

if __name__ == "__main__":
    main("tabletop_replay")
