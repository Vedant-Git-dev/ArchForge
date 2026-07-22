"""Entry point for `python -m archforge`."""

from __future__ import annotations

import sys

from archforge.cli import main

if __name__ == "__main__":
    sys.exit(main())
