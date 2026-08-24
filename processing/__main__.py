"""``python -m processing`` runs the extraction benchmark."""

from __future__ import annotations

import sys

from .benchmark import main

if __name__ == "__main__":
    sys.exit(main())
