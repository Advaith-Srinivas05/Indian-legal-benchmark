"""Allow ``python -m ingestion`` as an alias for ``python -m ingestion.download``."""

import sys

from .download import main

if __name__ == "__main__":
    sys.exit(main())
