"""Writing JSON the way every artefact in this project is written.

Two rules, in one place because every file the benchmark publishes depends on
them:

* **Atomic.** Write a temporary file beside the target and rename it. A run
  interrupted halfway leaves the previous file intact rather than a truncated
  one.
* **``newline=""``.** Never translate ``\\n`` to ``\\r\\n`` on Windows. Gold
  evidence is character offsets into these files; a platform that silently adds
  a byte per line moves every offset after it.

It lives on its own so that the scoring half of the package — the half someone
copies into their own project — does not have to import the corpus builder to
write a report.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def write_atomic(path: Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        handle.write(content)
    os.replace(tmp, path)


def dumps(obj, *, compact: bool = False) -> str:
    if compact:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"
    return json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
