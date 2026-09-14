"""Check that a downloaded chroma_v3 is complete and usable.

Google Drive reports no error when a file arrives truncated, so run this before the app.
It reports the first thing that is wrong rather than every consequence of it.

    python verify_chroma.py                 # checks ./data/chroma_v3
    python verify_chroma.py path/to/store   # checks somewhere else
"""

import sqlite3
import sys
from pathlib import Path

EXPECTED_CHUNKS = 49_782
EXPECTED_SOURCES = {"lloc": 38_281, "sjc": 11_339, "ccb": 162}

# The vectors live in these; chroma.sqlite3 alone opens fine and returns nothing.
VECTOR_FILES = [
    "data_level0.bin",
    "header.bin",
    "index_metadata.pickle",
    "length.bin",
    "link_lists.bin",
]


def fail(message):
    print("\nRESULT: unusable — " + message)
    sys.exit(1)


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data" / "chroma_v3"
    if not root.is_dir():
        fail(f"no store at {root}")

    db = root / "chroma.sqlite3"
    if not db.is_file():
        fail(f"{db.name} is missing")
    size_gb = db.stat().st_size / 2 ** 30
    print(f"database    {size_gb:.2f} GB")
    if size_gb < 0.9:
        fail(f"{db.name} is only {size_gb:.2f} GB, so the download was truncated")

    # The segment folder is named after an id stored inside the database. Renaming it
    # stops the collection loading, so find it rather than hard-coding the name.
    segments = [p for p in root.iterdir() if p.is_dir()]
    if not segments:
        fail("the vector folder is missing entirely — chroma.sqlite3 on its own returns "
             "no results for every question")
    segment = segments[0]
    for name in VECTOR_FILES:
        f = segment / name
        if not f.is_file():
            fail(f"{segment.name}/{name} is missing")
        if f.stat().st_size == 0:
            fail(f"{segment.name}/{name} is empty")
    level0 = (segment / "data_level0.bin").stat().st_size / 2 ** 20
    print(f"vectors     {len(VECTOR_FILES)} files, data_level0.bin {level0:.0f} MB")

    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    n = con.execute("SELECT count(*) FROM embeddings").fetchone()[0]
    highest = con.execute("SELECT max(embedding_id) FROM embeddings").fetchone()[0]
    rows = dict(con.execute(
        "SELECT string_value, count(*) FROM embedding_metadata "
        "WHERE key = 'source' GROUP BY 1"))

    print(f"chunks      {n:,}")
    print(f"highest id  {highest}")
    print("sources     " + " · ".join(f"{s} {rows.get(s, 0):,}" for s in EXPECTED_SOURCES))

    if n != EXPECTED_CHUNKS:
        fail(f"expected {EXPECTED_CHUNKS:,} chunks, found {n:,} — this is either the wrong "
             f"build or an incomplete download")
    missing = [s for s in EXPECTED_SOURCES if s not in rows]
    if missing:
        fail("no chunks at all from: " + ", ".join(missing))
    off = {s: rows[s] for s, want in EXPECTED_SOURCES.items() if rows[s] != want}
    if off:
        fail("source counts do not match: " + ", ".join(
            f"{s} {rows[s]:,} (expected {EXPECTED_SOURCES[s]:,})" for s in off))

    print("\nRESULT: usable")


if __name__ == "__main__":
    main()
