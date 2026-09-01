# Data

The corpus is ~5 GB and is not in this repository. This file explains what is needed, where
to get it, and how to confirm it arrived intact.

## What the app needs to run

Place these under `data/`, keeping the structure exactly as shown:

```
data/
├── chroma_v3/                              1.4 GB   the vector store
│   ├── chroma.sqlite3                      1.05 GB  passages + metadata
│   └── 698832b1-9b33-48b6-abcb-690d9f710614/  204 MB  the vectors themselves
│       ├── data_level0.bin                  201 MB
│       ├── header.bin
│       ├── index_metadata.pickle
│       ├── length.bin
│       └── link_lists.bin
└── processed/
    ├── sjc_normalized.json                  525 MB   full judgment text
    ├── document_splits_v2.json              168 MB   full legislation text
    └── ccb_normalized.json                    5 MB   full constitutional-court text
```

**Do not rename the `698832b1-…` folder.** Chroma matches it against the segment id stored
inside `chroma.sqlite3`; renamed, the collection will not load.

**`chroma.sqlite3` alone is not enough.** It holds the documents and metadata, but the
searchable vectors live in `data_level0.bin`. Without that folder Chroma opens the collection
and returns nothing for every question -- no error, just empty results.

The three JSON files are optional. Without them the chat is unaffected; البحث المباشر shows
200-character excerpts instead of full articles and judgments.

## Where to get it

Download from the project Google Drive folder (link supplied separately -- ask the authors).

## Verify the download

Google Drive reports no error when a file arrives truncated, so check before running:

```bash
python verify_chroma.py
```

It confirms the database size, all five vector files, the chunk count, and that all three
sources are present. It prints `RESULT: usable` or names the file that failed.

Expected:

```
chunks   49,782
highest id   c049781
sources   lloc 38,281 · sjc 11,339 · ccb 162
```

If the chunk count differs, the database is the wrong build or the download is incomplete.

## Rebuilding from source

Not required to run the app, and slow -- embedding 49,782 passages needs a GPU, which is why
`notebooks/04_build_chroma_v3.ipynb` runs on Colab.

```
raw scrape           notebooks/01a · 01b · 01c
  -> *_normalized.json     notebooks/02
  -> document_splits       notebooks/03
  -> chroma_v3             notebooks/04
```

**One gap worth stating plainly:** `notebooks/04` reads `document_splits_v3.json.gz`, while
`notebooks/03` in this repository produces `document_splits_v2.json`. The v3 file was made by
a later run of the same splitting notebook with adjusted parameters, and only its output was
kept -- the parameters themselves were not recorded. The difference is substantial:
`chroma_v3` holds 49,782 passages against the earlier store's 25,738.
