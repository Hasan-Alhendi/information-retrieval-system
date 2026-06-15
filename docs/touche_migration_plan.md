# Quora + Webis-Touché 2020 v2 Migration Plan

## Goal

Replace Natural Questions with Webis-Touché 2020 v2 after validating the dataset locally.

The final pair will be:

- Quora: duplicate-question retrieval
- Webis-Touché 2020 v2: argument retrieval

## Phase 1 — Metadata inspection

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a metadata-only inspection:

```bash
python scripts/inspect_dataset.py --dataset touche2020-v2
```

This prints:

- document count
- query count
- qrel count
- query fields
- qrel fields
- sample queries
- sample qrels

It does not request document samples by default because document iteration may trigger downloading the corpus.

## Phase 2 — Controlled corpus download

After checking free disk space, request one document sample:

```bash
python scripts/inspect_dataset.py --dataset touche2020-v2 --sample-documents 1
```

Then verify that a document contains:

- `doc_id`
- `text`
- `title`
- `stance`
- `url`

## Phase 3 — Dataset-specific processing

Planned processing profiles:

### Quora question profile

- normalize Unicode and whitespace
- preserve question words
- preserve negation
- use the question text as the document body

### Touché argument profile

- clean HTML and boilerplate
- preserve negation and argumentative terms
- boost the title field
- keep stance and URL as metadata
- chunk long texts before dense embedding

## Phase 4 — Full-corpus indexing

The existing subset mode remains available for development. Full indexing will use:

- streaming corpus iteration
- batch preprocessing
- incremental dense-vector indexing
- disk-backed lexical indexes
- checkpoints and resume support
- MiniBatchKMeans for full-corpus clustering

## Safety rule

Natural Questions remains supported until Touché metadata, download, and sample fields are successfully verified on the target laptop.
