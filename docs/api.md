# API Documentation

The FastAPI application is defined in:

```text
app/main.py
```

Run it with:

```bash
uvicorn app.main:app --reload
```

Open the interactive documentation:

```text
http://127.0.0.1:8000/docs
```

## Health Check

```text
GET /health
```

Response:

```json
{"status": "ok"}
```

## List Datasets

```text
GET /datasets
```

Returns the datasets registered in the system.

## Search

```text
POST /search
```

Example request:

```json
{
  "query": "information retrieval ranking",
  "dataset_name": "msmarco",
  "model_name": "bm25",
  "top_k": 10,
  "max_docs": 1000,
  "bm25_k1": 1.5,
  "bm25_b": 0.75,
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "use_query_refinement": false
}
```

Supported `model_name` values:

- `tfidf`
- `bm25`
- `embedding`
- `hybrid_serial`
- `hybrid_parallel`

## Evaluation

```text
GET /evaluation/{dataset_name}/{model_name}
```

Example:

```text
/evaluation/msmarco/bm25?max_docs=1000&max_queries=10&top_k=10
```

Response contains:

- MAP
- Recall
- Precision@10
- nDCG
- evaluated query count

## Indexing

```text
POST /indexes/{dataset_name}/{model_name}
```

This route currently returns a scheduled status placeholder. Index building is normally done through:

```bash
python scripts/build_indexes.py --dataset msmarco --model bm25 --max-docs 1000
```
