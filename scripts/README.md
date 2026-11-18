# Operational & Development Scripts

## Overview
Operational CLI entry points and seeding utilities for executing the server locally and ingesting technical training corpora into ChromaDB.

## Interface
- [run_server.py:main]: CLI entry point running `uvicorn.run("app.main:app")` configured from application settings.
- [seed_corpus.py:batch_ingest]: Batch ingestion function embedding text chunks and inserting them into ChromaDB.

## Invariants
- [seed_corpus.py:batch_ingest]: Requires initialized `RAGService` with operational ChromaDB collection and SentenceTransformer model.
- [run_server.py:main]: Configures `sys.path` to include project root before importing application modules.

## Side Effects
- [seed_corpus.py:batch_ingest]: Writes vector embeddings and metadata to `data/technical_corpus_db/chroma.sqlite3`.
- [run_server.py:main]: Binds and runs Uvicorn HTTP server on configured network host/port.

## Verification
- Run: `pytest`
