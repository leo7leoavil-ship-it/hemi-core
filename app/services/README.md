# Core Services & AI Pipelines

## Overview
Domain business logic and AI inference pipelines orchestrating faster-whisper speech recognition, ChromaDB vector retrieval, Groq Cloud LLM reasoning, and Hugging Face Hub corpus synchronization.

## Interface
- [asr_service.py:ASRService]: In-memory ASR singleton running `faster-whisper` (`large-v3-turbo`, CPU `int8`) over `io.BytesIO`.
- [asr_service.py:get_asr_service]: Dependency provider returning the `ASRService` singleton.
- [asr_service.py:compute_error_rates]: Computes Word Error Rate (WER) and Character Error Rate (CER) via `jiwer`.
- [rag_service.py:RAGService]: Singleton managing ChromaDB vector index and SentenceTransformer embeddings (`all-MiniLM-L6-v2`).
- [rag_service.py:get_rag_service]: Dependency provider returning the `RAGService` singleton.
- [llm_service.py:LLMService]: Singleton client for Groq Cloud (`openai/gpt-oss-120b`) with Structured Outputs and CEFR rubrics.
- [llm_service.py:get_llm_service]: Dependency provider returning the `LLMService` singleton.
- [llm_service.py:compute_lexical_density]: Deterministic Python computation of engineering lexical density ratio.
- [evaluation_pipeline.py:run_voice_pipeline]: Full voice assessment workflow (Audio -> ASR -> ChromaDB -> Groq LLM).
- [evaluation_pipeline.py:run_text_pipeline]: Text assessment workflow (Text -> ChromaDB -> Groq LLM).
- [corpus_sync.py:ensure_corpus_available]: Startup utility downloading the pre-indexed ChromaDB directory from Hugging Face Hub if missing locally.

## Invariants
- [llm_service.py:compute_lexical_density]: Lexical density is computed strictly deterministically in Python; never fabricated or guessed by the LLM.
- [llm_service.py:LLMServiceError]: LLM reasoning failures raise `LLMServiceError` (mapped to HTTP 502); fallback scores are never fabricated.
- [asr_service.py:ASRService.transcribe_audio_stream]: In-memory audio processing without writing temporary audio files to disk.
- [evaluation_pipeline.py:PipelineError]: Domain pipeline failures map to explicit HTTP status codes (400, 413, 422, 502, 503).

## Side Effects
- [corpus_sync.py:ensure_corpus_available]: Downloads dataset files from Hugging Face Hub to `data/technical_corpus_db`.
- [rag_service.py:RAGService]: Accesses persistent SQLite file `chroma.sqlite3` via ChromaDB client.
- [llm_service.py:LLMService.evaluate_technical_english]: Dispatches HTTPS network requests to Groq Cloud endpoint.

## Verification
- Run: `pytest`
