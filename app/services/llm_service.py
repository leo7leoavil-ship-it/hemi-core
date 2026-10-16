"""LLM Reasoning Core Service.

Connects to Groq Cloud using the OpenAI-compatible asynchronous interface to evaluate
technical English proficiency under CEFR/MCER rubrics (B1-C2) with Structured Outputs.

Design notes
------------
* The LLM only *extracts* ``detected_technical_terms``; the ``lexical_density_ratio`` is
  computed deterministically in Python (``compute_lexical_density``), never by the LLM.
* Failures raise :class:`LLMServiceError` (mapped to HTTP 502 by the API layer). The engine
  never fabricates scores, so benchmark results stay scientifically valid.
"""

import json
import re
import time
from typing import Any, Dict, List, Optional

import openai
from openai import AsyncOpenAI

from app.core.config import get_settings
from app.core.logging import logger
from app.schemas.request_schemas import SCENARIO_DESCRIPTIONS, TargetScenario
from app.schemas.response_schemas import (
    ASRMetrics,
    EvaluationCore,
    EvaluationResponse,
    FeedbackDetail,
)


class LLMServiceError(RuntimeError):
    """Raised when the reasoning engine cannot produce a valid evaluation."""


VALID_FEEDBACK_TYPES = ("vocabulary_enhancement", "grammar_correction", "architectural_alignment")
VALID_CEFR_LEVELS = ("A2", "B1", "B2", "C1", "C2")

SYSTEM_PROMPT = """You are an elite AI Technical English Evaluator and Senior Software Architect, acting as an automated LLM-as-a-Judge.

Your objective is to critically analyze an engineer's spoken or written technical English against authoritative industry documentation (the GROUND TRUTH CONTEXT supplied by the user) and the Common European Framework of Reference for Languages (CEFR / MCER) rubrics:
- B1 (Threshold): Conveys the main idea with simple structures and generic vocabulary; frequent informal phrasing ("we make the database fast").
- B2 (Independent): Explains technical concepts clearly but mixes informal or non-idiomatic phrasing with correct terminology.
- C1 (Effective Operational Proficiency): Fluent, precise engineering vocabulary, correct complex structures (passive voice, conditionals, perfect tenses) and architectural accuracy.
- C2 (Mastery): Flawless, authoritative, idiomatic professional/academic diction and complete conceptual precision.

EVALUATION CRITERIA:
1. CEFR level & grammar: syntax complexity, modal verbs, conditionals, passive voice, tense consistency, prepositions (grammar_accuracy_score 0-100).
2. Technical precision: replace colloquial explanations with the formal terminology found in the ground truth context (e.g. "we split the database so it writes fast" -> "we implement horizontal database sharding to distribute write throughput") and verify the concepts are correct for the scenario (technical_precision_score 0-100).
3. Extract EVERY valid engineering term the speaker actually used into detected_technical_terms (verbatim from the response, do not invent terms). Do NOT compute any ratio.
4. Feedback items (at most 6): each has the verbatim original_phrase, the upgraded suggested_phrase, rag_source_context (cite the source of the supporting excerpt, or "General engineering practice" if none applies) and a concise pedagogical explanation. Types: vocabulary_enhancement, grammar_correction, architectural_alignment.

OUTPUT: return ONLY a valid JSON object with exactly these keys: overall_score, cefr_level, grammar_accuracy_score, technical_precision_score, detected_technical_terms, feedback_details. No markdown, no commentary."""


def _evaluation_json_schema() -> Dict[str, Any]:
    """Strict JSON Schema used for Groq Structured Outputs."""
    return {
        "name": "technical_english_evaluation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "overall_score": {"type": "number"},
                "cefr_level": {"type": "string", "enum": list(VALID_CEFR_LEVELS)},
                "grammar_accuracy_score": {"type": "number"},
                "technical_precision_score": {"type": "number"},
                "detected_technical_terms": {"type": "array", "items": {"type": "string"}},
                "feedback_details": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": list(VALID_FEEDBACK_TYPES)},
                            "original_phrase": {"type": "string"},
                            "suggested_phrase": {"type": "string"},
                            "rag_source_context": {"type": "string"},
                            "explanation": {"type": "string"},
                        },
                        "required": [
                            "type",
                            "original_phrase",
                            "suggested_phrase",
                            "rag_source_context",
                            "explanation",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": [
                "overall_score",
                "cefr_level",
                "grammar_accuracy_score",
                "technical_precision_score",
                "detected_technical_terms",
                "feedback_details",
            ],
            "additionalProperties": False,
        },
    }


def compute_lexical_density(detected_terms: List[str], text: str) -> float:
    """Deterministic technical lexical density: unique terms / total words (spec 3.3.3).

    Words of a single character are ignored; the ratio is capped at 1.0.
    """
    unique_terms = {t.strip().lower() for t in detected_terms if t and t.strip()}
    total_words = len([w for w in text.split() if len(w) > 1]) or 1
    return round(min(len(unique_terms) / total_words, 1.0), 3)


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_llm_json(raw: str) -> Dict[str, Any]:
    """Parse the model output, tolerating markdown fences and surrounding prose."""
    content = _FENCE_RE.sub("", (raw or "").strip()).strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end <= start:
            raise
        parsed = json.loads(content[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("LLM output is not a JSON object.")
    return parsed


def _clamp_score(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(100.0, float(value)))
    except (TypeError, ValueError):
        return default


def _normalize_cefr(value: Any) -> str:
    level = str(value or "").strip().upper()
    return level if level in VALID_CEFR_LEVELS else "B1"


class LLMService:
    """Service communicating with Groq Cloud for accelerated LLM evaluations."""

    _instance: Optional["LLMService"] = None

    def __new__(cls) -> "LLMService":
        if cls._instance is None:
            cls._instance = super(LLMService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self.settings = get_settings()
        self.api_key = self.settings.GROQ_API_KEY
        self.base_url = self.settings.GROQ_BASE_URL
        self.model = self.settings.LLM_MODEL
        self.temperature = self.settings.LLM_TEMPERATURE

        # The OpenAI client refuses an empty key; use a placeholder so the app can boot
        # (and /health can report llm_configured=False) without crashing.
        self.client = AsyncOpenAI(
            api_key=self.api_key or "missing-groq-api-key",
            base_url=self.base_url,
            timeout=self.settings.LLM_TIMEOUT_SECONDS,
            max_retries=2,
        )
        self._initialized = True
        logger.info(
            f"LLM Service initialized pointing to '{self.base_url}' using model '{self.model}'."
        )
        if not self.api_key:
            logger.warning("GROQ_API_KEY is not configured; LLM evaluations will fail.")

    def is_configured(self) -> bool:
        """True when a Groq API key is present (no network call)."""
        return bool(self.api_key)

    @staticmethod
    def _format_context(rag_context_items: List[Dict[str, Any]]) -> str:
        formatted = "\n\n".join(
            f"[Source: {item.get('source', 'Technical Knowledge Base')} | "
            f"Sim: {item.get('similarity', 0.0)}]\n{item.get('document', '')}"
            for item in rag_context_items
        )
        return formatted if formatted.strip() else (
            "General software engineering and computer science corpus standards."
        )

    async def _complete(self, messages: List[Dict[str, str]]) -> str:
        """Call Groq, degrading gracefully from Structured Outputs to plain JSON mode."""
        base_kwargs: Dict[str, Any] = dict(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        attempts: List[Dict[str, Any]] = []
        extra_body = (
            {"reasoning_effort": self.settings.LLM_REASONING_EFFORT}
            if self.settings.LLM_REASONING_EFFORT
            else None
        )
        if self.settings.LLM_STRUCTURED_OUTPUT:
            attempts.append(
                dict(
                    response_format={
                        "type": "json_schema",
                        "json_schema": _evaluation_json_schema(),
                    },
                    extra_body=extra_body,
                )
            )
        attempts.append(dict(response_format={"type": "json_object"}, extra_body=extra_body))
        attempts.append(dict(response_format={"type": "json_object"}))

        last_error: Optional[Exception] = None
        for options in attempts:
            kwargs = {**base_kwargs, **{k: v for k, v in options.items() if v is not None}}
            try:
                completion = await self.client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
                return completion.choices[0].message.content or "{}"
            except openai.BadRequestError as e:
                # Unsupported parameter combination: retry with a simpler request.
                logger.warning(f"Groq rejected request options ({e}); retrying with a simpler mode.")
                last_error = e
        raise LLMServiceError(f"Groq rejected every request variant: {last_error}")

    async def evaluate_technical_english(
        self,
        candidate_text: str,
        target_scenario: str,
        rag_context_items: List[Dict[str, Any]],
        asr_metrics: Optional[ASRMetrics] = None,
        start_time_monotonic: Optional[float] = None,
    ) -> EvaluationResponse:
        """Execute accelerated Groq LLM inference with Structured Outputs.

        Args:
            candidate_text: Text transcribed or submitted by the candidate.
            target_scenario: Scenario identifier (see ``TargetScenario``); legacy labels accepted.
            rag_context_items: Retrieved chunks from ChromaDB.
            asr_metrics: Optional telemetry from Whisper ASR.
            start_time_monotonic: Pipeline start timestamp for execution_time_ms calculation.

        Returns:
            Fully validated EvaluationResponse Pydantic v2 model.

        Raises:
            LLMServiceError: on network/API failures or unparsable model output.
        """
        if start_time_monotonic is None:
            start_time_monotonic = time.perf_counter()

        if not self.api_key:
            raise LLMServiceError("GROQ_API_KEY is not configured.")

        scenario_label = str(getattr(target_scenario, "value", target_scenario))
        scenario_desc = ""
        try:
            scenario_enum = TargetScenario(scenario_label)
            scenario_label = scenario_enum.value
            scenario_desc = SCENARIO_DESCRIPTIONS[scenario_enum]
        except ValueError:
            pass

        user_content = (
            f"TARGET SCENARIO: {scenario_label}\n{scenario_desc}\n\n"
            f"GROUND TRUTH TECHNICAL CONTEXT (AUTHORITATIVE KNOWLEDGE BASE):\n"
            f"{self._format_context(rag_context_items)}\n\n"
            f"SPEAKER TRANSCRIPT TO EVALUATE:\n\"{candidate_text}\"\n\n"
            "Perform the CEFR technical English evaluation and return the JSON object."
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        logger.info(f"Dispatching LLM inference to Groq ({self.model})...")
        try:
            raw_response = await self._complete(messages)
        except LLMServiceError:
            raise
        except Exception as e:
            logger.error(f"Groq request failed: {e}", exc_info=True)
            raise LLMServiceError(f"Groq request failed: {e}") from e

        logger.info(f"Groq LLM response received ({len(raw_response)} bytes).")
        try:
            parsed = parse_llm_json(raw_response)
        except Exception as e:
            logger.error(f"Unparsable LLM output: {raw_response[:300]!r}")
            raise LLMServiceError(f"LLM returned invalid JSON: {e}") from e

        # Accept both the flat schema and a nested "evaluation_core" object.
        core = parsed.get("evaluation_core") if isinstance(parsed.get("evaluation_core"), dict) else parsed
        terms = [str(t).strip() for t in (core.get("detected_technical_terms") or []) if str(t).strip()]

        feedback_details: List[FeedbackDetail] = []
        for item in parsed.get("feedback_details") or []:
            try:
                f_type = item.get("type", "vocabulary_enhancement")
                if f_type not in VALID_FEEDBACK_TYPES:
                    f_type = "vocabulary_enhancement"
                feedback_details.append(
                    FeedbackDetail(
                        type=f_type,
                        original_phrase=str(item.get("original_phrase", "")),
                        suggested_phrase=str(item.get("suggested_phrase", "")),
                        rag_source_context=str(
                            item.get("rag_source_context", "Technical Corpus Reference")
                        ),
                        explanation=str(item.get("explanation", "")),
                    )
                )
            except Exception as parse_err:
                logger.warning(f"Skipping malformed feedback item: {parse_err}")

        try:
            eval_core = EvaluationCore(
                overall_score=_clamp_score(core.get("overall_score")),
                cefr_level=_normalize_cefr(core.get("cefr_level")),
                # Deterministic backend computation - never delegated to the LLM.
                lexical_density_ratio=compute_lexical_density(terms, candidate_text),
                grammar_accuracy_score=_clamp_score(core.get("grammar_accuracy_score")),
                technical_precision_score=_clamp_score(core.get("technical_precision_score")),
                detected_technical_terms=terms,
            )
        except Exception as e:
            raise LLMServiceError(f"LLM evaluation failed schema validation: {e}") from e

        execution_time_ms = round((time.perf_counter() - start_time_monotonic) * 1000.0, 2)
        return EvaluationResponse(
            status="success",
            execution_time_ms=execution_time_ms,
            asr_metrics=asr_metrics,
            evaluation_core=eval_core,
            feedback_details=feedback_details,
        )

    async def is_ready(self) -> bool:
        """Verify connectivity to Groq API (performs a network call)."""
        try:
            models = await self.client.models.list()
            return len(models.data) > 0
        except Exception:
            return False


_llm_singleton: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Retrieve singleton instance of LLMService."""
    global _llm_singleton
    if _llm_singleton is None:
        _llm_singleton = LLMService()
    return _llm_singleton
