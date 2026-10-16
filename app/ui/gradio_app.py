"""Gradio Testing Interface (Audio & Text Sandbox).

Provides an interactive sandbox mounted inside FastAPI at /ui with tabs for:
1. Voice Evaluation (Whisper ASR -> ChromaDB RAG -> Groq Cloud LLM)
2. Text Evaluation (ChromaDB RAG -> Groq Cloud LLM)
"""

import asyncio
import os
import time
from typing import Any, Dict, Optional, Tuple
from app.core.logging import logger
from app.services.asr_service import get_asr_service
from app.services.llm_service import get_llm_service
from app.services.rag_service import get_rag_service

try:
    import gradio as gr
except ImportError:
    gr = None  # type: ignore


CUSTOM_CSS = """
/* Tech Tutor Engine Modern Sandbox Styling */
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
.header-badge {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    color: white;
    padding: 18px 24px;
    border-radius: 12px;
    margin-bottom: 24px;
    box-shadow: 0 4px 12px rgba(30, 58, 138, 0.25);
}
.feedback-card {
    background: #f8fafc;
    border-left: 4px solid #3b82f6;
    padding: 12px 16px;
    margin-bottom: 12px;
    border-radius: 4px;
}
"""


def format_feedback_markdown(response_dict: Dict[str, Any]) -> str:
    """Format structured feedback details into clean, readable Markdown."""
    feedback_list = response_dict.get("feedback_details", [])
    if not feedback_list:
        return "###  No corrections needed!\nYour response conforms cleanly to professional technical standards."

    md_lines = ["###  Specific Technical & Grammatical Feedback\n"]
    for idx, item in enumerate(feedback_list, 1):
        f_type = item.get("type", "Enhancement").replace("_", " ").title()
        orig = item.get("original_phrase", "")
        sugg = item.get("suggested_phrase", "")
        rag_src = item.get("rag_source_context", "")
        expl = item.get("explanation", "")

        icon = "💡" if "vocabulary" in item.get("type", "") else "✏️"

        md_lines.append(f"#### {idx}. {icon} {f_type}")
        md_lines.append(f"- **Original Phrase:** `{orig}`")
        md_lines.append(f"- **Suggested Upgrade:** **{sugg}**")
        md_lines.append(f"- **Pedagogical Explanation:** {expl}")
        if rag_src:
            md_lines.append(f"- **Grounding Context / Source:** _{rag_src}_\n")

    return "\n".join(md_lines)


def process_voice_evaluation(
    audio_path: Optional[str], target_scenario: str
) -> Tuple[str, float, float, float, str, str, Dict[str, Any]]:
    """Handler for voice evaluation in Gradio."""
    if not audio_path or not os.path.exists(audio_path):
        return (
            "Error: Por favor graba o sube un archivo de audio válido.",
            0.0,
            0.0,
            0.0,
            "",
            "### ⚠️ No audio provided.",
            {"status": "error", "message": "No audio file provided."},
        )

    start_time = time.perf_counter()

    try:
        # Read audio binary directly from file path provided by Gradio
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        # Step 1: In-memory ASR with faster-whisper
        asr_service = get_asr_service()
        asr_metrics = asr_service.transcribe_audio_stream(audio_bytes)
        transcript = asr_metrics.transcription

        # Step 2: Retrieve Top-3 RAG Context from ChromaDB
        rag_service = get_rag_service()
        rag_context = rag_service.query_corpus(transcript, top_k=3)

        # Step 3: Groq LLM Evaluation
        llm_service = get_llm_service()
        # Run async call synchronously in worker thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response_model = loop.run_until_complete(
                llm_service.evaluate_technical_english(
                    candidate_text=transcript,
                    target_scenario=target_scenario,
                    rag_context_items=rag_context,
                    asr_metrics=asr_metrics,
                    start_time_monotonic=start_time,
                )
            )
        finally:
            loop.close()

        resp_dict = response_model.model_dump()
        core = resp_dict.get("evaluation_core", {})

        overall = float(core.get("overall_score", 0.0))
        grammar = float(core.get("grammar_accuracy_score", 0.0))
        lex_density = float(core.get("lexical_density_ratio", 0.0))
        terms = ", ".join(core.get("detected_technical_terms", []))
        feedback_md = format_feedback_markdown(resp_dict)

        return (
            transcript,
            round(overall, 1),
            round(grammar, 1),
            round(lex_density, 3),
            terms,
            feedback_md,
            resp_dict,
        )

    except Exception as e:
        logger.error(f"Error in Gradio voice evaluation handler: {e}", exc_info=True)
        return (
            f"Error processing audio: {str(e)}",
            0.0,
            0.0,
            0.0,
            "",
            f"### ⚠️ Evaluation Error\n`{str(e)}`",
            {"status": "error", "error": str(e)},
        )


def process_text_evaluation(
    candidate_text: str, target_scenario: str
) -> Tuple[float, float, float, str, str, Dict[str, Any]]:
    """Handler for text evaluation in Gradio."""
    if not candidate_text or not candidate_text.strip():
        return (
            0.0,
            0.0,
            0.0,
            "",
            "### ⚠️ Please enter a technical English candidate response to evaluate.",
            {"status": "error", "message": "Empty candidate text."},
        )

    start_time = time.perf_counter()
    clean_text = candidate_text.strip()

    try:
        # Step 1: Retrieve Top-3 RAG Context from ChromaDB
        rag_service = get_rag_service()
        rag_context = rag_service.query_corpus(clean_text, top_k=3)

        # Step 2: Groq LLM Evaluation
        llm_service = get_llm_service()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response_model = loop.run_until_complete(
                llm_service.evaluate_technical_english(
                    candidate_text=clean_text,
                    target_scenario=target_scenario,
                    rag_context_items=rag_context,
                    asr_metrics=None,
                    start_time_monotonic=start_time,
                )
            )
        finally:
            loop.close()

        resp_dict = response_model.model_dump()
        core = resp_dict.get("evaluation_core", {})

        overall = float(core.get("overall_score", 0.0))
        grammar = float(core.get("grammar_accuracy_score", 0.0))
        lex_density = float(core.get("lexical_density_ratio", 0.0))
        terms = ", ".join(core.get("detected_technical_terms", []))
        feedback_md = format_feedback_markdown(resp_dict)

        return (
            round(overall, 1),
            round(grammar, 1),
            round(lex_density, 3),
            terms,
            feedback_md,
            resp_dict,
        )

    except Exception as e:
        logger.error(f"Error in Gradio text evaluation handler: {e}", exc_info=True)
        return (
            0.0,
            0.0,
            0.0,
            "",
            f"### ⚠️ Evaluation Error\n`{str(e)}`",
            {"status": "error", "error": str(e)},
        )


def create_gradio_demo() -> Any:
    """Construct the Gradio Blocks testing sandbox interface."""
    if gr is None:
        logger.warning("Gradio is not installed yet. Skipping UI initialization.")
        return None

    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="indigo",
        neutral_hue="slate",
    )

    with gr.Blocks(title="Tech Tutor Engine - Sandbox") as demo:
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")
        with gr.Row():
            gr.HTML(
                """
                <div class="header-badge">
                    <h1 style="margin: 0; font-size: 26px; font-weight: 700;">Tech Tutor Engine Sandbox</h1>
                    <p style="margin: 6px 0 0 0; opacity: 0.9; font-size: 15px;">
                        Asistente de IA Especializado en Inglés Técnico para Ingenieros en Computación
                        | ASR (faster-whisper) + RAG (ChromaDB) + Groq (gpt-oss-120b)
                    </p>
                </div>
                """
            )

        with gr.Tabs():
            # ==========================================
            # TAB 1: Voice Evaluation (ASR + RAG + LLM)
            # ==========================================
            with gr.TabItem("🎙️ Voice Evaluation (ASR + RAG + LLM)"):
                with gr.Row():
                    with gr.Column(scale=5):
                        audio_input = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath",
                            label="Entrada de Audio (Micrófono o Archivo .wav / .mp3 / .webm)",
                        )
                        scenario_voice = gr.Dropdown(
                            choices=["System Design", "Code Review", "Thesis Defense"],
                            value="System Design",
                            label="Escenario Técnico (Target Scenario)",
                        )
                        eval_voice_btn = gr.Button("🎙️ Evaluar Audio", variant="primary", size="lg")

                    with gr.Column(scale=7):
                        transcript_out = gr.Textbox(
                            label="Transcripción Fonética (faster-whisper CPU int8)",
                            lines=3,
                            interactive=False,
                        )
                        with gr.Row():
                            overall_score_v = gr.Number(label="Overall Score (0-100)")
                            grammar_score_v = gr.Number(label="Grammar Accuracy (0-100)")
                            lexical_v = gr.Number(label="Lexical Density (0.0-1.0)")
                        terms_v = gr.Textbox(label="Términos Técnicos Detectados", interactive=False)

                feedback_md_v = gr.Markdown(label="Mejoras y Recomendaciones Grounded (RAG)")
                json_v = gr.JSON(label="Respuesta Completa JSON (EvaluationResponse)")

                eval_voice_btn.click(
                    fn=process_voice_evaluation,
                    inputs=[audio_input, scenario_voice],
                    outputs=[
                        transcript_out,
                        overall_score_v,
                        grammar_score_v,
                        lexical_v,
                        terms_v,
                        feedback_md_v,
                        json_v,
                    ],
                )

            # ==========================================
            # TAB 2: Text Evaluation (RAG + LLM)
            # ==========================================
            with gr.TabItem("📝 Text Evaluation (RAG + LLM)"):
                with gr.Row():
                    with gr.Column(scale=5):
                        text_input = gr.Textbox(
                            label="Candidate Response (Inglés Técnico)",
                            lines=5,
                            placeholder=(
                                "Ejemplo: In our microservices architecture, we implemented an in-memory "
                                "caching layer using Redis to mitigate database read latency and guarantee idempotency."
                            ),
                        )
                        scenario_text = gr.Dropdown(
                            choices=["System Design", "Code Review", "Thesis Defense"],
                            value="System Design",
                            label="Escenario Técnico (Target Scenario)",
                        )
                        eval_text_btn = gr.Button("📝 Evaluar Texto", variant="primary", size="lg")

                    with gr.Column(scale=7):
                        with gr.Row():
                            overall_score_t = gr.Number(label="Overall Score (0-100)")
                            grammar_score_t = gr.Number(label="Grammar Accuracy (0-100)")
                            lexical_t = gr.Number(label="Lexical Density (0.0-1.0)")
                        terms_t = gr.Textbox(label="Términos Técnicos Detectados", interactive=False)

                feedback_md_t = gr.Markdown(label="Mejoras Terminológicas y Citas de Documentación Oficial")
                json_t = gr.JSON(label="Respuesta Completa JSON (EvaluationResponse)")

                eval_text_btn.click(
                    fn=process_text_evaluation,
                    inputs=[text_input, scenario_text],
                    outputs=[
                        overall_score_t,
                        grammar_score_t,
                        lexical_t,
                        terms_t,
                        feedback_md_t,
                        json_t,
                    ],
                )

        return demo
