# UI Sandbox

## Overview
Interactive web-based testing and demonstration sandbox built with Gradio, optionally mounted at `/ui` for voice and text evaluation experiments.

## Interface
- [gradio_app.py:create_gradio_demo]: Factory building Gradio `Blocks` UI with Voice Evaluation and Text Evaluation tabs.
- [gradio_app.py:process_voice_evaluation]: Event callback executing voice evaluation workflow from uploaded audio or microphone inputs.
- [gradio_app.py:process_text_evaluation]: Event callback executing text evaluation workflow from candidate responses.
- [gradio_app.py:format_feedback_markdown]: Helper formatting feedback items into Markdown cards.

## Invariants
- [gradio_app.py:gr]: Requires optional `gradio` dependency; safely returns `None` if gradio is not installed.
- [app.core.config:Settings.ENABLE_UI]: The UI is disabled by default and only mounted when `ENABLE_UI=true`.

## Side Effects
- Reads temporary uploaded audio files passed by Gradio and dispatches async tasks using a local event loop.

## Verification
- Run: `pytest`
