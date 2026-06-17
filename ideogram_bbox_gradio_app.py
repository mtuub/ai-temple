from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import inspect
import json
import re
import threading
import uuid
from pathlib import Path
from typing import Any

import gradio as gr
from gradio.processing_utils import move_files_to_cache


APP_DIR = Path(__file__).resolve().parent
SOURCE_HTML = APP_DIR / "ideogram_bbox_editor.html"
DEFAULT_OUTPUT_NID = 158
PIPELINE: Any | None = None
EDITOR_COMPONENT: gr.HTML | None = None
EXECUTOR = ThreadPoolExecutor(max_workers=2)
GENERATION_JOBS: dict[str, dict[str, Any]] = {}
GENERATION_JOBS_LOCK = threading.Lock()
ASPECT_RATIO_CHOICES = [
    "1:1 (Square)",
    "2:3 (Portrait Photo)",
    "3:2 (Photo)",
    "3:4 (Portrait Standard)",
    "4:3 (Standard)",
    "9:16 (Portrait Widescreen)",
    "16:9 (Widescreen)",
    "21:9 (Ultrawide)",
]


def _extract_between(text: str, start: str, end: str) -> str:
    match = re.search(re.escape(start) + r"(.*?)" + re.escape(end), text, re.S)
    if not match:
        raise RuntimeError(f"Could not extract {start!r} block from {SOURCE_HTML}")
    return match.group(1).strip()


def _extract_gradio_parts() -> tuple[str, str, str, dict]:
    source = SOURCE_HTML.read_text(encoding="utf-8")

    css = _extract_between(source, "<style>", "</style>")
    html = _extract_between(source, "<body>", "<script>")
    js = _extract_between(source, "<script>", "</script>")

    sample_match = re.search(
        r"const samplePrompt = (\{.*?\n    \});\n\n    const \$",
        js,
        re.S,
    )
    if not sample_match:
        raise RuntimeError("Could not extract sample prompt from editor JavaScript")
    sample_prompt = json.loads(sample_match.group(1))

    # Make styles component-scoped instead of page-scoped.
    css = css.replace(":root {", ".ideogram-app {")
    css = css.replace("\n    body {", "\n    .ideogram-app {")
    css = css.replace("min-height: 100vh;", "min-height: 760px;")
    css = css.replace("max-height: calc(100vh - 118px);", "max-height: 690px;")

    # Give the root node a component namespace for scoped CSS variables.
    html = html.replace(
        '<main class="app" id="editor">',
        '<main class="ideogram-app app" id="editor">',
        1,
    )

    # In gr.HTML, all DOM lookups must stay inside this component's root element.
    js = js.replace(
        "const $ = (id) => document.getElementById(id);",
        "const $ = (id) => element.querySelector(`#${id}`);",
    )

    # Push the current prompt into Gradio's component value whenever JSON refreshes.
    js = js.replace(
        """    function refreshJson() {
      $("jsonOut").value = JSON.stringify(buildPrompt(), null, 2);
    }
""",
        """    let gradioChangeTimer = null;

    function refreshJson(notify = true) {
      const prompt = buildPrompt();
      const promptText = JSON.stringify(prompt, null, 2);
      $("jsonOut").value = promptText;
      if (notify) {
        clearTimeout(gradioChangeTimer);
        gradioChangeTimer = setTimeout(() => trigger("change", buildEditorValue()), 120);
      }
    }
""",
    )
    js = js.replace(
        """    function refreshAll() {
      syncFieldsFromState();
      refreshLayers();
      draw();
      refreshJson();
    }
""",
        """    function refreshAll(options = {}) {
      const notify = options.notify !== false;
      syncFieldsFromState();
      refreshLayers();
      draw();
      refreshJson(notify);
    }
""",
    )
    js = js.replace(
        "        refreshAll();\n      } else {\n        state.drag.current = point;",
        "        refreshAll({ notify: false });\n      } else {\n        state.drag.current = point;",
    )
    js = js.replace(
        "    window.addEventListener(\"resize\", draw);\n    loadEditorValue(samplePrompt);",
        """    window.addEventListener("resize", draw);
    let initialValue = samplePrompt;
    try {
      if (typeof props.value === "string" && props.value.trim()) {
        initialValue = JSON.parse(props.value);
      } else if (props.value && typeof props.value === "object") {
        initialValue = props.value;
      }
    } catch (error) {
      initialValue = samplePrompt;
    }
    loadEditorValue(initialValue);""",
    )
    js = (
        "const startIdeogramEditor = () => {\n"
        "  if (!element.querySelector('#canvas')) {\n"
        "    setTimeout(startIdeogramEditor, 50);\n"
        "    return;\n"
        "  }\n"
        + js
        + "\n};\nstartIdeogramEditor();"
    )

    return html, css, js, sample_prompt


HTML_TEMPLATE, CSS_TEMPLATE, JS_ON_LOAD, SAMPLE_PROMPT = _extract_gradio_parts()


def make_editor_value(prompt: dict, generation: dict | None = None) -> str:
    value = {
        "__generation_settings": {
            "aspect_ratio": "1:1 (Square)",
            "megapixels": 1.0,
            "multiple": 8,
            "seed": -1,
            "output_nid": DEFAULT_OUTPUT_NID,
            "canvas_width": 1024,
            "canvas_height": 1024,
            **(generation or {}),
        },
        "__generation_result": None,
        **prompt,
    }
    return json.dumps(value, ensure_ascii=False, indent=2)


def parse_editor_value(value: dict | str | None) -> tuple[dict, dict]:
    if not value:
        return (
            SAMPLE_PROMPT,
            json.loads(make_editor_value(SAMPLE_PROMPT))["__generation_settings"],
        )

    if isinstance(value, str):
        try:
            data = json.loads(value)
        except json.JSONDecodeError as exc:
            raise gr.Error(f"Editor value is not valid JSON: {exc}") from exc
    else:
        data = value

    if isinstance(data, dict) and "prompt" in data:
        prompt = data.get("prompt") or SAMPLE_PROMPT
        generation = data.get("generation") or {}
    elif isinstance(data, dict) and "__generation_settings" in data:
        generation = data.get("__generation_settings") or {}
        prompt = {
            k: v
            for k, v in data.items()
            if k not in {"__generation_settings", "__generation_result"}
        }
    else:
        prompt = data
        generation = {}

    if isinstance(prompt, str):
        try:
            prompt = json.loads(prompt)
        except json.JSONDecodeError as exc:
            raise gr.Error(f"Prompt is not valid JSON: {exc}") from exc

    if not isinstance(prompt, dict):
        raise gr.Error("Prompt must be a JSON object.")

    defaults = json.loads(make_editor_value(SAMPLE_PROMPT))["__generation_settings"]
    generation = {**defaults, **generation}
    return prompt, generation


def pretty_prompt(value: dict | str | None) -> str:
    prompt, _ = parse_editor_value(value)
    return json.dumps(prompt, ensure_ascii=False, indent=2)


def pretty_prompt_event(evt: gr.EventData) -> str:
    return pretty_prompt(evt._data)


def set_pipeline(pipeline: Any) -> None:
    """Attach the notebook's ComfyUI/Ideogram pipeline object for generation."""
    global PIPELINE
    PIPELINE = pipeline


def _get_pipeline() -> Any:
    pipeline = PIPELINE or globals().get("p")
    if pipeline is None:
        raise RuntimeError(
            "Pipeline is not configured. In Colab, run: import "
            "ideogram_bbox_gradio_app as app; app.set_pipeline(p)"
        )
    return pipeline


def _cache_image_url(image_path: str) -> str:
    if EDITOR_COMPONENT is None:
        return image_path
    payload = {
        "path": str(image_path),
        "meta": {"_type": "gradio.FileData"},
    }
    cached = move_files_to_cache(
        payload,
        EDITOR_COMPONENT,
        postprocess=True,
        keep_in_cache=True,
    )
    return cached.get("url") or str(image_path)


def _execute_supports_progress_callback(execute_fn: Any) -> bool:
    try:
        signature = inspect.signature(execute_fn)
    except (TypeError, ValueError):
        return False
    return "progress_callback" in signature.parameters or any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )


def _set_generation_job(job_id: str, **updates: Any) -> dict:
    with GENERATION_JOBS_LOCK:
        job = GENERATION_JOBS.setdefault(job_id, {})
        job.update(updates)
        return dict(job)


def _get_generation_job(job_id: str) -> dict:
    with GENERATION_JOBS_LOCK:
        return dict(GENERATION_JOBS.get(job_id, {}))


def _make_progress_callback(job_id: str):
    def progress_callback(update: dict | None = None, **kwargs: Any) -> None:
        data = dict(update or {})
        data.update(kwargs)
        current = data.get("current", data.get("value", 0)) or 0
        total = data.get("total", data.get("max", 0)) or 0
        try:
            percent = round((float(current) / float(total)) * 100) if float(total) else 0
        except (TypeError, ValueError, ZeroDivisionError):
            percent = 0
        percent = max(0, min(99, percent))
        status = data.get("status") or (
            f"Progress: {current}/{total}" if total else "Running workflow..."
        )
        _set_generation_job(
            job_id,
            state="running",
            progress=percent,
            status=status,
            progress_current=current,
            progress_total=total,
        )

    return progress_callback


def generate_image_from_editor(editor_value: dict | str | None) -> dict:
    try:
        return _generate_image_from_editor(editor_value)
    except Exception as exc:
        return {
            "error": True,
            "status": str(exc),
            "image_url": "",
            "image_path": "",
        }


def start_generation_from_editor(editor_value: dict | str | None) -> dict:
    job_id = str(uuid.uuid4())
    _set_generation_job(
        job_id,
        state="queued",
        progress=0,
        status="Queued",
        result=None,
    )
    EXECUTOR.submit(_run_generation_job, job_id, editor_value)
    return {
        "job_id": job_id,
        "state": "queued",
        "progress": 0,
        "status": "Queued",
    }


def get_generation_status(job_id: str) -> dict:
    job = _get_generation_job(job_id)
    if not job:
        return {
            "job_id": job_id,
            "state": "missing",
            "progress": 100,
            "status": "Generation job was not found.",
            "result": {
                "error": True,
                "status": "Generation job was not found.",
                "image_url": "",
                "image_path": "",
            },
        }
    return {"job_id": job_id, **job}


def _run_generation_job(job_id: str, editor_value: dict | str | None) -> None:
    try:
        _set_generation_job(job_id, state="running", progress=3, status="Starting workflow...")
        result = _generate_image_from_editor(
            editor_value,
            progress_callback=_make_progress_callback(job_id),
        )
        _set_generation_job(
            job_id,
            state="done",
            progress=100,
            status=result.get("status", "Done"),
            result=result,
        )
    except Exception as exc:
        result = {
            "error": True,
            "status": str(exc),
            "image_url": "",
            "image_path": "",
        }
        _set_generation_job(
            job_id,
            state="error",
            progress=100,
            status=str(exc),
            result=result,
        )


def _generate_image_from_editor(
    editor_value: dict | str | None,
    progress_callback: Any | None = None,
) -> dict:
    pipeline = _get_pipeline()
    prompt, generation = parse_editor_value(editor_value)
    prompt_text = json.dumps(prompt, ensure_ascii=False, indent=2)
    if not prompt_text.strip():
        raise gr.Error("Prompt is empty.")

    aspect_ratio = generation["aspect_ratio"]
    megapixels_float = float(generation["megapixels"])
    multiple_int = int(generation.get("multiple", 8))
    seed_int = int(generation["seed"])
    if seed_int == -1:
        seed_int = int(pipeline.generate_random_seed())
    output_nid_int = int(generation["output_nid"])

    w_updates = {
        "37": {
            "aspect_ratio": aspect_ratio,
            "megapixels": megapixels_float,
            "multiple": multiple_int,
        },
        "98:24": {"text": prompt_text},
        "98:18": {"noise_seed": seed_int},
    }

    execute_kwargs = {
        "w_name": "image_ideogram4_t2i_api",
        "w_updates": w_updates,
    }
    if progress_callback and _execute_supports_progress_callback(pipeline.execute):
        execute_kwargs["progress_callback"] = progress_callback

    p_id = pipeline.execute(**execute_kwargs)

    image_path = pipeline.get_output_path(
        p_id=p_id,
        output_nid=output_nid_int,
        output_dir="output",
        type="image",
    )

    if not image_path:
        raise gr.Error("Generation finished, but no image path was returned.")

    image_url = _cache_image_url(str(image_path))
    return {
        "error": False,
        "seed": seed_int,
        "image_path": str(image_path),
        "image_url": image_url,
        "status": f"Done\nSeed: {seed_int}",
    }


APP_CSS = """
html,
body {
    margin: 0 !important;
    padding: 0 !important;
    background: #0f1117 !important;
}

.gradio-container {
    max-width: none !important;
    margin: 0 !important;
    padding: 0 !important;
    min-height: 100vh !important;
}

.gradio-container main,
.gradio-container .main,
.gradio-container .contain,
.gradio-container .wrap,
.gradio-container .column,
.gradio-container .block,
.gradio-container .html-container {
    max-width: none !important;
    margin: 0 !important;
    padding: 0 !important;
}

footer,
.footer,
#footer,
.api-docs,
button[aria-label="Settings"] {
    display: none !important;
}
"""


with gr.Blocks(
    title="Ideogram 4 Studio 🔥 (AI Temple)",
    fill_width=True,
) as demo:
    editor = gr.HTML(
        value="ready",
        html_template=HTML_TEMPLATE,
        css_template=CSS_TEMPLATE,
        js_on_load=JS_ON_LOAD,
        server_functions=[
            generate_image_from_editor,
            start_generation_from_editor,
            get_generation_status,
        ],
        apply_default_css=False,
        container=False,
        padding=False,
    )
    EDITOR_COMPONENT = editor


if __name__ == "__main__":
    demo.launch(css=APP_CSS, server_port=8765, footer_links=[])
