import html as _html_module
import json
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image, ImageDraw, ImageFont

from utils import content_to_text, save_json


STEP_REFLECTION_SYSTEM_MESSAGE = """You are a step-level reasoning auditor for a multimodal agent.
Your job is to assess whether the agent's current reasoning step was consistent with prior memory and whether the action taken was correct.
Return only one valid JSON object. Do not wrap it in Markdown fences and do not include prose outside the JSON object.

Use this shape:
{
  "consistency_issues": ["..."],
  "action_assessment": "correct" | "partially_correct" | "incorrect",
  "key_corrections": ["..."],
  "guidance_for_memory": "..."
}

Rules:
- consistency_issues: list concrete mismatches between the agent's THOUGHT and what the previous memory states. Use an empty list if none.
- action_assessment: was the code/action reasonable and correct given the THOUGHT and prior memory? Mark "incorrect" only when there is clear evidence of a mistake; mark "partially_correct" when the direction is right but execution had issues.
- key_corrections: concrete things that should be fixed or flagged in the next memory update. Use an empty list if none.
- guidance_for_memory: one short sentence telling the memory organizer what to emphasize, correct, or discard.
- Never invent evidence. Only report what is visible in the inputs you receive.
- If there is no prior memory yet, only assess action correctness from the observation alone.
"""


MEMORY_EXTRACTION_SYSTEM_MESSAGE = """You are a dynamic memory extractor for a multimodal reasoning agent.
Your job is not to solve the task. Your job is to extract and reorganize only the visible state that will be useful for the next reasoning step.
Return only one valid JSON object. Do not wrap it in Markdown fences and do not include prose outside the JSON object.

Use this canonical memory shape:
{
  "title": "Dynamic Reasoning Memory",
  "stage": "...",
  "context": {
    "task": "...",
    "latest_observation": "...",
    "visible_reasoning": "...",
    "previous_memory": "..."
  },
  "evidence": [
    {
      "id": "evidence_1",
      "type": "image" | "file" | "text",
      "path": "...",
      "caption": "...",
      "notes": ["..."],
      "annotations": ["..."]
    }
  ],
  "notes": {
    "evidence_notes": ["..."],
    "corrections": ["..."],
    "open_issues": ["..."],
    "open_checks": ["..."]
  }
}

Rules:
- Keep the object compact and cumulative.
- Keep only useful visible evidence, corrections, and open checks.
- The "latest_observation" field must be a short observation note, not a full HTML page, not a full JSON object, and not a verbatim dump of raw markup.
- It may mention key status, image findings, errors, or execution results in concise natural language.
- Only keep uncertainty when it materially affects the answer or blocks the next verification step.
- Never write the final answer.
- Never invent hidden reasoning, coordinates, or visual details.
- If generated image files are available, prefer retaining them as evidence entries with nearby local notes.
- If an annotated image truly exists, keep it as an image evidence entry; otherwise keep the note textual.
- If "model_reflection" is non-empty, treat it as the model's self-assessment of the previous step. If it signals a mistake or misleading prior memory, record the correction in notes.corrections and update the relevant context fields accordingly.
- Omit empty fields whenever possible.
"""


GATED_SYMBOLIC_TASKS = {
    "graph_connectivity",
    "graph_maxflow",
    "graph_isomorphism",
    "math_breakpoint",
    "math_convexity",
    "math_parity",
}


# ---------------------------------------------------------------------------
# Text / HTML utilities
# ---------------------------------------------------------------------------

def _truncate(text: str, limit: int = 8000) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[: limit // 2] + "\n...[truncated]...\n" + text[-limit // 2 :]


def _strip_fences(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:html|json|text)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


def _extract_summary(text: str, limit: int = 700) -> str:
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean[:limit]


def _summarize_observation_text(text: str, limit: int = 1200) -> str:
    text = _strip_fences(text or "")
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<script[\s\S]*?</script>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<!doctype[^>]*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"</?(html|head|body|main|section|article|div)[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<img\s+[^>]*src=['\"]([^'\"]+)['\"][^>]*>", r"\n[image: \1]\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<pre[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</pre>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return _truncate(text.strip(), limit)


def _html_escape(value: Any) -> str:
    return _html_module.escape(str(value))


def _render_observation_block(text: str) -> str:
    summary = _summarize_observation_text(text)
    if not summary:
        return ""
    blocks = []
    bullet_items = []
    for raw_line in summary.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        normalized = line.lstrip("-* ").strip()
        if raw_line.strip().startswith(("-", "*")):
            bullet_items.append(f"<li>{_html_escape(normalized)}</li>")
            continue
        if bullet_items:
            blocks.append("<ul>" + "".join(bullet_items) + "</ul>")
            bullet_items = []
        lowered = line.lower()
        if lowered in {"success", "error"} or lowered.startswith("status:"):
            status_text = line.split(":", 1)[-1].strip() if ":" in line else line
            tone = "obs-success" if "success" in lowered else "obs-error" if "error" in lowered else "obs-neutral"
            blocks.append(
                f"<p><strong>Observation</strong> "
                f"<span class=\"obs-pill {tone}\">{_html_escape(status_text)}</span></p>"
            )
        else:
            blocks.append(f"<p>{_html_escape(line)}</p>")
    if bullet_items:
        blocks.append("<ul>" + "".join(bullet_items) + "</ul>")
    return "".join(blocks)


def _safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(text)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _extract_image_paths_from_json_payload(payload: Dict[str, Any]) -> List[str]:
    image_paths: List[str] = []
    for item in payload.get("evidence", []):
        if not isinstance(item, dict):
            continue
        if item.get("type") != "image":
            continue
        path = item.get("path")
        if path and path not in image_paths:
            image_paths.append(str(path))
    return image_paths


def _render_prompt_image_gallery(image_paths: Iterable[str], heading: str) -> str:
    unique_paths: List[str] = []
    for path in image_paths or []:
        if path and path not in unique_paths:
            unique_paths.append(str(path))
    if not unique_paths:
        return ""
    parts = [f"\n\n{heading}\n"]
    for idx, path in enumerate(unique_paths, start=1):
        escaped = _html_escape(path)
        parts.append(
            f"Retained image {idx}: <code>{escaped}</code>\n"
            f'<img src="{escaped}" alt="retained image {idx}">\n'
        )
    return "".join(parts)


# ---------------------------------------------------------------------------
# Canonical payload helpers
# ---------------------------------------------------------------------------

def _default_memory_payload(
    task_prompt: str,
    previous_summary: str,
    assistant_text: str,
    observation_text: str,
    generated_files: Iterable[str],
    stage: str,
) -> Dict[str, Any]:
    return {
        "title": "Dynamic Reasoning Memory",
        "stage": stage,
        "context": {
            "task": _truncate(task_prompt, 1200),
            "previous_memory": previous_summary,
            "visible_reasoning": _truncate(assistant_text, 1800),
            "latest_observation": _summarize_observation_text(observation_text, 1200),
        },
        "evidence": [
            {
                "id": f"evidence_{idx}",
                "type": "image" if str(path).lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg")) else "file",
                "path": str(path),
                "caption": "Generated or referenced evidence retained for the next reasoning step.",
                "notes": [],
                "annotations": [],
            }
            for idx, path in enumerate(list(generated_files or []), start=1)
        ],
        "notes": {},
    }


def _compact_canonical_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "title": payload.get("title", "Dynamic Reasoning Memory"),
        "stage": payload.get("stage", ""),
    }

    context = {}
    for key in ("task", "latest_observation", "visible_reasoning", "previous_memory"):
        value = ((payload.get("context") or {}) if isinstance(payload.get("context"), dict) else {}).get(key)
        if value:
            if key == "latest_observation":
                value = _summarize_observation_text(str(value), 1200)
            context[key] = value
    if context:
        result["context"] = context

    evidence_items = []
    for item in payload.get("evidence", []):
        if not isinstance(item, dict):
            continue
        cleaned = {}
        for key in ("id", "type", "path", "caption"):
            if item.get(key):
                cleaned[key] = item.get(key)
        for key in ("notes", "annotations"):
            values = item.get(key)
            if isinstance(values, list) and values:
                cleaned[key] = values
        if cleaned:
            evidence_items.append(cleaned)
    if evidence_items:
        result["evidence"] = evidence_items

    notes = {}
    payload_notes = payload.get("notes") if isinstance(payload.get("notes"), dict) else {}
    for key in ("evidence_notes", "corrections", "open_issues", "open_checks"):
        values = payload_notes.get(key)
        if isinstance(values, list) and values:
            notes[key] = values
    if notes:
        result["notes"] = notes
    return result


def _payload_to_json(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _compact_canonical_payload(payload)


def _payload_to_html(payload: Dict[str, Any]) -> str:
    payload = _compact_canonical_payload(payload)

    evidence_items = []
    for idx, item in enumerate(payload.get("evidence", []), start=1):
        if not isinstance(item, dict):
            continue
        path = item.get("path", "")
        escaped = _html_escape(path)
        caption = _html_escape(item.get("caption", f"Evidence {idx}"))
        notes = "".join(f"<li>{_html_escape(note)}</li>" for note in item.get("notes", []) if note)
        annotations = "".join(
            f"<li><strong>Annotation:</strong> {_html_escape(note)}</li>"
            for note in item.get("annotations", [])
            if note
        )
        note_block = f"<ul>{notes}{annotations}</ul>" if (notes or annotations) else ""
        if item.get("type") == "image" and path:
            evidence_items.append(
                "<figure class='evidence-figure'>"
                f"<div class='image-wrap'><img src='{escaped}' alt='evidence image {idx}'>"
                f"<span class='badge'>{idx}</span></div>"
                f"<figcaption><strong>{caption}</strong>"
                + (f" <code>{escaped}</code>" if path else "")
                + f"{note_block}</figcaption>"
                "</figure>"
            )
        elif path:
            evidence_items.append(
                "<div class='evidence-card'>"
                f"<p><strong>{caption}</strong> <code>{escaped}</code></p>"
                f"{note_block}"
                "</div>"
            )
        elif caption or note_block:
            evidence_items.append(
                "<div class='evidence-card'>"
                f"<p><strong>{caption}</strong></p>{note_block}"
                "</div>"
            )

    context = payload.get("context", {}) if isinstance(payload.get("context"), dict) else {}
    overview_parts = []
    if context.get("task"):
        overview_parts.append(
            f"<div><strong>Task</strong><pre>{_html_escape(context.get('task', ''))}</pre></div>"
        )
    if context.get("latest_observation"):
        overview_parts.append(
            "<div class='observation-card'><strong>Latest Observation</strong>"
            f"{_render_observation_block(context.get('latest_observation', ''))}"
            "</div>"
        )
    if context.get("visible_reasoning"):
        overview_parts.append(
            f"<details><summary>Visible Reasoning Snapshot</summary>"
            f"<pre>{_html_escape(context.get('visible_reasoning', ''))}</pre></details>"
        )
    if context.get("previous_memory"):
        overview_parts.append(
            f"<details><summary>Previous Memory Snapshot</summary>"
            f"<pre>{_html_escape(context.get('previous_memory', ''))}</pre></details>"
        )

    payload_notes = payload.get("notes", {}) if isinstance(payload.get("notes"), dict) else {}
    note_items = []
    for item in payload_notes.get("evidence_notes", []):
        note_items.append(f"<li>{_html_escape(item)}</li>")
    for item in payload_notes.get("corrections", []):
        note_items.append(f"<li><strong>Correction:</strong> {_html_escape(item)}</li>")
    for item in payload_notes.get("open_issues", []):
        note_items.append(f"<li><strong>Open issue:</strong> {_html_escape(item)}</li>")
    for item in payload_notes.get("open_checks", []):
        note_items.append(f"<li><strong>Open check:</strong> {_html_escape(item)}</li>")

    evidence_html = "".join(evidence_items) or "<p>No retained image evidence.</p>"
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{_html_escape(payload.get("title", "Dynamic Reasoning Memory"))}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; line-height: 1.45; }}
    section {{ border-top: 1px solid #d0d7de; padding: 12px 0; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; }}
    figure {{ display: inline-block; max-width: 46%; margin: 8px 12px 8px 0; vertical-align: top; }}
    .image-wrap {{ position: relative; display: inline-block; }}
    img {{ max-width: 100%; border: 1px solid #d0d7de; }}
    .badge {{ position: absolute; top: 6px; left: 6px; background: #0b5fff; color: white;
              border-radius: 999px; padding: 2px 7px; font-weight: bold; }}
    .callouts {{ background: #f6f8fa; border-left: 4px solid #0b5fff; padding: 8px 12px; }}
    .artifact-note {{ background: #fff8db; border-left: 4px solid #b26a00; padding: 10px 12px; margin: 12px 0 18px; }}
    .artifact-note p {{ margin: 6px 0; }}
    .observation-card {{ background: #f7fbff; border: 1px solid #cfe3ff;
                         border-left: 4px solid #0b5fff; padding: 10px 12px; border-radius: 8px; }}
    .observation-card p, .observation-card ul {{ margin: 8px 0; }}
    .obs-pill {{ display: inline-block; padding: 2px 8px; border-radius: 999px;
                 font-size: 12px; font-weight: bold; }}
    .obs-success {{ background: #dcfce7; color: #166534; }}
    .obs-error {{ background: #fee2e2; color: #991b1b; }}
    .obs-neutral {{ background: #e5e7eb; color: #374151; }}
    figcaption, code {{ overflow-wrap: anywhere; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>{_html_escape(payload.get("title", "Dynamic Reasoning Memory"))}</h1>
  <p><strong>Stage:</strong> {_html_escape(payload.get("stage", ""))}</p>
  <aside class="artifact-note">
    <p><strong>Use this as evidence memory, not as an answer sheet.</strong></p>
    <p>Preserve image evidence, local notes, corrections, and unresolved checks.
       Keep uncertainty only when it materially affects the answer;
       otherwise prefer concrete supported evidence.</p>
  </aside>
  {"<section><h2>Overview</h2>" + ''.join(overview_parts) + "</section>" if overview_parts else ""}
  <section><h2>Evidence</h2>{evidence_html}</section>
  {"<section><h2>Working Notes</h2><ul>" + ''.join(note_items) + "</ul></section>" if note_items else ""}
  <section class="callouts">
    <p>Keep evidence local: source image, zoomed image, and any real annotated image can be shown
       side by side with short captions. If no rendered overlay exists, describe the target region
       or relation in nearby text instead of inventing a fake annotation.</p>
  </section>
</body>
</html>"""


def _render_image_memory(payload: Dict[str, Any], path: Path) -> None:
    """Render the canonical memory payload as a PIL image (for image format)."""
    width, height = 1400, 1800
    margin = 40
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    try:
        title_font = ImageFont.truetype("Arial.ttf", 34)
        header_font = ImageFont.truetype("Arial.ttf", 24)
        body_font = ImageFont.truetype("Arial.ttf", 18)
    except Exception:
        title_font = header_font = body_font = ImageFont.load_default()

    payload = _compact_canonical_payload(payload)
    context = payload.get("context", {}) or {}
    notes = payload.get("notes", {}) or {}

    y = margin
    draw.text((margin, y), "Dynamic Reasoning Memory", fill=(20, 32, 44), font=title_font)
    y += 54
    blocks = [
        ("Stage", payload.get("stage", "")),
        ("Task", context.get("task", "")),
        ("Previous Memory", context.get("previous_memory", "")),
        ("Visible Reasoning", context.get("visible_reasoning", "")),
        ("Latest Observation", context.get("latest_observation", "")),
        ("Evidence Files", "\n".join(
            item.get("path", "") for item in payload.get("evidence", []) if isinstance(item, dict)
        )),
        ("Open Issues", "\n".join(map(str, notes.get("open_issues", [])))),
        ("Corrections", "\n".join(map(str, notes.get("corrections", [])))),
    ]
    for header, value in blocks:
        if y > height - 120:
            break
        draw.text((margin, y), header, fill=(15, 81, 112), font=header_font)
        y += 32
        wrapped_lines: List[str] = []
        for line in str(value).splitlines() or [""]:
            wrapped_lines.extend(textwrap.wrap(line, width=120) or [""])
        for line in wrapped_lines[:24]:
            if y > height - 60:
                break
            draw.text((margin, y), line, fill=(31, 41, 55), font=body_font)
            y += 24
        y += 18
    img.save(path)


# ---------------------------------------------------------------------------
# MemoryOrganizerAgent
# ---------------------------------------------------------------------------

class MemoryOrganizerAgent:
    def __init__(
        self,
        memory_format: Optional[str],
        llm_client: Optional[object],
        working_dir: str,
        task_name: Optional[str] = None,
        path_aliases: Optional[Dict[str, str]] = None,
    ) -> None:
        self.memory_format = memory_format
        self.llm_client = llm_client
        self.working_dir = Path(working_dir)
        self.task_name = task_name
        self.path_aliases = {
            str(src): str(dst)
            for src, dst in (path_aliases or {}).items()
            if src and dst
        }
        self.step = 0
        self.previous_text = ""
        self.previous_summary = ""
        self.previous_canonical_text = ""
        self.records: List[Dict[str, Any]] = []

    @property
    def enabled(self) -> bool:
        return self.memory_format in {"html", "json", "image"}

    def _call_llm(self, prompt: str, system_message: str = MEMORY_EXTRACTION_SYSTEM_MESSAGE) -> Optional[str]:
        if self.llm_client is None:
            return None
        try:
            response = self.llm_client.create(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ]
            )
            extracted = self.llm_client.extract_text_or_completion_object(response)[0]
            return extracted if isinstance(extracted, str) else str(extracted)
        except Exception as exc:
            return f"MEMORY_AGENT_ERROR: {exc}"

    def reflect_on_step(
        self,
        assistant_message: Any,
        observation_text: str,
    ) -> str:
        """Call LLM to audit consistency of THOUGHT+ACTION vs prior memory and assess action correctness.

        Returns a JSON string passed as reflection_text into update().
        Returns empty string when memory is disabled, LLM is unavailable, or this is the first step
        (no prior memory to check consistency against).
        """
        if not self.enabled or self.llm_client is None:
            return ""
        # Skip on the first step: no prior memory means the consistency check has nothing to anchor on.
        if not self.previous_canonical_text:
            return ""

        # For HTML format, use the rendered HTML (includes image tags for multimodal context).
        # For JSON/image formats, use the compact canonical JSON.
        if self.memory_format == "html" and self.previous_text:
            prior_memory_block = _truncate(self.previous_text, 3000)
            prior_memory_label = "Prior memory (HTML artifact from the previous step)"
        else:
            prior_memory_block = _truncate(self.previous_canonical_text, 3000)
            prior_memory_label = "Prior memory (canonical JSON from the previous step)"

        prompt = f"""Audit the agent's current reasoning step for consistency and correctness.

{prior_memory_label}:
{prior_memory_block}

Agent's current THOUGHT and ACTION (this step's assistant output):
{_truncate(content_to_text(assistant_message), 4000)}

Execution observation (result of running the ACTION):
{_truncate(_summarize_observation_text(observation_text), 2000)}

Return a JSON object with the fields: consistency_issues, action_assessment, key_corrections, guidance_for_memory.
"""
        raw = self._call_llm(prompt, system_message=STEP_REFLECTION_SYSTEM_MESSAGE)
        if raw is None or raw.startswith("MEMORY_AGENT_ERROR:"):
            return ""
        return _strip_fences(raw)

    def _normalize_path(self, path: str) -> str:
        return self.path_aliases.get(str(path), str(path))

    def _rewrite_artifact_paths(self, artifact_text: str) -> str:
        for src, dst in self.path_aliases.items():
            artifact_text = artifact_text.replace(src, dst)
        return artifact_text

    def _html_img_tag(self, path: str, idx: int) -> str:
        escaped = _html_escape(path)
        return f'<img src="{escaped}" alt="generated evidence image {idx}">'

    def _inject_generated_images(self, artifact_text: str, generated_files: Iterable[str]) -> str:
        image_files = [
            str(path)
            for path in generated_files
            if str(path).lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"))
        ]
        if not image_files:
            return artifact_text

        image_iter = iter(enumerate(image_files, start=1))

        def replace_placeholder(_match):
            try:
                idx, path = next(image_iter)
            except StopIteration:
                idx, path = len(image_files), image_files[-1]
            return self._html_img_tag(path, idx)

        artifact_text = re.sub(
            r"<image\s*/?>",
            replace_placeholder,
            artifact_text,
            flags=re.IGNORECASE,
        )

        if "<img" not in artifact_text.lower():
            gallery = "\n<section><h2>Generated Image Evidence</h2>" + "".join(
                f"<figure>{self._html_img_tag(path, idx)}"
                f"<figcaption>Generated evidence image {idx}: "
                f"<code>{_html_escape(path)}</code></figcaption></figure>"
                for idx, path in enumerate(image_files, start=1)
            ) + "</section>\n"
            if "</body>" in artifact_text.lower():
                artifact_text = re.sub(r"</body>", gallery + "</body>", artifact_text, flags=re.IGNORECASE)
            else:
                artifact_text += gallery
        return artifact_text

    def _build_extraction_prompt(
        self,
        task_prompt: str,
        assistant_message: Any,
        observation_text: str,
        generated_files: Iterable[str],
        stage: str,
        reflection_text: str = "",
    ) -> str:
        return f"""Extract one compact canonical memory JSON object.

Memory revision rules:
- Preserve useful previous memory inside the new artifact itself, but remove irrelevant content.
- Correct stale or wrong previous-memory claims when the latest observation contradicts them.
- Keep corrected claims only when supported.
- Keep discarded or corrected claims in notes.corrections.
- Keep unresolved contradictions or missing evidence in notes.open_issues.
- Do not answer the task independently; extract visible state for the reasoner.
- The returned canonical object should be sufficient to render either HTML or JSON memory.
- Prefer local evidence notes over global prose. Keep uncertainty sparse.
- For "latest_observation", write only a compact note about the newest execution result or visual finding. Do not copy a whole HTML page, JSON blob, CSS block, or raw markup dump into that field.
- If "model_reflection" is non-empty, treat it as the model's self-assessment of the previous step. If it signals a mistake or misleading prior memory, record the correction in notes.corrections and revise the relevant context fields.

Task prompt:
{_truncate(task_prompt, 5000)}

Previous memory summary:
{_truncate(self.previous_summary, 2000)}

Previous canonical memory excerpt:
{_truncate(self.previous_canonical_text, 3500)}

Latest visible assistant message:
{_truncate(content_to_text(assistant_message), 5000)}

Latest observation / feedback:
{_truncate(observation_text, 5000)}

Generated or referenced files from latest execution:
{json.dumps(list(generated_files or []), ensure_ascii=False)}

Image path aliases that must be used when retaining images:
{json.dumps(self.path_aliases, ensure_ascii=False)}

Model reflection on this step (REFLECTION section from the assistant's output):
{reflection_text or "(none)"}
"""

    def update(
        self,
        task_prompt: str,
        assistant_message: Any,
        observation_text: str,
        generated_files: Optional[Iterable[str]] = None,
        stage: str = "execution",
        reflection_text: str = "",
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        generated_files = [self._normalize_path(p) for p in (generated_files or [])]
        fallback_payload = _default_memory_payload(
            task_prompt=task_prompt,
            previous_summary=self.previous_summary,
            assistant_text=content_to_text(assistant_message),
            observation_text=observation_text,
            generated_files=generated_files,
            stage=stage,
        )
        raw = self._call_llm(
            self._build_extraction_prompt(
                task_prompt, assistant_message, observation_text,
                generated_files, stage, reflection_text,
            )
        )
        if raw is None or raw.startswith("MEMORY_AGENT_ERROR:"):
            canonical_payload = _payload_to_json(fallback_payload)
            error = raw
        else:
            parsed = _safe_json_loads(_strip_fences(raw))
            canonical_payload = _payload_to_json(parsed or fallback_payload)
            error = None

        canonical_text = json.dumps(canonical_payload, indent=2, ensure_ascii=False)

        if self.memory_format == "html":
            artifact_text = _payload_to_html(canonical_payload)
            artifact_text = self._rewrite_artifact_paths(artifact_text)
            artifact_text = self._inject_generated_images(artifact_text, generated_files)
            if "<" not in artifact_text or ">" not in artifact_text:
                artifact_text = _payload_to_html(canonical_payload)
            elif "<html" not in artifact_text.lower() or "<body" not in artifact_text.lower():
                artifact_text = (
                    '<!doctype html>\n<html>\n<head><meta charset="utf-8">'
                    '<title>Dynamic Reasoning Memory</title></head>\n'
                    f'<body>\n{artifact_text}\n</body>\n</html>'
                )
            path = self.working_dir / f"memory_step_{self.step:03d}.html"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(artifact_text, encoding="utf-8")
            summary = _extract_summary(artifact_text)
            prompt_payload = artifact_text

        elif self.memory_format == "json":
            path = self.working_dir / f"memory_step_{self.step:03d}.json"
            artifact_text = json.dumps(canonical_payload, indent=2, ensure_ascii=False)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(artifact_text, encoding="utf-8")
            summary = _extract_summary(artifact_text)
            prompt_payload = artifact_text

        else:  # image
            path = self.working_dir / f"memory_step_{self.step:03d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            _render_image_memory(canonical_payload, path)
            artifact_text = canonical_text
            summary = _extract_summary(canonical_text)
            prompt_payload = f"<img src='{path.name}'>\n{canonical_text}"

        record = {
            "step": self.step,
            "format": self.memory_format,
            "path": str(path),
            "summary": summary,
            "stage": stage,
            "error": error,
        }
        self.records.append(record)
        self.previous_text = artifact_text
        self.previous_canonical_text = canonical_text
        self.previous_summary = summary
        self.step += 1
        save_json(self.working_dir / "memory_index.json", self.records)
        return {**record, "prompt_payload": prompt_payload}

    def _previous_memory_has_open_issue(self) -> bool:
        if not self.previous_text:
            return False
        lowered = self.previous_text.lower()
        if "open_issues" not in lowered and "open issues" not in lowered:
            return False
        return not any(
            marker in lowered
            for marker in [
                '"open_issues": []',
                "<h2>open issues</h2><ul></ul>",
                "<h2>open issues</h2><pre></pre>",
            ]
        )

    def should_inject_full(
        self,
        record: Optional[Dict[str, Any]],
        generated_files=None,
        stage: str = "",
    ) -> bool:
        if not record:
            return False
        if self.task_name not in GATED_SYMBOLIC_TASKS:
            return True
        generated_files = list(generated_files or [])
        if generated_files:
            return True
        if "error" in (stage or ""):
            return True
        if self._previous_memory_has_open_issue():
            return True
        return int(record.get("step", 0)) > 0

    def format_feedback(self, record: Optional[Dict[str, Any]], full: bool = True) -> str:
        if not record:
            return ""
        if not full:
            return (
                "\n\nDYNAMIC MEMORY SUMMARY:\n"
                f"Memory format: {record['format']}\n"
                f"Memory path: {Path(record['path']).name}\n"
                f"Memory summary: {record['summary']}\n"
                "Full memory was not injected for this simple symbolic/math/graph step to avoid "
                "unnecessary context noise. If the task continues, if an error occurs, if evidence "
                "files are generated, or if open issues remain, the full memory will be provided.\n"
            )
        payload = _truncate(record.get("prompt_payload", ""), 10000)
        image_gallery = ""
        if record.get("format") == "json":
            parsed = _safe_json_loads(record.get("prompt_payload", "") or "")
            image_gallery = _render_prompt_image_gallery(
                _extract_image_paths_from_json_payload(parsed or {}),
                heading="Images:",
            )
        return (
            "\n\nDYNAMIC MEMORY UPDATE:\n"
            f"Memory format: {record['format']}\n"
            f"Memory path: {Path(record['path']).name}\n"
            "Use this external memory as mutable state for the next step. It is not a final answer.\n"
            "Before the next action, check the dynamic memory for retained evidence, "
            "open issues, and corrections.\n"
            "Memory artifact content follows:\n"
            f"{payload}\n"
            f"{image_gallery}"
        )
