import json
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image, ImageDraw, ImageFont

from utils import content_to_text, save_json


MEMORY_SYSTEM_MESSAGE = """You are a dynamic memory organizer for a multimodal reasoning agent.
Your job is to maintain an external, inspectable memory artifact from the visible task context, visible reasoning summaries, code actions, execution observations, generated files, and known errors.
Do not solve the task independently and do not invent hidden chain-of-thought. Record concise, useful state only: evidence, variables, objects, relations, derived facts, uncertainties, revisions, and next-action hints.
Preserve useful prior memory, correct stale or wrong claims, and remove irrelevant content.
If the latest observation or visible assistant message contradicts the previous memory, explicitly revise the memory: move the old claim into revisions, replace it with the corrected state, and keep unresolved contradictions in open_issues rather than treating them as facts.
Only promote a derived fact when it is supported by execution output, explicit symbolic calculation, or visible evidence. Mark uncertain claims as tentative.
"""


GATED_SYMBOLIC_TASKS = {
    "graph_connectivity",
    "graph_maxflow",
    "graph_isomorphism",
    "math_breakpoint",
    "math_convexity",
    "math_parity",
}


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


def _safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(text)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


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
        "task_summary": _truncate(task_prompt, 1200),
        "previous_memory_summary": previous_summary,
        "visible_reasoning_summary": _truncate(assistant_text, 1800),
        "latest_observation": _truncate(observation_text, 1800),
        "evidence_files": list(generated_files or []),
        "retained_state": [],
        "revisions": [],
        "open_issues": [],
        "next_action_hints": [],
    }


def _payload_to_html(payload: Dict[str, Any]) -> str:
    def esc(value: Any) -> str:
        import html

        return html.escape(str(value))

    evidence = "\n".join(f"<li><code>{esc(path)}</code></li>" for path in payload.get("evidence_files", []))
    sections = [
        ("Task", payload.get("task_summary", "")),
        ("Previous Memory", payload.get("previous_memory_summary", "")),
        ("Visible Reasoning", payload.get("visible_reasoning_summary", "")),
        ("Latest Observation", payload.get("latest_observation", "")),
        ("Retained State", payload.get("retained_state", [])),
        ("Revisions", payload.get("revisions", [])),
        ("Open Issues", payload.get("open_issues", [])),
        ("Next Action Hints", payload.get("next_action_hints", [])),
    ]
    body = []
    for title, value in sections:
        if isinstance(value, list):
            content = "<ul>" + "".join(f"<li>{esc(item)}</li>" for item in value) + "</ul>"
        else:
            content = f"<pre>{esc(value)}</pre>"
        body.append(f"<section><h2>{esc(title)}</h2>{content}</section>")
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{esc(payload.get("title", "Dynamic Reasoning Memory"))}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}
    section {{ border-top: 1px solid #d0d7de; padding: 12px 0; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; }}
    code {{ overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <h1>{esc(payload.get("title", "Dynamic Reasoning Memory"))}</h1>
  <p><strong>Stage:</strong> {esc(payload.get("stage", ""))}</p>
  {''.join(body)}
  <section><h2>Evidence Files</h2><ul>{evidence}</ul></section>
</body>
</html>"""


def _render_image_memory(payload: Dict[str, Any], path: Path) -> None:
    width, height = 1400, 1800
    margin = 40
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("Arial.ttf", 34)
        header_font = ImageFont.truetype("Arial.ttf", 24)
        body_font = ImageFont.truetype("Arial.ttf", 18)
    except Exception:
        title_font = header_font = body_font = ImageFont.load_default()

    y = margin
    draw.text((margin, y), "Dynamic Reasoning Memory", fill=(20, 32, 44), font=title_font)
    y += 54
    blocks = [
        ("Stage", payload.get("stage", "")),
        ("Task", payload.get("task_summary", "")),
        ("Previous Memory", payload.get("previous_memory_summary", "")),
        ("Visible Reasoning", payload.get("visible_reasoning_summary", "")),
        ("Latest Observation", payload.get("latest_observation", "")),
        ("Evidence Files", "\n".join(payload.get("evidence_files", []))),
        ("Open Issues", "\n".join(map(str, payload.get("open_issues", [])))),
        ("Next Action Hints", "\n".join(map(str, payload.get("next_action_hints", [])))),
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
    image.save(path)


class MemoryOrganizerAgent:
    def __init__(
        self,
        memory_format: Optional[str],
        llm_client: Optional[object],
        working_dir: str,
        task_name: Optional[str] = None,
    ) -> None:
        self.memory_format = memory_format
        self.llm_client = llm_client
        self.working_dir = Path(working_dir)
        self.task_name = task_name
        self.step = 0
        self.previous_text = ""
        self.previous_summary = ""
        self.records: List[Dict[str, Any]] = []

    @property
    def enabled(self) -> bool:
        return self.memory_format in {"html", "json", "image"}

    def _call_llm(self, prompt: str) -> Optional[str]:
        if self.llm_client is None:
            return None
        try:
            response = self.llm_client.create(
                messages=[
                    {"role": "system", "content": MEMORY_SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ]
            )
            extracted = self.llm_client.extract_text_or_completion_object(response)[0]
            return extracted if isinstance(extracted, str) else str(extracted)
        except Exception as exc:
            return f"MEMORY_AGENT_ERROR: {exc}"

    def _build_prompt(
        self,
        task_prompt: str,
        assistant_message: Any,
        observation_text: str,
        generated_files: Iterable[str],
        stage: str,
    ) -> str:
        artifact_contract = {
            "html": (
                "Return only one complete UTF-8 HTML document. It must include html/head/body tags and visible sections "
                "for Task, Evidence, Objects/Variables, Relations, Derived Facts, Revisions, Open Issues, and Next Action Hints. "
                "Use tables, lists, image references, and concise captions when useful."
            ),
            "json": (
                "Return only one valid JSON object with keys: title, stage, task_summary, evidence_files, objects_variables, "
                "relations_constraints, derived_facts, retained_state, revisions, open_issues, next_action_hints."
            ),
            "image": (
                "Return only one valid JSON object with concise fields suitable for rendering a single linear memory image: "
                "title, stage, task_summary, visible_reasoning_summary, latest_observation, evidence_files, open_issues, next_action_hints."
            ),
        }[self.memory_format]
        return f"""{artifact_contract}

Memory revision rules:
- Correct stale or wrong previous-memory claims when the latest observation contradicts them.
- Put corrected claims in retained_state or derived_facts only when supported.
- Put discarded or corrected claims in revisions.
- Put unresolved contradictions or missing evidence in open_issues.

Task prompt:
{_truncate(task_prompt, 5000)}

Previous memory summary:
{_truncate(self.previous_summary, 2000)}

Previous memory artifact excerpt:
{_truncate(self.previous_text, 3000)}

Latest visible assistant message:
{_truncate(content_to_text(assistant_message), 5000)}

Latest observation / feedback:
{_truncate(observation_text, 5000)}

Generated or referenced files from latest execution:
{json.dumps(list(generated_files or []), ensure_ascii=False)}
"""

    def update(
        self,
        task_prompt: str,
        assistant_message: Any,
        observation_text: str,
        generated_files: Optional[Iterable[str]] = None,
        stage: str = "execution",
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        generated_files = list(generated_files or [])
        fallback_payload = _default_memory_payload(
            task_prompt=task_prompt,
            previous_summary=self.previous_summary,
            assistant_text=content_to_text(assistant_message),
            observation_text=observation_text,
            generated_files=generated_files,
            stage=stage,
        )
        prompt = self._build_prompt(task_prompt, assistant_message, observation_text, generated_files, stage)
        raw = self._call_llm(prompt)
        if raw is None or raw.startswith("MEMORY_AGENT_ERROR:"):
            artifact_text = (
                _payload_to_html(fallback_payload)
                if self.memory_format == "html"
                else json.dumps(fallback_payload, indent=2, ensure_ascii=False)
            )
            error = raw
        else:
            artifact_text = _strip_fences(raw)
            error = None

        if self.memory_format == "html":
            if "<html" not in artifact_text.lower() or "<body" not in artifact_text.lower():
                parsed = _safe_json_loads(artifact_text)
                artifact_text = _payload_to_html(parsed or fallback_payload)
            path = self.working_dir / f"memory_step_{self.step:03d}.html"
            path.write_text(artifact_text, encoding="utf-8")
            summary = _extract_summary(artifact_text)
            prompt_payload = artifact_text
        elif self.memory_format == "json":
            parsed = _safe_json_loads(artifact_text)
            if parsed is None:
                parsed = fallback_payload
            path = self.working_dir / f"memory_step_{self.step:03d}.json"
            artifact_text = json.dumps(parsed, indent=2, ensure_ascii=False)
            path.write_text(artifact_text, encoding="utf-8")
            summary = _extract_summary(json.dumps(parsed, ensure_ascii=False))
            prompt_payload = artifact_text
        else:
            parsed = _safe_json_loads(artifact_text) or fallback_payload
            path = self.working_dir / f"memory_step_{self.step:03d}.png"
            _render_image_memory(parsed, path)
            artifact_text = json.dumps(parsed, indent=2, ensure_ascii=False)
            summary = _extract_summary(json.dumps(parsed, ensure_ascii=False))
            prompt_payload = f"<img src='{path.name}'>\n{artifact_text}"

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

    def should_inject_full(self, record: Optional[Dict[str, Any]], generated_files=None, stage: str = "") -> bool:
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
        # For simple symbolic/math/graph tasks, avoid injecting full memory on
        # the first successful step. If the task continues, memory becomes useful
        # as external state for the second and later reasoning steps.
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
                "Full memory was not injected for this simple symbolic/math/graph step to avoid unnecessary context noise. "
                "If the task continues, if an error occurs, if evidence files are generated, or if open issues remain, the full memory will be provided.\n"
            )
        payload = _truncate(record.get("prompt_payload", ""), 9000)
        return (
            "\n\nDYNAMIC MEMORY UPDATE:\n"
            f"Memory format: {record['format']}\n"
            f"Memory path: {Path(record['path']).name}\n"
            f"Memory summary: {record['summary']}\n"
            "Use this external memory as mutable state for the next step. It is not a final answer.\n"
            "Before the next action, explicitly check the dynamic memory and state what retained evidence or open issue you are using.\n"
            "Memory artifact content follows:\n"
            f"{payload}\n"
        )
