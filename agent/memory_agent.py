import base64
import html as _html_module
import io
import json
import re
import textwrap
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

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
  "key_finding": "One sentence: the single most important discovery or action this step.",
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
      "notes": ["short observation about this image"],
      "annotations": ["object label for detection"],
      "arrows": [
        {"from_xy": [0.125, 0.875], "to_xy": [0.875, 0.125], "label": "Rook e1→e8", "color": "#e11d48"}
      ]
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
- Keep the object compact. Aim for ≤ 3 evidence items unless the task genuinely requires more.
- Evidence pruning: compare against previous_evidence (provided in the prompt). Remove items that have been fully analyzed, are superseded by a better crop, or are no longer needed to answer the task. Only carry forward evidence that the next reasoning step will actually use.
- key_finding: required, one sentence, describes what was learned or done this step.
- latest_observation: short note about the newest result. Never paste raw HTML, JSON blobs, or markup.
- Never write the final answer. Never invent visual details.
- annotations: ≤3 short object labels for GroundingDINO detection on that image. Leave empty if not needed.
- arrows: use to annotate moves or paths ON an image. Coordinates are normalized [0,1] from top-left of the image. Use for chess moves, spatial relationships, flow direction, etc. Leave empty if not needed.
- If model_reflection signals a mistake, record the correction in notes.corrections and revise context fields.
- Omit empty arrays and empty strings.
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


# ---------------------------------------------------------------------------
# Sketch-canvas helpers
# ---------------------------------------------------------------------------

def _is_canvas_file(path: str) -> bool:
    """Return True if path is a sketch_canvas HTML file (has .items.json sidecar)."""
    return Path(str(path) + ".items.json").exists()


def _load_canvas_sidecar(path: str) -> Optional[Dict[str, Any]]:
    sidecar = Path(str(path) + ".items.json")
    if not sidecar.exists():
        return None
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return None


def _canvas_image_paths(items: List[Dict]) -> List[str]:
    return [it["path"] for it in items if it.get("type") == "image" and it.get("path")]


def _canvas_text_notes(items: List[Dict]) -> List[str]:
    notes = []
    for it in items:
        if it.get("type") == "text" and it.get("content"):
            notes.append(str(it["content"]))
        elif it.get("type") == "arrow" and it.get("label"):
            notes.append(f"[arrow] {it['label']}")
    return notes


def _canvas_structure_summary(sidecar: Dict[str, Any]) -> str:
    items = sidecar.get("items", [])
    by_type: Dict[str, int] = {}
    for it in items:
        t = it.get("type", "?")
        by_type[t] = by_type.get(t, 0) + 1
    parts = [f"{v} {k}(s)" for k, v in sorted(by_type.items())]
    title = sidecar.get("title", "")
    return (f'"{title}" — ' if title else "") + (", ".join(parts) if parts else "empty")


def _canvas_evidence_section_html(canvas_path: str, idx: int) -> str:
    """Render a canvas as a memory HTML section (images + text notes)."""
    sidecar = _load_canvas_sidecar(canvas_path)
    if not sidecar:
        esc = _html_escape(canvas_path)
        return (
            f"<div class='evidence-card'>"
            f"<p><strong>Sketch Canvas {idx}</strong> <code>{esc}</code></p>"
            f"</div>"
        )
    items = sidecar.get("items", [])
    title = _html_escape(sidecar.get("title", f"Sketch Canvas {idx}"))
    img_paths  = _canvas_image_paths(items)
    text_notes = _canvas_text_notes(items)

    img_figures = ""
    for i, p in enumerate(img_paths, start=1):
        esc = _html_escape(p)
        img_figures += (
            f"<figure style='display:inline-block;max-width:46%;margin:6px 8px 6px 0;vertical-align:top;'>"
            f"<img src='{esc}' alt='canvas image {i}' style='max-width:100%;border:1px solid #d0d7de;'>"
            f"<figcaption style='font-size:11px;overflow-wrap:anywhere;'><code>{esc}</code></figcaption>"
            f"</figure>"
        )

    note_items = "".join(f"<li>{_html_escape(n)}</li>" for n in text_notes)
    note_block = f"<ul>{note_items}</ul>" if note_items else ""

    summary = _canvas_structure_summary(sidecar)
    canvas_esc = _html_escape(canvas_path)

    return (
        f"<div class='evidence-card' style='border-left:4px solid #7c3aed;padding:10px 12px;"
        f"background:#faf5ff;margin-bottom:12px;'>"
        f"<p><strong>Spatial Reasoning Canvas: {title}</strong>"
        f" <code style='font-size:11px;'>{canvas_esc}</code></p>"
        f"<p style='font-size:12px;color:#6b7280;'>Structure: {_html_escape(summary)}</p>"
        + (f"<div>{img_figures}</div>" if img_figures else "")
        + (f"<div style='margin-top:8px;'><strong>Canvas notes &amp; arrows:</strong>{note_block}</div>"
           if note_block else "")
        + "</div>"
    )


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

    if payload.get("key_finding"):
        result["key_finding"] = str(payload["key_finding"])

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
        # preserve arrows for move/path annotation
        arrows = item.get("arrows")
        if isinstance(arrows, list) and arrows:
            cleaned["arrows"] = arrows
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


def _embed_image_with_arrows(
    path: str,
    img_w: int,
    arrows: List[Dict],
    grounding_fn: Optional[Callable] = None,
    labels: Optional[List[str]] = None,
) -> tuple:
    """Open image, optionally annotate with GroundingDINO boxes and arrow overlays.
    Returns (b64_data_uri, rendered_height) or (None, fallback_height)."""
    try:
        pil_img = grounding_fn(path, labels or []) if (grounding_fn and labels) else Image.open(path).convert("RGB")
        nat_w, nat_h = pil_img.size
        ih = max(60, int(img_w * nat_h / nat_w))
        resample = getattr(Image, "LANCZOS", getattr(Image, "BICUBIC", 1))
        pil_img = pil_img.resize((img_w, ih), resample=resample)

        # draw move/path arrows on the image
        if arrows:
            draw = ImageDraw.Draw(pil_img)
            for arr in arrows:
                try:
                    fx, fy = arr["from_xy"]
                    tx, ty = arr["to_xy"]
                    color = arr.get("color", "#e11d48")
                    x1, y1 = int(fx * img_w), int(fy * ih)
                    x2, y2 = int(tx * img_w), int(ty * ih)
                    # thick arrow line
                    draw.line([(x1, y1), (x2, y2)], fill=color, width=3)
                    # arrowhead triangle
                    import math
                    angle = math.atan2(y2 - y1, x2 - x1)
                    tip_len = 12
                    spread = 0.45
                    pts = [
                        (x2, y2),
                        (x2 - int(tip_len * math.cos(angle - spread)),
                         y2 - int(tip_len * math.sin(angle - spread))),
                        (x2 - int(tip_len * math.cos(angle + spread)),
                         y2 - int(tip_len * math.sin(angle + spread))),
                    ]
                    draw.polygon(pts, fill=color)
                    lbl = arr.get("label", "")
                    if lbl:
                        mid_x, mid_y = (x1 + x2) // 2 + 4, (y1 + y2) // 2 - 12
                        draw.rectangle([mid_x - 2, mid_y - 10, mid_x + len(lbl) * 6, mid_y + 2], fill="white")
                        draw.text((mid_x, mid_y - 9), lbl, fill=color)
                except Exception:
                    pass

        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), ih
    except Exception:
        return None, int(img_w * 0.75)


def _build_timeline_svg(
    step_history: List[Dict],
    grounding_fn: Optional[Callable] = None,
    canvas_w: int = 960,
) -> str:
    """Render a cross-step vertical timeline SVG.

    Each step_history entry: {step, stage, key_finding, evidence: [...]}
    Evidence thumbnails (max 3 per step) are placed to the right of the step header.
    Dashed vertical arrows connect consecutive states.
    The latest state is highlighted with a blue border.
    Returns SVG string, or "" if history is empty.
    """
    if not step_history:
        return ""

    THUMB_W = 130
    THUMB_MAX_H = 110
    LEFT_COL = 220    # width of step-header column
    THUMB_GAP = 10
    BLOCK_PAD = 12
    BLOCK_MARGIN = 8  # vertical gap between blocks
    BADGE_R = 14
    canvas_h_so_far = 16  # top padding

    STAGE_COLORS = {
        "execution_success": "#16a34a",
        "execution_error": "#dc2626",
        "analysis": "#7c3aed",
        "planning": "#0369a1",
    }

    parts: List[str] = []
    parts.append(
        "<defs>"
        "<marker id='tl-arr' markerWidth='8' markerHeight='6' refX='8' refY='3' orient='auto'>"
        "<polygon points='0 0,8 3,0 6' fill='#64748b'/></marker>"
        "<filter id='tl-shadow' x='-5%' y='-5%' width='110%' height='110%'>"
        "<feDropShadow dx='0' dy='1' stdDeviation='1.5' flood-color='#00000018'/></filter>"
        "</defs>"
    )

    block_rects: List[Dict] = []  # store y/h per block for connector arrows

    for idx, state in enumerate(step_history):
        step_num = state.get("step", idx)
        stage = state.get("stage", "execution")
        key_finding = str(state.get("key_finding", "") or "")[:120]
        evidence = [e for e in (state.get("evidence") or []) if isinstance(e, dict)]
        img_evidence = [e for e in evidence if e.get("type") == "image" and e.get("path") and Path(str(e["path"])).exists()][:3]

        is_latest = idx == len(step_history) - 1
        badge_color = STAGE_COLORS.get(stage, "#1e40af") if not is_latest else "#0f172a"
        block_fill = "#f0f9ff" if is_latest else "#f8fafc"
        block_stroke = "#1e40af" if is_latest else "#e2e8f0"
        block_stroke_w = 2 if is_latest else 1

        # Compute thumbnail row height
        thumb_row_h = 0
        thumb_entries: List[Dict] = []
        for ev in img_evidence:
            b64, ih = _embed_image_with_arrows(
                str(ev["path"]),
                THUMB_W,
                ev.get("arrows") or [],
                grounding_fn,
                [str(a) for a in (ev.get("annotations") or []) if a],
            )
            ih = min(ih, THUMB_MAX_H)
            thumb_entries.append({"ev": ev, "b64": b64, "ih": ih})
            thumb_row_h = max(thumb_row_h, ih)

        # Block height: header text rows + thumbnail row
        n_chips = sum(len(e.get("notes") or []) + len(e.get("annotations") or []) for e in img_evidence)
        text_h = max(48, 18 + (16 if key_finding else 0))
        block_h = BLOCK_PAD + max(text_h, thumb_row_h + 4) + BLOCK_PAD + (16 if n_chips > 0 else 0)

        block_y = canvas_h_so_far
        block_rects.append({"y": block_y, "h": block_h})
        canvas_h_so_far += block_h + BLOCK_MARGIN

        # block background
        filter_attr = "filter='url(#tl-shadow)'" if is_latest else ""
        parts.append(
            f"<rect x='12' y='{block_y}' width='{canvas_w - 24}' height='{block_h}' "
            f"rx='6' fill='{block_fill}' stroke='{block_stroke}' stroke-width='{block_stroke_w}' {filter_attr}/>"
        )

        # step badge
        bx, by = 12 + BADGE_R + 6, block_y + block_h // 2
        parts.append(f"<circle cx='{bx}' cy='{by}' r='{BADGE_R}' fill='{badge_color}'/>")
        parts.append(
            f"<text x='{bx}' y='{by + 5}' text-anchor='middle' font-size='12' "
            f"font-weight='bold' fill='white'>{step_num + 1}</text>"
        )

        # stage label + key_finding
        tx = 12 + BADGE_R * 2 + 16
        ty = block_y + BLOCK_PAD + 14
        stage_color = STAGE_COLORS.get(stage, "#475569")
        parts.append(
            f"<text x='{tx}' y='{ty}' font-size='10' fill='{stage_color}' font-weight='bold'>"
            f"{_html_escape(stage.replace('_', ' ').upper())}</text>"
        )
        if key_finding:
            # wrap key_finding across up to 2 lines within LEFT_COL
            max_chars = (LEFT_COL - 20) // 6
            line1 = key_finding[:max_chars]
            line2 = key_finding[max_chars:max_chars * 2]
            parts.append(
                f"<text x='{tx}' y='{ty + 16}' font-size='11' fill='#1e293b' font-weight='bold'>"
                f"{_html_escape(line1)}</text>"
            )
            if line2:
                parts.append(
                    f"<text x='{tx}' y='{ty + 30}' font-size='11' fill='#1e293b'>"
                    f"{_html_escape(line2)}</text>"
                )

        # thumbnails
        if thumb_entries:
            thumb_x = LEFT_COL + 20
            for te in thumb_entries:
                ev, b64, ih = te["ev"], te["b64"], te["ih"]
                thumb_y = block_y + (block_h - ih) // 2
                if b64:
                    parts.append(
                        f"<image href='{b64}' xlink:href='{b64}' "
                        f"x='{thumb_x}' y='{thumb_y}' width='{THUMB_W}' height='{ih}' "
                        f"preserveAspectRatio='xMidYMid meet' rx='3'/>"
                    )
                else:
                    parts.append(
                        f"<rect x='{thumb_x}' y='{thumb_y}' width='{THUMB_W}' height='{ih}' "
                        f"fill='#e2e8f0' rx='3'/>"
                    )
                # caption below thumb
                cap = _html_escape(str(ev.get("caption") or "")[:28])
                parts.append(
                    f"<text x='{thumb_x}' y='{thumb_y + ih + 11}' font-size='9' fill='#64748b'>{cap}</text>"
                )
                thumb_x += THUMB_W + THUMB_GAP

    # inter-block connector arrows
    for i in range(len(block_rects) - 1):
        b1, b2 = block_rects[i], block_rects[i + 1]
        mx = 12 + BADGE_R + 6  # center of badges
        y1 = b1["y"] + b1["h"] + 1
        y2 = b2["y"] - 1
        parts.append(
            f"<line x1='{mx}' y1='{y1}' x2='{mx}' y2='{y2}' "
            f"stroke='#64748b' stroke-width='1.5' stroke-dasharray='5 3' "
            f"marker-end='url(#tl-arr)'/>"
        )

    total_h = canvas_h_so_far + 8
    svg_header = (
        f"<svg xmlns='http://www.w3.org/2000/svg' "
        f"xmlns:xlink='http://www.w3.org/1999/xlink' "
        f"width='100%' viewBox='0 0 {canvas_w} {total_h}' "
        f"style='font-family:Arial,sans-serif;display:block;max-width:{canvas_w}px;'>"
        f"<rect width='{canvas_w}' height='{total_h}' fill='#f8fafc' rx='8'/>"
    )
    return svg_header + "".join(parts) + "</svg>"


def _build_evidence_svg(
    evidence_items: List[Dict],
    grounding_fn: Optional[Callable] = None,
    canvas_w: int = 960,
) -> str:
    """Render image evidence in a multi-row grid SVG with arrows and note chips.

    Layout: up to 3 columns per row, wraps automatically for N > 3 images.
    Sequential dashed arrows connect images left-to-right within a row;
    a descending arrow connects the last image of one row to the first of the next.
    grounding_fn: optional callable(path, labels) -> PIL.Image for bbox overlay.
    Returns an SVG string, or "" if no valid image evidence items exist.
    """
    img_items = [
        e for e in (evidence_items or [])
        if isinstance(e, dict)
        and e.get("type") == "image"
        and e.get("path")
        and Path(str(e["path"])).exists()
    ]
    if not img_items:
        return ""

    MAX_COLS = 3
    margin = 20
    col_gap = 16
    row_gap = 28
    header_h = 28
    cap_h = 18
    note_line_h = 20

    N = len(img_items)
    cols = min(N, MAX_COLS)
    img_w = max(80, (canvas_w - 2 * margin - (cols - 1) * col_gap) // cols)

    # ── Build per-entry data ───────────────────────────────────────────────────
    entries: List[Dict] = []
    for i, item in enumerate(img_items):
        path = str(item["path"])
        labels = [str(a) for a in (item.get("annotations") or []) if a]
        arrows = item.get("arrows") or []
        b64uri, ih = _embed_image_with_arrows(path, img_w, arrows, grounding_fn, labels)
        notes_raw = [str(n)[:60] for n in (item.get("notes") or [])]
        annots_raw = [f"[detect] {a}"[:60] for a in (item.get("annotations") or [])]
        # show arrow labels as chips too
        arrow_chips = [f"→ {a.get('label', '')}"[:60] for a in arrows if a.get("label")]
        chips = (notes_raw + annots_raw + arrow_chips)[:4]
        entries.append({
            "col": i % cols,
            "row": i // cols,
            "iw": img_w,
            "ih": ih,
            "b64uri": b64uri,
            "item": item,
            "chips": chips,
        })

    # ── Compute per-row heights ────────────────────────────────────────────────
    num_rows = (N + cols - 1) // cols
    row_max_ih: List[int] = []
    row_max_chips: List[int] = []
    for r in range(num_rows):
        row_es = [e for e in entries if e["row"] == r]
        row_max_ih.append(max(e["ih"] for e in row_es))
        row_max_chips.append(max(len(e["chips"]) for e in row_es))

    # y coordinate of img top for each row
    row_img_y: List[int] = [header_h]
    for r in range(1, num_rows):
        prev_h = row_max_ih[r - 1] + cap_h + max(1, row_max_chips[r - 1]) * note_line_h + row_gap
        row_img_y.append(row_img_y[r - 1] + prev_h)

    canvas_h = (
        row_img_y[-1]
        + row_max_ih[-1]
        + cap_h
        + max(1, row_max_chips[-1]) * note_line_h
        + margin
    )

    # ── Emit SVG ──────────────────────────────────────────────────────────────
    parts: List[str] = []
    parts.append(f"<rect width='{canvas_w}' height='{canvas_h}' fill='#f8fafc' rx='6'/>")
    parts.append(
        "<defs><marker id='mem-arr' markerWidth='8' markerHeight='6' "
        "refX='8' refY='3' orient='auto'>"
        "<polygon points='0 0,8 3,0 6' fill='#94a3b8'/></marker></defs>"
    )
    # canvas label
    parts.append(
        f"<text x='{margin}' y='18' font-size='11' fill='#64748b' "
        f"font-weight='bold'>Evidence Canvas  ({N} item{'s' if N != 1 else ''})</text>"
    )

    for idx, e in enumerate(entries):
        c, r = e["col"], e["row"]
        iw, ih = e["iw"], e["ih"]
        b64uri, item, chips = e["b64uri"], e["item"], e["chips"]
        x = margin + c * (iw + col_gap)
        img_y = row_img_y[r]

        # image or placeholder
        if b64uri:
            parts.append(
                f"<image href='{b64uri}' xlink:href='{b64uri}' "
                f"x='{x}' y='{img_y}' width='{iw}' height='{ih}' "
                f"preserveAspectRatio='xMidYMid meet'/>"
            )
        else:
            parts.append(
                f"<rect x='{x}' y='{img_y}' width='{iw}' height='{ih}' "
                f"fill='#e2e8f0' stroke='#94a3b8' stroke-width='1' rx='4'/>"
            )
            parts.append(
                f"<text x='{x + iw // 2}' y='{img_y + ih // 2}' "
                f"text-anchor='middle' font-size='11' fill='#94a3b8'>unavailable</text>"
            )

        # sequence badge
        parts.append(
            f"<rect x='{x + iw - 18}' y='{img_y + 2}' width='16' height='16' "
            f"rx='8' fill='#1e40af' opacity='0.85'/>"
        )
        parts.append(
            f"<text x='{x + iw - 10}' y='{img_y + 14}' font-size='10' "
            f"text-anchor='middle' fill='white' font-weight='bold'>{idx + 1}</text>"
        )

        # caption
        caption = _html_escape(str(item.get("caption") or item.get("id") or "Evidence")[:52])
        parts.append(
            f"<text x='{x}' y='{img_y + ih + 13}' "
            f"font-size='11' font-weight='bold' fill='#334155'>{caption}</text>"
        )

        # note / annotation chips
        for j, chip in enumerate(chips):
            ny = img_y + ih + cap_h + (j + 1) * note_line_h
            nw = min(iw, len(chip) * 6 + 12)
            if "[detect]" in chip:
                fill, stroke, fg = "#ede9fe", "#7c3aed", "#4c1d95"
            else:
                fill, stroke, fg = "#fef9c3", "#fbbf24", "#78350f"
            parts.append(
                f"<rect x='{x}' y='{ny - 13}' width='{nw}' height='16' "
                f"rx='3' fill='{fill}' stroke='{stroke}' stroke-width='1'/>"
            )
            parts.append(
                f"<text x='{x + 5}' y='{ny}' font-size='10' fill='{fg}'>"
                f"{_html_escape(chip)}</text>"
            )

        # ── arrows ────────────────────────────────────────────────────────────
        if idx == 0:
            continue
        prev = entries[idx - 1]
        pc, pr = prev["col"], prev["row"]
        px = margin + pc * (img_w + col_gap)
        py_img = row_img_y[pr]

        if pr == r:
            # same row: horizontal arrow
            x1, y1 = px + img_w + 2, py_img + prev["ih"] // 2
            x2, y2 = x - 2, img_y + ih // 2
            parts.append(
                f"<path d='M{x1},{y1} L{x2},{y2}' stroke='#94a3b8' stroke-width='1.5' "
                f"stroke-dasharray='5 3' marker-end='url(#mem-arr)' fill='none'/>"
            )
        else:
            # row break: elbow down from last of prev row to first of this row
            x1, y1 = px + img_w // 2, py_img + prev["ih"] + cap_h + 4
            x2, y2 = x + img_w // 2, img_y - 4
            mid_y = (y1 + y2) // 2
            parts.append(
                f"<path d='M{x1},{y1} C{x1},{mid_y} {x2},{mid_y} {x2},{y2}' "
                f"stroke='#94a3b8' stroke-width='1.5' stroke-dasharray='5 3' "
                f"marker-end='url(#mem-arr)' fill='none'/>"
            )

    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' "
        f"xmlns:xlink='http://www.w3.org/1999/xlink' "
        f"width='100%' viewBox='0 0 {canvas_w} {canvas_h}' "
        f"style='font-family:Arial,sans-serif;display:block;max-width:{canvas_w}px;'>"
        + "".join(parts)
        + "</svg>"
    )


def _payload_to_html(payload: Dict[str, Any], evidence_svg: str = "") -> str:
    payload = _compact_canonical_payload(payload)

    # Separate image and non-image evidence
    non_image_evidence = []
    fallback_image_figures = []
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
            fallback_image_figures.append(
                "<figure class='evidence-figure'>"
                f"<div class='image-wrap'><img src='{escaped}' alt='evidence image {idx}'>"
                f"<span class='badge'>{idx}</span></div>"
                f"<figcaption><strong>{caption}</strong>"
                + (f" <code>{escaped}</code>" if path else "")
                + f"{note_block}</figcaption>"
                "</figure>"
            )
        elif path:
            non_image_evidence.append(
                "<div class='evidence-card'>"
                f"<p><strong>{caption}</strong> <code>{escaped}</code></p>"
                f"{note_block}"
                "</div>"
            )
        elif caption or note_block:
            non_image_evidence.append(
                "<div class='evidence-card'>"
                f"<p><strong>{caption}</strong></p>{note_block}"
                "</div>"
            )

    # Use SVG spatial layout when available, fall back to <figure> list
    if evidence_svg:
        evidence_html = (
            "<div class='evidence-svg-wrap' style='margin-bottom:12px;'>"
            + evidence_svg
            + "</div>"
            + "".join(non_image_evidence)
        )
    else:
        all_items = fallback_image_figures + non_image_evidence
        evidence_html = "".join(all_items) or "<p>No retained image evidence.</p>"

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
        enable_grounding: bool = False,
        llm_model: Optional[str] = None,
    ) -> None:
        self.memory_format = memory_format
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.working_dir = Path(working_dir)
        self.task_name = task_name
        self.path_aliases = {
            str(src): str(dst)
            for src, dst in (path_aliases or {}).items()
            if src and dst
        }
        self.enable_grounding = enable_grounding
        self.step = 0
        self.previous_text = ""
        self.previous_summary = ""
        self.previous_canonical_text = ""
        self.previous_canonical_payload: Dict[str, Any] = {}
        self.step_history: List[Dict[str, Any]] = []  # [{step, stage, key_finding, evidence}]
        self.records: List[Dict[str, Any]] = []

    @property
    def enabled(self) -> bool:
        return self.memory_format in {"html", "json", "image"}

    def _canvas_context_block(self, generated_files: List[str]) -> str:
        """Build the sketch-canvas context block for the extraction prompt."""
        canvas_files = [p for p in generated_files if _is_canvas_file(p)]
        if not canvas_files:
            return ""
        lines = ["Sketch canvas files generated at this step (spatial reasoning artifacts):"]
        for p in canvas_files:
            sidecar = _load_canvas_sidecar(p)
            if sidecar:
                items = sidecar.get("items", [])
                img_paths  = _canvas_image_paths(items)
                text_notes = _canvas_text_notes(items)
                lines.append(f"  Canvas: {p}")
                lines.append(f"    Structure: {_canvas_structure_summary(sidecar)}")
                if img_paths:
                    lines.append(f"    Images on canvas: {img_paths}")
                if text_notes:
                    lines.append(f"    Annotations: {text_notes}")
            else:
                lines.append(f"  Canvas: {p}")
        lines.append(
            "When retaining a canvas, add it to evidence with type='sketch_canvas' and path=<canvas_path>."
        )
        return "\n".join(lines)

    def _inject_canvas_sections(self, artifact_text: str, generated_files: List[str]) -> str:
        """Inject spatial canvas sections into a memory HTML artifact."""
        canvas_files = [p for p in generated_files if _is_canvas_file(p)]
        if not canvas_files:
            return artifact_text
        canvas_html = "\n<section>\n<h2>Spatial Reasoning Canvas</h2>\n"
        for idx, p in enumerate(canvas_files, start=1):
            canvas_html += _canvas_evidence_section_html(p, idx)
        canvas_html += "</section>\n"
        if "</body>" in artifact_text.lower():
            artifact_text = re.sub(
                r"</body>", canvas_html + "</body>", artifact_text, flags=re.IGNORECASE, count=1
            )
        else:
            artifact_text += canvas_html
        return artifact_text

    def _canvas_image_paths_for_gallery(self, generated_files: List[str]) -> List[str]:
        """Collect all image paths referenced in canvas files (for JSON format gallery)."""
        paths: List[str] = []
        for p in generated_files:
            if not _is_canvas_file(p):
                continue
            sidecar = _load_canvas_sidecar(p)
            if sidecar:
                for ip in _canvas_image_paths(sidecar.get("items", [])):
                    if ip not in paths:
                        paths.append(ip)
        return paths

    def _try_annotate_image(self, image_path: str, labels: List[str]):
        """Call GroundingDINO and draw detection boxes on the image using PIL.

        Returns an annotated PIL.Image on success, or the original PIL.Image if
        the server is unavailable or detection fails. Silently swallows errors.
        """
        img = Image.open(image_path).convert("RGB")
        if not self.enable_grounding or not labels:
            return img
        try:
            from tools import detection  # requires running Gradio server
            _, boxes = detection(img, labels[:3])
            draw = ImageDraw.Draw(img)
            palette = ["#e11d48", "#7c3aed", "#0b5fff"]
            try:
                font = ImageFont.truetype("Arial.ttf", 14)
            except Exception:
                font = ImageFont.load_default()
            iw, ih = img.size
            for j, (box, label) in enumerate(zip(boxes, labels)):
                bx, by, bw, bh = box
                px = int(bx * iw)
                py = int(by * ih)
                pw = int(bw * iw)
                ph = int(bh * ih)
                color = palette[j % len(palette)]
                draw.rectangle([px, py, px + pw, py + ph], outline=color, width=3)
                draw.text((px + 4, max(0, py - 16)), label, fill=color, font=font)
        except Exception:
            pass
        return img

    def _call_llm(self, prompt: str, system_message: str = MEMORY_EXTRACTION_SYSTEM_MESSAGE) -> Optional[str]:
        if self.llm_client is None:
            return None
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
        try:
            # OpenAI-style client (has .chat.completions.create)
            if hasattr(self.llm_client, "chat"):
                model = self.llm_model or "gpt-4o"
                response = self.llm_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                )
                return response.choices[0].message.content
            # AutoGen-style client
            response = self.llm_client.create(messages=messages)
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
        # Longest keys first so image_1 cannot clobber image_10's prefix.
        for src in sorted(self.path_aliases, key=len, reverse=True):
            dst = self.path_aliases[src]
            # Guard against partial-token matches (a trailing digit means a different
            # variable) and against re-appending a suffix the text already carries
            # (image_1 -> image_1.png must leave an existing image_1.png untouched).
            if dst.startswith(src) and len(dst) > len(src):
                lookahead = r"(?!\d)(?!" + re.escape(dst[len(src):]) + r")"
            else:
                lookahead = r"(?!\d)"
            artifact_text = re.sub(
                re.escape(src) + lookahead, lambda _m, d=dst: d, artifact_text
            )
        return artifact_text

    def _html_img_tag(self, path: str, idx: int) -> str:
        try:
            p = Path(path)
            if p.exists():
                ext = p.suffix.lower().lstrip(".")
                mime = {"jpg": "jpeg", "jpeg": "jpeg"}.get(ext, ext) or "png"
                b64 = base64.b64encode(p.read_bytes()).decode()
                src = f"data:image/{mime};base64,{b64}"
                return f'<img src="{src}" alt="generated evidence image {idx}" style="max-width:100%;">'
        except Exception:
            pass
        escaped = _html_escape(path)
        return f'<img src="{escaped}" alt="generated evidence image {idx}" style="max-width:100%;">'

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

        already_has_images = "<img" in artifact_text.lower() or "data:image/" in artifact_text
        if not already_has_images:
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
        prev_evidence = self.previous_canonical_payload.get("evidence", [])
        prev_evidence_summary = json.dumps(
            [{"id": e.get("id"), "type": e.get("type"), "caption": e.get("caption"), "path": e.get("path")}
             for e in prev_evidence if isinstance(e, dict)],
            ensure_ascii=False,
        )
        return f"""Extract one compact canonical memory JSON object.

Memory revision rules:
- EVIDENCE PRUNING: Review the previous evidence list below. Only carry forward evidence items that the NEXT reasoning step will actually need. Drop items that have been fully analyzed or are superseded by a newer crop. Aim for ≤ 3 evidence items total.
- Add new evidence from generated_files only when it contains information not already captured.
- Correct stale or wrong previous-memory claims when the latest observation contradicts them.
- Keep discarded or corrected claims in notes.corrections.
- Keep unresolved contradictions in notes.open_issues.
- For "latest_observation", write only a compact note. Do not paste raw HTML, JSON blobs, or markup.
- If "model_reflection" is non-empty, treat it as the model's self-assessment. Record corrections in notes.corrections.
- arrows: if the task involves spatial movement (chess moves, paths, flows), add arrows to the relevant evidence image using normalized [0,1] coordinates from the image's top-left.

Task prompt:
{_truncate(task_prompt, 5000)}

Previous evidence (decide what to KEEP, DROP, or UPDATE):
{prev_evidence_summary}

Previous memory summary:
{_truncate(self.previous_summary, 1500)}

Previous canonical memory excerpt:
{_truncate(self.previous_canonical_text, 2500)}

Latest visible assistant message:
{_truncate(content_to_text(assistant_message), 5000)}

Latest observation / feedback:
{_truncate(observation_text, 5000)}

New generated files from this step:
{json.dumps(list(generated_files or []), ensure_ascii=False)}

Image path aliases:
{json.dumps(self.path_aliases, ensure_ascii=False)}

Model reflection on this step:
{reflection_text or "(none)"}

{self._canvas_context_block(generated_files or [])}"""

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

        # Append to cross-step timeline history
        self.step_history.append({
            "step": self.step,
            "stage": canonical_payload.get("stage", stage),
            "key_finding": canonical_payload.get("key_finding", ""),
            "evidence": canonical_payload.get("evidence", []),
        })
        self.previous_canonical_payload = canonical_payload

        if self.memory_format == "html":
            grounding_fn = self._try_annotate_image if self.enable_grounding else None
            evidence_svg = _build_evidence_svg(
                canonical_payload.get("evidence", []),
                grounding_fn=grounding_fn,
            )
            artifact_text = _payload_to_html(canonical_payload, evidence_svg=evidence_svg)
            artifact_text = self._rewrite_artifact_paths(artifact_text)
            artifact_text = self._inject_generated_images(artifact_text, generated_files)
            artifact_text = self._inject_canvas_sections(artifact_text, generated_files)
            if "<" not in artifact_text or ">" not in artifact_text:
                artifact_text = _payload_to_html(canonical_payload, evidence_svg=evidence_svg)
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
            "generated_files": generated_files,
        }
        self.records.append(record)
        self.previous_text = artifact_text
        self.previous_canonical_text = canonical_text
        self.previous_summary = summary
        self.step += 1
        # update step_history with final stage (may differ from canonical_payload if LLM corrected it)
        if self.step_history:
            self.step_history[-1]["stage"] = canonical_payload.get("stage", stage)
        save_json(self.working_dir / "memory_index.json", self.records)
        return {**record, "prompt_payload": prompt_payload, "canonical_payload": canonical_payload}

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
            regular_paths = _extract_image_paths_from_json_payload(parsed or {})
            canvas_paths  = self._canvas_image_paths_for_gallery(
                record.get("generated_files", [])
            )
            all_paths = regular_paths + [p for p in canvas_paths if p not in regular_paths]
            image_gallery = _render_prompt_image_gallery(all_paths, heading="Images:")
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
