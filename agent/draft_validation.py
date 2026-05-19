import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional


@dataclass
class DraftValidationResult:
    ok: bool
    message: str
    path: Optional[str] = None


class _HTMLStructureParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.errors = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag.lower())

    def error(self, message):
        self.errors.append(message)


def _extract_printed_path(output: str, draft_format: str) -> Optional[str]:
    label = "HTML_DRAFT_PATH" if draft_format == "html" else "JSON_DRAFT_PATH"
    match = re.search(rf"^{label}\s*:\s*(.+?)\s*$", output, flags=re.MULTILINE)
    if match:
        return match.group(1).strip().strip("'\"")
    return None


def _resolve_path(path_text: str, working_dir: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = Path(working_dir) / path
    return path


def _validate_html(path: Path, require_image_ref: bool = False) -> DraftValidationResult:
    text = path.read_text(encoding="utf-8")
    lower = text.lower()
    parser = _HTMLStructureParser()
    parser.feed(text)

    required_tags = {"html", "head", "body"}
    missing_tags = sorted(tag for tag in required_tags if tag not in parser.tags)
    required_markers = [
        "thinking",
        "objects",
        "relations",
        "state",
        "revision",
        "retained",
        "image",
    ]
    missing_markers = [marker for marker in required_markers if marker not in lower]

    if path.suffix.lower() not in {".html", ".htm"}:
        return DraftValidationResult(False, "HTML draft path must end with .html or .htm.", str(path))
    if not text.strip():
        return DraftValidationResult(False, "HTML draft file is empty.", str(path))
    if missing_tags:
        return DraftValidationResult(False, f"HTML draft is missing tags: {', '.join(missing_tags)}.", str(path))
    if missing_markers:
        return DraftValidationResult(
            False,
            "HTML draft is missing required reasoning sections or labels: "
            + ", ".join(missing_markers)
            + ".",
            str(path),
        )
    if parser.errors:
        return DraftValidationResult(False, "HTML parser reported structural errors.", str(path))
    if require_image_ref and "img" not in parser.tags:
        return DraftValidationResult(
            False,
            "HTML draft must reference at least one evidence image with an <img> tag for this task.",
            str(path),
        )
    return DraftValidationResult(True, "HTML draft validation passed.", str(path))


def _validate_json(path: Path, require_image_ref: bool = False) -> DraftValidationResult:
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, dict):
        return DraftValidationResult(False, "JSON draft root must be an object.", str(path))

    required_keys = [
        "title",
        "original_prompt",
        "thinking_text",
        "objects_entities",
        "relations_constraints",
        "state_effects_view",
        "retained_context",
        "referenced_images",
        "open_issues_revision_targets",
    ]
    missing = [key for key in required_keys if key not in data]
    if path.suffix.lower() != ".json":
        return DraftValidationResult(False, "JSON draft path must end with .json.", str(path))
    if missing:
        return DraftValidationResult(False, f"JSON draft is missing keys: {', '.join(missing)}.", str(path))
    if not isinstance(data.get("referenced_images"), list):
        return DraftValidationResult(False, "JSON draft key referenced_images must be a list.", str(path))
    if require_image_ref and len(data.get("referenced_images", [])) == 0:
        return DraftValidationResult(
            False,
            "JSON draft must include at least one evidence image in referenced_images for this task.",
            str(path),
        )
    if not isinstance(data.get("retained_context"), dict):
        return DraftValidationResult(False, "JSON draft key retained_context must be an object.", str(path))
    return DraftValidationResult(True, "JSON draft validation passed.", str(path))


def validate_draft_output(
    output: str,
    working_dir: str,
    draft_format: Optional[str],
    require_image_ref: bool = False,
) -> DraftValidationResult:
    if draft_format not in {"html", "json"}:
        return DraftValidationResult(True, "No draft validation requested.")

    path_text = _extract_printed_path(output, draft_format)
    label = "HTML_DRAFT_PATH" if draft_format == "html" else "JSON_DRAFT_PATH"
    if not path_text:
        return DraftValidationResult(False, f"Execution output must print `{label}: <path>`.")

    path = _resolve_path(path_text, working_dir)
    if not path.exists():
        return DraftValidationResult(False, f"Printed draft path does not exist: {path}.", str(path))
    if not path.is_file():
        return DraftValidationResult(False, f"Printed draft path is not a file: {path}.", str(path))

    try:
        if draft_format == "html":
            return _validate_html(path, require_image_ref=require_image_ref)
        return _validate_json(path, require_image_ref=require_image_ref)
    except json.JSONDecodeError as exc:
        return DraftValidationResult(False, f"JSON draft is invalid: {exc}.", str(path))
    except UnicodeDecodeError:
        return DraftValidationResult(False, "Draft file must be UTF-8 text.", str(path))
    except Exception as exc:
        return DraftValidationResult(False, f"Draft validation failed: {exc}.", str(path))
