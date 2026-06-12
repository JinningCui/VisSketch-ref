import argparse
import json
import math
import re
from pathlib import Path

from task_registry import TASK_SPECS, get_outputs_root, get_tasks_root, iter_task_instances, resolve_tasks
from utils import content_to_text, extract_answer_text, save_json


def _read_json(path):
    return json.loads(Path(path).read_text())


def _load_raw_output(output_path):
    if not output_path.exists():
        return None
    return _read_json(output_path)


def _extract_final_answer(raw_output):
    if not isinstance(raw_output, list):
        return None
    for message in reversed(raw_output):
        if message.get("role") != "assistant":
            continue
        text = content_to_text(message.get("content", ""))
        answer = extract_answer_text(text)
        if answer:
            return answer
    return None


def _normalize_choice(text):
    if not text:
        return None
    match = re.search(r"\(([A-Da-d])\)|\b([A-Da-d])\b", text)
    if not match:
        return None
    return (match.group(1) or match.group(2)).upper()


def _normalize_bool(text):
    if not text:
        return None
    lower = text.lower()
    if "true" in lower or "isomorphic" in lower and "not isomorphic" not in lower:
        return True
    if "false" in lower or "not isomorphic" in lower:
        return False
    if re.search(r"\byes\b", lower):
        return True
    if re.search(r"\bno\b", lower):
        return False
    return None


def _normalize_label(text):
    if not text:
        return None
    lower = text.lower()
    for token in ["convex", "concave", "even", "odd", "neither", "white", "black", "draw"]:
        if re.search(rf"\b{re.escape(token)}\b", lower):
            return token
    return None


def _extract_number(text):
    if not text:
        return None
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not matches:
        return None
    return float(matches[-1])


def _normalize_math_text(text):
    if not text:
        return ""
    text = text.lower()
    text = text.replace("\\", "")
    text = text.replace("{", "")
    text = text.replace("}", "")
    text = text.replace("(", " ").replace(")", " ")
    text = text.replace("[", " ").replace("]", " ")
    text = text.replace("^circ", " ")
    text = text.replace("circumference", " ")
    text = text.replace("approximately", " ")
    text = text.replace("rounded to the nearest tenth", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _geometry_choice_from_text(text, task_instance):
    if not text:
        return None
    ex = _read_json(Path(task_instance) / "ex.json")
    choices = ex.get("choices") or ex.get("compact_choices") or []
    normalized_answer = _normalize_math_text(text)
    if not normalized_answer:
        return None

    for idx, choice in enumerate(choices):
        normalized_choice = _normalize_math_text(str(choice))
        if normalized_choice and normalized_choice in normalized_answer:
            return "ABCD"[idx]
    return None


def _geometry_choice_from_numeric(text, task_instance):
    if not text:
        return None
    number = _extract_number(text)
    if number is None:
        return None

    ex = _read_json(Path(task_instance) / "ex.json")
    values = ex.get("precise_value") or ex.get("rough_value")
    if not values:
        return None

    best_idx = None
    best_delta = None
    for idx, value in enumerate(values):
        try:
            delta = abs(float(number) - float(value))
        except Exception:
            continue
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_idx = idx

    if best_idx is None:
        return None

    # Geometry options are A/B/C/D in order. Reject wildly mismatched numbers.
    tolerance = max(0.25, abs(float(values[best_idx])) * 0.03)
    if best_delta is not None and best_delta <= tolerance:
        return "ABCD"[best_idx]
    return None


def _mantis_meta(task_instance):
    """Return (question_type, options) for a Mantis instance."""
    req = _read_json(Path(task_instance) / "request.json")
    return req.get("question_type", "multi-choice"), req.get("options", [])


def _mantis_choice(text, task_instance):
    """Extract a choice letter (A, B, C, ...) from the model's free-form answer.

    Mantis has up to 5 options, so the A-D-only `_normalize_choice` is too narrow.
    Prefer the explicit parenthesized form, then fall back to matching the option
    body text, then an "answer is X" phrasing.
    """
    if not text:
        return None
    _, options = _mantis_meta(task_instance)
    match = re.search(r"\(([A-Za-z])\)", text)
    if match:
        return match.group(1).upper()
    lower = text.lower()
    for option in options:
        body_match = re.match(r"\(?([A-Za-z])\)?[\s:.)]*(.+)", str(option))
        if not body_match:
            continue
        letter, body = body_match.group(1).upper(), body_match.group(2).strip().lower()
        if body and body in lower:
            return letter
    match = re.search(r"answer\s*(?:is|:)\s*\(?([A-Za-z])\)?\b", lower)
    if match:
        return match.group(1).upper()
    return None


def _mantis_is_choice_gold(gold):
    return bool(re.fullmatch(r"[A-Za-z]", str(gold).strip()))


def _to_float(value):
    if value is None:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group()) if match else None


_MV_LETTERS = "ABCDEFGHIJ"


def _mathvista_choice(text, choices):
    """Extract an option letter from a MathVista multi-choice answer."""
    if not text:
        return None
    match = re.search(r"\(([A-Za-z])\)", text)
    if match:
        return match.group(1).upper()
    lower = text.lower()
    for idx, choice in enumerate(choices):
        body = str(choice).strip().lower()
        if body and body in lower and idx < len(_MV_LETTERS):
            return _MV_LETTERS[idx]
    match = re.search(r"answer\s*(?:is|:)\s*\(?([A-Za-z])\)?\b", lower)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([A-Za-z])\b\s*$", text.strip())
    if match:
        return match.group(1).upper()
    return None


def normalize_prediction(task_name, final_answer, task_instance=None):
    if task_name == "Mantis":
        question_type = "multi-choice"
        if task_instance is not None:
            question_type, _ = _mantis_meta(task_instance)
        if question_type == "multi-choice":
            return _mantis_choice(final_answer, task_instance)
        return (final_answer or "").strip().lower() or None
    if task_name == "MathVista":
        if task_instance is None:
            return (final_answer or "").strip().lower() or None
        meta = _read_json(Path(task_instance) / "request.json")
        if meta.get("question_type") == "multi_choice":
            return _mathvista_choice(final_answer, meta.get("choices", []))
        answer_type = meta.get("answer_type")
        if answer_type in ("float", "integer"):
            number = _extract_number(final_answer)
            if number is None:
                return None
            if answer_type == "integer":
                return str(int(round(number)))
            precision = meta.get("precision")
            precision = precision if isinstance(precision, int) else 1
            return f"{round(number, precision):.{precision}f}"
        return (final_answer or "").strip().lower() or None
    if task_name in {"geometry", "blink_depth", "blink_jigsaw", "blink_spatial", "mmvp", "vstar"}:
        choice = _normalize_choice(final_answer)
        if choice is not None:
            return choice
        if task_name == "geometry" and task_instance is not None:
            text_choice = _geometry_choice_from_text(final_answer, task_instance)
            if text_choice is not None:
                return text_choice
            return _geometry_choice_from_numeric(final_answer, task_instance)
        return None
    if task_name in {"graph_connectivity", "graph_isomorphism"}:
        return _normalize_bool(final_answer)
    if task_name == "graph_maxflow":
        return _extract_number(final_answer)
    if task_name in {"math_convexity", "math_parity", "winner_id"}:
        return _normalize_label(final_answer)
    return final_answer


def normalize_gold(task_name, gold):
    if task_name == "Mantis":
        gold = str(gold).strip()
        match = re.fullmatch(r"\(?([A-Za-z])\)?", gold)
        if match:
            return match.group(1).upper()
        return gold.lower()
    if task_name == "MathVista":
        gold = str(gold).strip()
        if re.fullmatch(r"[A-Za-z]", gold):
            return gold.upper()
        return gold
    if task_name in {"geometry", "blink_depth", "blink_jigsaw", "blink_spatial", "mmvp", "vstar"}:
        return str(gold).strip().replace("(", "").replace(")", "").upper()
    if task_name in {"graph_connectivity", "graph_isomorphism"}:
        return bool(gold)
    if task_name == "graph_maxflow":
        return float(gold)
    if task_name in {"math_convexity", "math_parity", "winner_id"}:
        return str(gold).strip().lower()
    return gold


def compare_prediction(task_name, pred, gold):
    if pred is None:
        return False
    if task_name == "graph_maxflow":
        return math.isclose(float(pred), float(gold), rel_tol=0.0, abs_tol=1e-6)
    if task_name == "MathVista":
        pred_text, gold_text = str(pred).strip(), str(gold).strip()
        # Numeric answers (float/integer): compare with tolerance. Guard against
        # list/letter golds by requiring the gold to be a single clean number.
        clean_gold = gold_text.replace(",", "")
        if re.fullmatch(r"-?\d+(?:\.\d+)?", clean_gold):
            pred_num, gold_num = _to_float(pred_text), _to_float(gold_text)
            if pred_num is None or gold_num is None:
                return False
            if "." in clean_gold:
                decimals = len(clean_gold.split(".")[1])
                return abs(round(pred_num, decimals) - round(gold_num, decimals)) < 1e-9
            return int(round(pred_num)) == int(round(gold_num))
        return pred_text.lower() == gold_text.lower()
    if task_name == "Mantis":
        if _mantis_is_choice_gold(gold):
            return pred == gold
        # Short-answer: lenient containment in either direction.
        pred_text, gold_text = str(pred).strip().lower(), str(gold).strip().lower()
        if not pred_text or not gold_text:
            return False
        return gold_text == pred_text or gold_text in pred_text or pred_text in gold_text
    return pred == gold


def load_gold_label(task_name, task_instance):
    spec = TASK_SPECS[task_name]
    data = _read_json(task_instance / spec["input_file"])
    return data[spec["label_key"]]


def evaluate_task(task_name, outputs_root=None, project_root=None):
    outputs_root = Path(outputs_root) if outputs_root else get_outputs_root(project_root)
    task_output_root = outputs_root / task_name

    rows = []
    for task_instance in iter_task_instances(task_name, project_root):
        gold = normalize_gold(task_name, load_gold_label(task_name, task_instance))
        output_path = task_output_root / task_instance.name / "output.json"
        raw_output = _load_raw_output(output_path)
        final_answer = _extract_final_answer(raw_output)
        pred = normalize_prediction(task_name, final_answer, task_instance=task_instance)
        correct = compare_prediction(task_name, pred, gold)
        rows.append(
            {
                "task": task_name,
                "instance": task_instance.name,
                "gold": gold,
                "final_answer": final_answer,
                "prediction": pred,
                "correct": correct,
                "has_output": raw_output is not None,
            }
        )

    total = len(rows)
    answered = sum(1 for row in rows if row["has_output"])
    parsed = sum(1 for row in rows if row["prediction"] is not None)
    correct = sum(1 for row in rows if row["correct"])
    correct_instances = [row["instance"] for row in rows if row["correct"]]
    return {
        "task": task_name,
        "total": total,
        "answered": answered,
        "parsed": parsed,
        "correct": correct,
        "correct_instances": correct_instances,
        "accuracy": (correct / total) if total else 0.0,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", nargs="+", default=["all"])
    parser.add_argument("--outputs-dir", type=str, default=str(get_outputs_root()))
    parser.add_argument("--report-dir", type=str, default=None)
    args = parser.parse_args()

    task_names = resolve_tasks(args.task)
    results = [evaluate_task(task_name, args.outputs_dir) for task_name in task_names]
    summary_tasks = []
    correct_task_lists = {}
    for result in results:
        compact = {key: value for key, value in result.items() if key != "rows"}
        summary_tasks.append(compact)
        correct_task_lists[result["task"]] = result["correct_instances"]
    summary = {"tasks": summary_tasks}

    report_dir = Path(args.report_dir) if args.report_dir else Path(args.outputs_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    save_json(report_dir / "evaluation_summary.json", summary)
    save_json(report_dir / "evaluation_details.json", results)
    save_json(report_dir / "correct_task_lists.json", correct_task_lists)

    for item in summary["tasks"]:
        print(
            f"{item['task']}: accuracy={item['accuracy']:.4f} "
            f"correct={item['correct']}/{item['total']} answered={item['answered']}"
        )


if __name__ == "__main__":
    main()
