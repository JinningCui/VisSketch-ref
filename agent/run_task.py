import argparse
import contextlib
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from main import run_agent
from task_registry import (
    TASK_SPECS,
    get_outputs_root,
    get_runtime_output_root,
    iter_task_instances,
    resolve_tasks,
)
from utils import save_json


def _task_output_dir(task_name, task_instance, output_root):
    return Path(output_root) / task_name / Path(task_instance).name


def _write_run_error(task_name, task_instance, output_root, error, tb_text):
    task_instance = Path(task_instance)
    task_output_dir = _task_output_dir(task_name, task_instance, output_root)
    task_output_dir.mkdir(parents=True, exist_ok=True)
    for child in task_instance.iterdir():
        if child.is_file():
            shutil.copy2(child, task_output_dir / child.name)
    save_json(
        task_output_dir / "run_error.json",
        {
            "task_name": task_name,
            "task_instance": str(task_instance),
            "error": str(error),
            "traceback": tb_text,
        },
    )


def _is_done(task_name, task_instance, output_root):
    pred = _task_output_dir(task_name, task_instance, output_root) / "prediction_summary.json"
    if not pred.exists():
        return False
    try:
        import json
        d = json.loads(pred.read_text())
        err = d.get("error")
        if err is None:
            return True
        err_str = str(err).lower()
        bad = any(k in err_str for k in ["rate", "quota", "limit", "429", "billing",
                                          "insufficient", "connect", "timeout",
                                          "network", "ssl", "socket"])
        return not bad
    except Exception:
        return False


def _run_one_instance(task_name, task_instance, output_root, runtime_kwargs):
    spec = TASK_SPECS[task_name]
    task_instance = Path(task_instance)
    task_output_dir = _task_output_dir(task_name, task_instance, output_root)

    if _is_done(task_name, task_instance, output_root):
        return {"task": task_name, "instance": task_instance.name, "status": "ok"}

    task_output_dir.mkdir(parents=True, exist_ok=True)
    log_path = task_output_dir / "run.log"

    try:
        with open(log_path, "a", encoding="utf-8") as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                run_agent(
                    str(task_instance),
                    str(Path(output_root) / task_name),
                    task_type=spec["task_type"],
                    task_name=task_name,
                    **runtime_kwargs,
                )
        return {"task": task_name, "instance": task_instance.name, "status": "ok"}
    except Exception as error:
        _write_run_error(task_name, task_instance, output_root, error, traceback.format_exc())
        return {
            "task": task_name,
            "instance": task_instance.name,
            "status": "error",
            "error": str(error),
        }


def run_single_task(task_name, output_root, limit=None, runtime_kwargs=None):
    runtime_kwargs = runtime_kwargs or {}
    instances = iter_task_instances(task_name)
    if limit is not None:
        instances = instances[:limit]

    summary = {"task": task_name, "ok": 0, "error": 0, "total": len(instances)}
    for task_instance in tqdm(instances, desc=task_name):
        result = _run_one_instance(task_name, task_instance, output_root, runtime_kwargs)
        summary[result["status"]] += 1
    return summary


def run_task_group(task_name, output_root, limit=None, runtime_kwargs=None):
    group_log = Path(output_root) / task_name / "_task_group.log"
    group_log.parent.mkdir(parents=True, exist_ok=True)
    with open(group_log, "a", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            return run_single_task(task_name, output_root, limit=limit, runtime_kwargs=runtime_kwargs)


def run_tasks(task_names, output_root, limit=None, workers=1, runtime_kwargs=None):
    runtime_kwargs = runtime_kwargs or {}
    summaries = []
    if workers <= 1:
        for task_name in task_names:
            summaries.append(run_task_group(task_name, output_root, limit=limit, runtime_kwargs=runtime_kwargs))
        return summaries

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(run_task_group, task_name, output_root, limit, runtime_kwargs): task_name
            for task_name in task_names
        }
        with tqdm(total=len(future_map), desc="task-groups") as progress:
            for future in as_completed(future_map):
                summaries.append(future.result())
                progress.update(1)
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        nargs="+",
        default=["all"],
        help="Task names to run, or `all`.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional per-task instance limit.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(get_outputs_root()),
        help="Directory used to store outputs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel task-category workers.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["api", "local"],
        help="LLM backend to use.",
    )
    parser.add_argument("--model", type=str, default=None, help="Model name or local model path.")
    parser.add_argument("--base-url", type=str, default=None, help="API base URL for the API backend.")
    parser.add_argument("--api-key", type=str, default=None, help="API key for the API backend.")
    parser.add_argument("--local-model", type=str, default=None, help="Local model override.")
    parser.add_argument("--local-dtype", type=str, default=None, help="Local model dtype (auto/fp16/bf16/fp32).")
    parser.add_argument("--device-map", type=str, default=None, help="Device map for the local backend.")
    parser.add_argument(
        "--memory-format",
        type=str,
        default=None,
        choices=["html", "json", "image"],
        help="Optional external dynamic memory format maintained by a separate memory agent.",
    )
    args = parser.parse_args()

    task_names = resolve_tasks(args.task)
    runtime_kwargs = {
        "backend": args.backend,
        "model": args.model,
        "base_url": args.base_url,
        "api_key": args.api_key,
        "local_model": args.local_model,
        "local_dtype": args.local_dtype,
        "device_map": args.device_map,
        "memory_format": args.memory_format,
    }
    scoped_output_root = get_runtime_output_root(args.output_dir, runtime_kwargs=runtime_kwargs)
    if args.memory_format:
        scoped_output_root = scoped_output_root / f"memory_{args.memory_format}"
    summaries = run_tasks(
        task_names,
        str(scoped_output_root),
        limit=args.limit,
        workers=max(1, args.workers),
        runtime_kwargs=runtime_kwargs,
    )
    print({"output_root": str(scoped_output_root), "summaries": summaries})


if __name__ == "__main__":
    main()
