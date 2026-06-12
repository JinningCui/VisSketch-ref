from pathlib import Path

from llm_backend import get_runtime_spec


TASK_SPECS = {
    "blink_depth": {"task_type": "vision", "input_file": "request.json", "label_key": "answer"},
    "blink_jigsaw": {"task_type": "vision", "input_file": "request.json", "label_key": "answer"},
    "blink_spatial": {"task_type": "vision", "input_file": "request.json", "label_key": "answer"},
    "ChartQA": {"task_type": "vision", "input_file": "request.json", "label_key": "answer", "split": "test"},
    "geometry": {"task_type": "geo", "input_file": "ex.json", "label_key": "answer"},
    "graph_connectivity": {"task_type": "math", "input_file": "example.json", "label_key": "label"},
    "graph_isomorphism": {"task_type": "math", "input_file": "example.json", "label_key": "label"},
    "graph_maxflow": {"task_type": "math", "input_file": "example.json", "label_key": "label"},
    "Mantis": {"task_type": "vision", "input_file": "request.json", "label_key": "answer"},
    "MathVista": {"task_type": "vision", "input_file": "request.json", "label_key": "answer"},
    "vstar": {"task_type": "vision", "input_file": "request.json", "label_key": "answer"},
    "math_convexity": {"task_type": "math", "input_file": "example.json", "label_key": "label"},
    "math_parity": {"task_type": "math", "input_file": "example.json", "label_key": "label"},
    "mmvp": {"task_type": "vision", "input_file": "request.json", "label_key": "answer"},
    "winner_id": {"task_type": "math", "input_file": "example.json", "label_key": "label"},
}


TASK_ORDER = list(TASK_SPECS.keys())


def resolve_tasks(task_names):
    if not task_names or task_names == ["all"]:
        return TASK_ORDER
    unknown = [task for task in task_names if task not in TASK_SPECS]
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}")
    return task_names


def get_project_root():
    return Path(__file__).resolve().parents[1]


def get_tasks_root(project_root=None):
    project_root = Path(project_root) if project_root else get_project_root()
    return project_root / "tasks"


def get_outputs_root(project_root=None):
    project_root = Path(project_root) if project_root else get_project_root()
    return project_root / "outputs"


def sanitize_runtime_component(value):
    value = str(value).strip()
    if not value:
        return "unknown"
    sanitized = []
    for char in value:
        if char.isalnum() or char in {"-", "_", "."}:
            sanitized.append(char)
        else:
            sanitized.append("_")
    normalized = "".join(sanitized).strip("._")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized or "unknown"


def get_runtime_output_root(output_root, runtime_kwargs=None):
    runtime_kwargs = runtime_kwargs or {}
    backend, model_name = get_runtime_spec(
        backend=runtime_kwargs.get("backend"),
        model=runtime_kwargs.get("model"),
        base_url=runtime_kwargs.get("base_url"),
        api_key=runtime_kwargs.get("api_key"),
        local_model=runtime_kwargs.get("local_model"),
        local_dtype=runtime_kwargs.get("local_dtype"),
        device_map=runtime_kwargs.get("device_map"),
    )
    return Path(output_root) / sanitize_runtime_component(backend) / sanitize_runtime_component(model_name)


def iter_task_instances(task_name, project_root=None):
    tasks_root = get_tasks_root(project_root)
    task_root = tasks_root / task_name
    spec = TASK_SPECS[task_name]
    if spec["task_type"] == "vision":
        split = spec.get("split")
        instance_root = task_root / "processed" / split if split else task_root / "processed"
    else:
        instance_root = task_root
    return sorted(path for path in instance_root.iterdir() if path.is_dir())
