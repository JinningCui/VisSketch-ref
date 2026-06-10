import os
from typing import Optional

from llm_backend import build_llm_runtime, get_runtime_spec


# set up the agent
MAX_REPLY = 10

os.environ.setdefault("AUTOGEN_USE_DOCKER", "False")


def get_backend_mode(backend: Optional[str] = None) -> str:
    return get_runtime_spec(backend=backend)[0]


def get_model_name(backend: Optional[str] = None) -> str:
    resolved_backend, resolved_model = get_runtime_spec(backend=backend)
    return resolved_model


def get_openai_client_config(
    backend: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
):
    resolved_backend = get_backend_mode(backend)
    if resolved_backend == "local":
        raise ValueError("Local backend does not use an OpenAI client config.")

    _default_key = "sk-IEh7KU9A1qT3FYuKe8Ak0RNb3fc4AjHkguBZYwJN0OBxv3Sj"
    resolved_api_key = api_key if api_key is not None else os.environ.get("VISUAL_SKETCHPAD_API_KEY") or os.environ.get("OPENAI_API_KEY") or _default_key
    resolved_base_url = base_url or os.environ.get("VISUAL_SKETCHPAD_API_BASE_URL", "https://api.kksj.org/v1")
    resolved_model = model or os.environ.get("VISUAL_SKETCHPAD_API_MODEL", os.environ.get("VISUAL_SKETCHPAD_MODEL", "gpt-4o-2024-11-20"))

    return {
        "api_key": resolved_api_key,
        "base_url": resolved_base_url,
        "model": resolved_model,
    }


def get_llm_config():
    """
    Build an AutoGen/OpenAIWrapper-compatible llm_config for the API backend.

    Important: when backend=local, the agent uses `llm_client` instead of `llm_config`.
    Returning `False` here avoids import-time failures when VISUAL_SKETCHPAD_BACKEND=local.
    """
    if get_backend_mode() == "local":
        return False
    client_config = get_openai_client_config()
    return {
        "cache_seed": None,
        "config_list": [
            {
                "model": client_config["model"],
                "temperature": 0.0,
                "api_key": client_config["api_key"],
                "base_url": client_config["base_url"],
            }
        ],
    }


llm_config = get_llm_config()


def build_llm_client(
    backend: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    local_model: Optional[str] = None,
    local_dtype: Optional[str] = None,
    device_map: Optional[str] = None,
):
    return build_llm_runtime(
        backend=backend,
        model=model,
        base_url=base_url,
        api_key=api_key,
        local_model=local_model,
        local_dtype=local_dtype,
        device_map=device_map,
    ).client


def build_llm_runtime_config(
    backend: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    local_model: Optional[str] = None,
    local_dtype: Optional[str] = None,
    device_map: Optional[str] = None,
):
    return build_llm_runtime(
        backend=backend,
        model=model,
        base_url=base_url,
        api_key=api_key,
        local_model=local_model,
        local_dtype=local_dtype,
        device_map=device_map,
    )


def validate_llm_config(backend: Optional[str] = None, api_key: Optional[str] = None):
    resolved_backend = get_backend_mode(backend)
    if resolved_backend == "api":
        resolved_api_key = api_key or os.environ.get("VISUAL_SKETCHPAD_API_KEY") or os.environ.get("OPENAI_API_KEY") or "sk-IEh7KU9A1qT3FYuKe8Ak0RNb3fc4AjHkguBZYwJN0OBxv3Sj"
        if not resolved_api_key:
            raise ValueError(
                "API backend selected but no API key was found. "
                "Set VISUAL_SKETCHPAD_API_KEY or OPENAI_API_KEY, or switch to backend=local."
            )
    else:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception as exc:
            raise ImportError(
                "Local backend selected but PyTorch / Transformers are not available."
            ) from exc


# use this after building your own server. You can also set up the server in other machines and paste them here.
SOM_ADDRESS = os.environ.get("SOM_ADDRESS", "http://localhost:8080/")
GROUNDING_DINO_ADDRESS = os.environ.get("GROUNDING_DINO_ADDRESS", "http://localhost:8081/")
DEPTH_ANYTHING_ADDRESS = os.environ.get("DEPTH_ANYTHING_ADDRESS", "http://localhost:8082/")
