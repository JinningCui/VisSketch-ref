import base64
import io
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image


@dataclass
class LLMRuntime:
    backend: str
    model_name: str
    client: Any
    config: Optional[Dict[str, Any]]


def _normalize_backend(backend: Optional[str] = None) -> str:
    value = (backend or os.environ.get("VISUAL_SKETCHPAD_BACKEND", "api")).strip().lower()
    if value in {"api", "openai", "remote"}:
        return "api"
    if value in {"local", "qwen", "qwen3", "qwen3-vl"}:
        return "local"
    raise ValueError(f"Unknown backend: {backend!r}. Expected 'api' or 'local'.")


def _default_api_model() -> str:
    return os.environ.get("VISUAL_SKETCHPAD_API_MODEL", os.environ.get("VISUAL_SKETCHPAD_MODEL", "gpt-5.1"))


_FALLBACK_API_KEY = "sk-IEh7KU9A1qT3FYuKe8Ak0RNb3fc4AjHkguBZYwJN0OBxv3Sj"


def _default_api_key() -> Optional[str]:
    return os.environ.get("VISUAL_SKETCHPAD_API_KEY") or os.environ.get("OPENAI_API_KEY") or _FALLBACK_API_KEY


def _default_base_url() -> str:
    return os.environ.get("VISUAL_SKETCHPAD_API_BASE_URL", "https://api.kksj.org/v1")


def _default_local_model() -> str:
    return os.environ.get("VISUAL_SKETCHPAD_LOCAL_MODEL", "Qwen/Qwen3-VL-4B-Instruct")


def _default_local_dtype() -> str:
    return os.environ.get("VISUAL_SKETCHPAD_LOCAL_DTYPE", "auto")


def _default_local_device_map() -> str:
    return os.environ.get("VISUAL_SKETCHPAD_LOCAL_DEVICE_MAP", "auto")


def get_runtime_spec(
    backend: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    local_model: Optional[str] = None,
    local_dtype: Optional[str] = None,
    device_map: Optional[str] = None,
) -> Tuple[str, str]:
    resolved_backend = _normalize_backend(backend)
    if resolved_backend == "local":
        return resolved_backend, local_model or model or _default_local_model()
    return resolved_backend, model or _default_api_model()


def build_api_config(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_model = model or _default_api_model()
    resolved_api_key = api_key if api_key is not None else _default_api_key()
    resolved_base_url = base_url or _default_base_url()

    if not resolved_api_key:
        raise ValueError(
            "API backend selected but no API key was provided. "
            "Set VISUAL_SKETCHPAD_API_KEY or OPENAI_API_KEY, or switch to backend=local."
        )

    return {
        "cache_seed": None,
        "config_list": [
            {
                "model": resolved_model,
                "temperature": 0.0,
                "api_key": resolved_api_key,
                "base_url": resolved_base_url,
            }
        ],
    }


def _decode_data_url(url: str) -> Image.Image:
    header, encoded = url.split(",", 1)
    raw = base64.b64decode(encoded)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _pil_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _extract_image_reference(block: Dict[str, Any]) -> Optional[str]:
    if "image" in block:
        image_value = block["image"]
        if isinstance(image_value, Image.Image):
            return _pil_to_data_url(image_value)
        if isinstance(image_value, (str, Path)):
            image_path = Path(image_value)
            if image_path.exists():
                return str(image_path)
            if str(image_value).startswith("data:image/"):
                return str(image_value)
    image_url = block.get("image_url")
    if isinstance(image_url, dict):
        image_url = image_url.get("url")
    if isinstance(image_url, str):
        if image_url.startswith("data:image/"):
            return image_url
        image_path = Path(image_url)
        if image_path.exists():
            return str(image_path)
    url = block.get("url")
    if isinstance(url, str):
        if url.startswith("data:image/"):
            return url
        image_path = Path(url)
        if image_path.exists():
            return str(image_path)
    return None


def _load_image_reference(image_ref: str) -> Image.Image:
    if image_ref.startswith("data:image/"):
        return _decode_data_url(image_ref)
    return Image.open(image_ref).convert("RGB")


def _normalize_message_content(content: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, dict):
        content = [content]
    if not isinstance(content, list):
        return [{"type": "text", "text": str(content)}]

    for block in content:
        if isinstance(block, str):
            normalized.append({"type": "text", "text": block})
            continue
        if not isinstance(block, dict):
            normalized.append({"type": "text", "text": str(block)})
            continue

        block_type = block.get("type")
        if block_type == "text":
            normalized.append({"type": "text", "text": block.get("text", "")})
            continue

        image_ref = _extract_image_reference(block)
        if image_ref is not None:
            if image_ref.startswith("data:image/"):
                normalized.append({"type": "image", "image": _decode_data_url(image_ref)})
            else:
                normalized.append({"type": "image", "image": image_ref})
            continue

        normalized.append({"type": "text", "text": str(block)})

    return normalized


def normalize_messages_for_qwen(messages: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized_messages: List[Dict[str, Any]] = []
    for message in messages:
        role = message.get("role", "user")
        content = _normalize_message_content(message.get("content", ""))
        normalized_messages.append({"role": role, "content": content})
    return normalized_messages


def _messages_include_images(messages: Iterable[Dict[str, Any]]) -> bool:
    for message in messages:
        for block in message.get("content", []):
            if isinstance(block, dict) and block.get("type") == "image":
                return True
    return False


def _flatten_images(messages: Iterable[Dict[str, Any]]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for message in messages:
        for block in message.get("content", []):
            if isinstance(block, dict) and block.get("type") == "image":
                image_value = block.get("image")
                if isinstance(image_value, Image.Image):
                    images.append(image_value.convert("RGB"))
                elif isinstance(image_value, (str, Path)):
                    images.append(_load_image_reference(str(image_value)))
    return images


def _text_only_messages(messages: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    rendered_messages: List[Dict[str, str]] = []
    for message in messages:
        text_parts: List[str] = []
        for block in message.get("content", []):
            if not isinstance(block, dict):
                text_parts.append(str(block))
                continue
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "image":
                text_parts.append("[image]")
        rendered_messages.append({"role": message.get("role", "user"), "content": "\n".join(text_parts).strip()})
    return rendered_messages


class LocalQwen3VLClient:
    """Thin OpenAIWrapper-compatible adapter for local Qwen3-VL inference."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        dtype: Optional[str] = None,
        device_map: Optional[str] = None,
        trust_remote_code: bool = True,
    ) -> None:
        try:
            import torch
            from transformers import AutoProcessor, AutoTokenizer
        except Exception as exc:  # pragma: no cover - import error path is environment specific
            raise ImportError(
                "Local backend requires PyTorch and Transformers."
            ) from exc

        resolved_model = model_name or _default_local_model()
        resolved_dtype = (dtype or _default_local_dtype()).strip().lower()
        resolved_device_map = device_map or _default_local_device_map()

        if resolved_dtype in {"auto", "default"}:
            if torch.cuda.is_available():
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
        elif resolved_dtype in {"fp16", "float16", "half"}:
            torch_dtype = torch.float16
        elif resolved_dtype in {"bf16", "bfloat16"}:
            torch_dtype = torch.bfloat16
        elif resolved_dtype in {"fp32", "float32"}:
            torch_dtype = torch.float32
        else:
            raise ValueError(
                f"Unsupported local dtype: {dtype!r}. Use auto, fp16/float16, bf16/bfloat16, or fp32/float32."
            )

        self.torch = torch
        self.model_name = resolved_model
        self.processor = None
        self.tokenizer = None
        self.supports_vision = False
        self.runtime_family = "text"

        model_load_errors: List[str] = []

        try:
            vision_model_class = None
            vision_model_name = None
            try:
                from transformers import Qwen3VLForConditionalGeneration

                vision_model_class = Qwen3VLForConditionalGeneration
                vision_model_name = "Qwen3VLForConditionalGeneration"
            except Exception:
                pass
            if vision_model_class is None:
                try:
                    from transformers import Qwen2VLForConditionalGeneration

                    vision_model_class = Qwen2VLForConditionalGeneration
                    vision_model_name = "Qwen2VLForConditionalGeneration"
                except Exception:
                    pass
            if vision_model_class is None:
                try:
                    from transformers import AutoModelForVision2Seq

                    vision_model_class = AutoModelForVision2Seq
                    vision_model_name = "AutoModelForVision2Seq"
                except Exception:
                    pass

            if vision_model_class is not None:
                self.processor = AutoProcessor.from_pretrained(
                    resolved_model,
                    trust_remote_code=trust_remote_code,
                )
                self.model = vision_model_class.from_pretrained(
                    resolved_model,
                    torch_dtype=torch_dtype,
                    device_map=resolved_device_map,
                    trust_remote_code=trust_remote_code,
                )
                self.supports_vision = True
                self.runtime_family = vision_model_name or "vision"
            else:
                raise ImportError("No vision-capable Qwen model class is available in this Transformers version.")
        except Exception as vision_exc:
            model_load_errors.append(f"vision backend unavailable: {vision_exc}")
            try:
                from transformers import AutoModelForCausalLM

                self.tokenizer = AutoTokenizer.from_pretrained(
                    resolved_model,
                    trust_remote_code=trust_remote_code,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    resolved_model,
                    torch_dtype=torch_dtype,
                    device_map=resolved_device_map,
                    trust_remote_code=trust_remote_code,
                )
                self.runtime_family = "AutoModelForCausalLM"
            except Exception as text_exc:
                model_load_errors.append(f"text backend unavailable: {text_exc}")
                raise ImportError(
                    "Failed to initialize a local Qwen backend. "
                    "For local multimodal tasks, install a recent Transformers version with Qwen2-VL/Qwen3-VL support. "
                    "For local text-only tasks, provide a causal-language-model checkpoint. "
                    f"Details: {'; '.join(model_load_errors)}"
                ) from text_exc

        self.model.eval()
        self.total_usage_summary = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.actual_usage_summary = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _move_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        first_param = next(self.model.parameters())
        target_device = first_param.device
        moved = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                moved[key] = value.to(target_device)
            else:
                moved[key] = value
        return moved

    def _update_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        total_tokens = prompt_tokens + completion_tokens
        self.total_usage_summary["prompt_tokens"] += prompt_tokens
        self.total_usage_summary["completion_tokens"] += completion_tokens
        self.total_usage_summary["total_tokens"] += total_tokens
        self.actual_usage_summary = dict(self.total_usage_summary)

    def create(self, messages: Optional[List[Dict[str, Any]]] = None, context: Any = None, **kwargs: Any) -> Dict[str, Any]:
        del context
        if not messages:
            return {"choices": [{"message": {"content": ""}}]}

        qwen_messages = normalize_messages_for_qwen(messages)
        max_new_tokens = int(
            kwargs.pop(
                "max_new_tokens",
                os.environ.get("VISUAL_SKETCHPAD_MAX_NEW_TOKENS", "512"),
            )
        )
        temperature = kwargs.pop("temperature", None)
        do_sample = bool(kwargs.pop("do_sample", False))
        generation_kwargs = {"max_new_tokens": max_new_tokens}
        if temperature is not None and float(temperature) > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = float(temperature)
        elif do_sample:
            generation_kwargs["do_sample"] = True

        if self.supports_vision:
            try:
                input_kwargs = self.processor.apply_chat_template(
                    qwen_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            except Exception:
                rendered_prompt = self.processor.apply_chat_template(
                    qwen_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                images = _flatten_images(qwen_messages)
                processor_kwargs = {
                    "text": [rendered_prompt],
                    "return_tensors": "pt",
                }
                if images:
                    processor_kwargs["images"] = images
                input_kwargs = self.processor(**processor_kwargs)
            input_kwargs.pop("token_type_ids", None)
        else:
            if _messages_include_images(qwen_messages):
                raise ValueError(
                    f"Local model `{self.model_name}` is running in text-only mode via {self.runtime_family}, "
                    "but the current task contains images. Use an API multimodal model or a local Qwen-VL checkpoint "
                    "with a newer Transformers installation."
                )
            rendered_messages = _text_only_messages(qwen_messages)
            if hasattr(self.tokenizer, "apply_chat_template"):
                rendered_prompt = self.tokenizer.apply_chat_template(
                    rendered_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                rendered_prompt = "\n\n".join(
                    f"{message['role']}: {message['content']}" for message in rendered_messages
                )
            input_kwargs = self.tokenizer([rendered_prompt], return_tensors="pt")

        input_kwargs = self._move_inputs(input_kwargs)

        with self.torch.no_grad():
            generated_ids = self.model.generate(**input_kwargs, **generation_kwargs)

        prompt_len = int(input_kwargs["input_ids"].shape[-1])
        completion_len = int(generated_ids.shape[-1] - prompt_len)
        self._update_usage(prompt_len, max(0, completion_len))

        trimmed_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_kwargs["input_ids"], generated_ids)]
        decoder = self.processor if self.processor is not None else self.tokenizer
        output_text = decoder.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": output_text,
                    }
                }
            ],
            "model": self.model_name,
        }

    def extract_text_or_completion_object(self, response: Any) -> Tuple[Any, Any]:
        if isinstance(response, tuple) and len(response) == 2:
            return response
        if isinstance(response, str):
            return response, response
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", "")
                return content, response
            if "content" in response:
                return response["content"], response
        return response, response


@lru_cache(maxsize=16)
def build_llm_runtime(
    backend: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    local_model: Optional[str] = None,
    local_dtype: Optional[str] = None,
    device_map: Optional[str] = None,
) -> LLMRuntime:
    resolved_backend = _normalize_backend(backend)
    if resolved_backend == "local":
        resolved_model = local_model or model or _default_local_model()
        client = LocalQwen3VLClient(
            model_name=resolved_model,
            dtype=local_dtype,
            device_map=device_map,
        )
        return LLMRuntime(backend="local", model_name=resolved_model, client=client, config=None)

    resolved_model = model or _default_api_model()
    config = build_api_config(model=resolved_model, base_url=base_url, api_key=api_key)
    from autogen.oai.client import OpenAIWrapper

    client = OpenAIWrapper(**config)
    return LLMRuntime(backend="api", model_name=resolved_model, client=client, config=config)


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
