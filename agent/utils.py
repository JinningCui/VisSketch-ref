import json
import re
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=4, ensure_ascii=False, default=custom_encoder))

def image_to_base64(pil_img):
    # Create a BytesIO buffer to save the image
    buffered = BytesIO()
    # Save the image in the buffer using a format like PNG
    pil_img.save(buffered, format="PNG")
    # Get the byte data from the buffer
    img_byte = buffered.getvalue()
    # Encode the bytes to base64
    img_base64 = base64.b64encode(img_byte)
    # Decode the base64 bytes to string
    return img_base64.decode('utf-8')


def custom_encoder(obj):
    """Custom JSON encoder function that replaces Image objects with '<image>'.
       Delegates the encoding of other types to the default encoder."""
    if isinstance(obj, Image.Image):
        return image_to_base64(obj)
    # Let the default JSON encoder handle any other types
    return json.JSONEncoder().default(obj)


def content_to_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                else:
                    text_parts.append(str(item))
            else:
                text_parts.append(str(item))
        return "".join(text_parts)
    if isinstance(content, dict):
        return content_to_text(content.get("content", ""))
    return str(content)


def extract_code_block(text):
    match = re.search(r"```python\s*(.*?)```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_answer_text(text):
    matches = re.findall(r"ANSWER:\s*(.*?)(?:TERMINATE|$)", text, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def _extract_tagged_section(text, tag):
    pattern = rf"{tag}\s*(\d+)?\s*:\s*(.*?)(?=\n[A-Z][A-Z ]*\d*\s*:|\Z)"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if not matches:
        return None
    step, body = matches[-1]
    return {
        "step": int(step) if step else None,
        "text": body.strip(),
    }


def build_structured_trace(messages):
    if isinstance(messages, dict):
        return {
            "status": "error",
            "error": messages.get("error", str(messages)),
            "messages": [],
            "turns": [],
            "final_answer": None,
        }

    simple_messages = []
    turns = []
    current_turn = None

    for idx, message in enumerate(messages):
        role = message.get("role")
        text = content_to_text(message.get("content", ""))
        simple_messages.append(
            {
                "index": idx,
                "role": role,
                "text": text,
                "raw_content": message.get("content"),
            }
        )

        if role == "assistant":
            reflection = _extract_tagged_section(text, "REFLECTION")
            thought = _extract_tagged_section(text, "THOUGHT")
            action = _extract_tagged_section(text, "ACTION")
            answer = extract_answer_text(text)
            if reflection or thought or action or answer:
                current_turn = {
                    "assistant_message_index": idx,
                    "reflection": reflection,
                    "thought": thought,
                    "action": {
                        **action,
                        "code": extract_code_block(action["text"]) if action else None,
                    } if action else None,
                    "answer": answer,
                }
                turns.append(current_turn)
        elif role == "user" and current_turn is not None:
            observation = _extract_tagged_section(text, "OBSERVATION")
            if observation:
                current_turn["observation"] = observation["text"]
                current_turn["user_message_index"] = idx

    final_answer = None
    for message in reversed(simple_messages):
        if message["role"] != "assistant":
            continue
        final_answer = extract_answer_text(message["text"])
        if final_answer:
            break

    return {
        "status": "ok",
        "messages": simple_messages,
        "turns": turns,
        "final_answer": final_answer,
    }
