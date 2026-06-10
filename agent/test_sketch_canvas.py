"""
test_sketch_canvas.py  —  Multi-step iterative reasoning with sketch_canvas

Agent loop (ReACT style):
  Each turn the agent can call ONE of three tools:
    zoom_crop(x, y, w, h)       — crop a normalized bbox from the last image,
                                   saves crop to /tmp, returns path + new image id
    add_to_canvas(new_items)    — append annotation items to the growing canvas
                                   and re-render the HTML
    final_answer(text)          — record the answer and end the loop

  The canvas accumulates across steps: each step only adds new items.
  After every add_to_canvas call the HTML is re-rendered and the agent is shown
  a compact summary of what is currently on the canvas so it can plan next step.

Run: conda run -n sketchpad python agent/test_sketch_canvas.py
"""

import base64, json, os, sys, textwrap
sys.path.insert(0, os.path.dirname(__file__))

from openai import OpenAI
from sketch_canvas import sketch_canvas

# ── API config ────────────────────────────────────────────────────────────────
API_KEY  = "sk-IEh7KU9A1qT3FYuKe8Ak0RNb3fc4AjHkguBZYwJN0OBxv3Sj"
BASE_URL = "https://api.kksj.org/v1"
MODEL    = "gpt-5.1"
MAX_STEPS = 6

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ── tool schemas ──────────────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "zoom_crop",
            "description": (
                "Crop a region from the current image using normalized coordinates "
                "[0,1]. The crop is saved as a new file and a new image id is assigned. "
                "Use this to zoom in on a specific area for closer inspection."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "Left edge, normalized [0,1]"},
                    "y": {"type": "number", "description": "Top edge, normalized [0,1]"},
                    "w": {"type": "number", "description": "Width, normalized [0,1]"},
                    "h": {"type": "number", "description": "Height, normalized [0,1]"},
                    "label": {"type": "string", "description": "Short label for the cropped region"},
                },
                "required": ["x", "y", "w", "h", "label"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_canvas",
            "description": (
                "Append new annotation items to the shared reasoning canvas and re-render it. "
                "The canvas is cumulative — previous items are preserved. "
                "Use image items to place images, arrow to connect them, bbox to highlight "
                "regions, text to add notes. "
                "Canvas is 960×660 px. First image is typically at x=20,y=80,width=420. "
                "Second image (zoom crop) at x=500,y=80,width=380."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "new_items": {
                        "type": "array",
                        "description": (
                            "New items to add. Allowed types ONLY: "
                            "image {type,id,path,x,y,width,panel,panel_title,panel_color}, "
                            "arrow {type,x1,y1,x2,y2,style,color,label,curve}, "
                            "text  {type,x,y,content,size,bg,bold}. "
                            "Do NOT use bbox or circle — spatial coordinates are unreliable."
                        ),
                        "items": {"type": "object"},
                    },
                    "step_note": {
                        "type": "string",
                        "description": "One-line description of what this step is doing.",
                    },
                },
                "required": ["new_items", "step_note"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Record the final answer and end the reasoning loop.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "The final answer to the question."},
                    "reasoning": {"type": "string", "description": "Brief explanation of how you reached the answer."},
                },
                "required": ["answer", "reasoning"],
            },
        },
    },
]

# ── helpers ───────────────────────────────────────────────────────────────────

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def image_content(path: str) -> dict:
    ext  = os.path.splitext(path)[1].lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg"}.get(ext, ext) or "png"
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/{mime};base64,{encode_image(path)}"},
    }


def do_zoom_crop(source_path: str, x: float, y: float,
                 w: float, h: float, label: str, crop_index: int) -> str:
    from PIL import Image as PILImage
    img = PILImage.open(source_path)
    iw, ih = img.size
    left   = int(x * iw)
    top    = int(y * ih)
    right  = int((x + w) * iw)
    bottom = int((y + h) * ih)
    crop   = img.crop((left, top, right, bottom))
    out    = f"/tmp/zoom_crop_{crop_index}.png"
    crop.save(out)
    return out


def canvas_summary(items: list) -> str:
    counts = {}
    for it in items:
        t = it.get("type", "?")
        counts[t] = counts.get(t, 0) + 1
    parts = [f"{v} {k}(s)" for k, v in counts.items()]
    return "Canvas currently has: " + ", ".join(parts) if parts else "Canvas is empty."


# ── main reasoning loop ───────────────────────────────────────────────────────

def run_iterative(image_path: str, question: str, output_prefix: str, max_steps: int = MAX_STEPS) -> str:
    print(f"\n{'='*65}")
    print(f"Image   : {image_path}")
    print(f"Question: {question}")
    print(f"{'='*65}")

    canvas_items: list  = []
    crop_counter: int   = 0
    # track images added so zoom_crop knows the latest image
    images_on_canvas: list = []  # list of (id, path)
    html_path = f"/tmp/{output_prefix}_canvas.html"

    system_msg = {
        "role": "system",
        "content": (
            "You are a careful visual reasoning assistant working step by step.\n"
            "You have a shared sketch canvas (960×660 px) that accumulates across steps.\n"
            "Reasoning strategy:\n"
            "  1. First call add_to_canvas to place the original image (x=20,y=60,width=400).\n"
            "  2. For EVERY piece or region you want to identify: call zoom_crop to cut it out,\n"
            "     then add_to_canvas to place the crop beside the overview and add a text label.\n"
            "     The crop image IS the evidence — do not just write text without cropping first.\n"
            "  3. Use arrows to connect each crop back to the overview image.\n"
            "  4. Only call final_answer when all evidence is on the canvas.\n"
            "Layout guide: place crops in a grid to the right of the overview.\n"
            "  First crop:  x=440, y=60,  width=120\n"
            "  Second crop: x=580, y=60,  width=120\n"
            "  Third crop:  x=720, y=60,  width=120\n"
            "  Fourth crop: x=440, y=220, width=120\n"
            "  Text labels go just below each crop (y = crop_y + crop_height + 10).\n"
            "Only use image, arrow, and text items — do NOT use bbox or circle.\n"
            f"The original image path is: {image_path}"
        ),
    }

    messages = [
        system_msg,
        {
            "role": "user",
            "content": [
                image_content(image_path),
                {"type": "text", "text": question},
            ],
        },
    ]

    final_answer_text = None

    for step in range(1, max_steps + 1):
        print(f"\n── Step {step} ──────────────────────────────────────────────")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="required",
            max_tokens=1024,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            print("  (no tool call — stopping)")
            break

        # append assistant message first (may have multiple tool calls)
        messages.append(msg)

        done = False
        # process ALL tool calls in this response, collecting result messages
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            print(f"  Tool: {name}")

            # ── zoom_crop ─────────────────────────────────────────────────────
            if name == "zoom_crop":
                src = images_on_canvas[-1][1] if images_on_canvas else image_path
                crop_counter += 1
                crop_path = do_zoom_crop(
                    src,
                    args["x"], args["y"], args["w"], args["h"],
                    args["label"], crop_counter,
                )
                crop_id = f"crop_{crop_counter}"
                images_on_canvas.append((crop_id, crop_path))
                print(f"    Cropped {args['label']} → {crop_path}")
                tool_result = json.dumps({
                    "status": "ok",
                    "crop_id": crop_id,
                    "crop_path": crop_path,
                    "message": (
                        f"Crop saved as id='{crop_id}', path='{crop_path}'. "
                        "Now call add_to_canvas to place it on the canvas."
                    ),
                })

            # ── add_to_canvas ─────────────────────────────────────────────────
            elif name == "add_to_canvas":
                note      = args.get("step_note", "")
                new_items = args.get("new_items", [])
                for it in new_items:
                    if it.get("type") == "image":
                        images_on_canvas.append((it.get("id", ""), it.get("path", "")))
                canvas_items.extend(new_items)
                canvas_path = sketch_canvas(
                    canvas_items,
                    output_path=html_path,
                    title=f"Step-by-step reasoning — {output_prefix}",
                )
                size = os.path.getsize(canvas_path)
                print(f"    Note  : {note}")
                print(f"    Added : {len(new_items)} item(s) → total {len(canvas_items)}")
                print(f"    Canvas: {canvas_path}  ({size} bytes)")
                tool_result = json.dumps({
                    "status": "ok",
                    "canvas_path": canvas_path,
                    "canvas_summary": canvas_summary(canvas_items),
                    "message": "Canvas updated. Continue reasoning or call final_answer.",
                })

            # ── final_answer ──────────────────────────────────────────────────
            elif name == "final_answer":
                final_answer_text = args["answer"]
                print(f"    Answer   : {final_answer_text}")
                print(f"    Reasoning: {textwrap.fill(args['reasoning'], 58)}")
                tool_result = json.dumps({"status": "ok"})
                done = True

            else:
                tool_result = json.dumps({"status": "error", "message": f"Unknown tool: {name}"})

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_result,
            })

        if done:
            break

    print(f"\nFinal canvas : {html_path}")
    print(f"Final answer : {final_answer_text}")
    return html_path


# ── test cases ────────────────────────────────────────────────────────────────

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(AGENT_DIR)

TEST_CASES = [
    {
        "image":  os.path.join(ROOT_DIR, "截屏2026-06-09 23.07.30.png"),
        "question": (
            "What percentage of people found home working very difficult? "
            "Zoom in to find the exact number, annotate your steps on the canvas."
        ),
        "prefix": "infographic",
    },
    {
        "image":  os.path.join(AGENT_DIR, "board.png"),
        "question": (
            "Count the total number of white pieces on the board. "
            "Annotate each piece you find on the canvas step by step."
        ),
        "prefix": "chess",
        "max_steps": 12,
    },
]

if __name__ == "__main__":
    outputs = []
    for tc in TEST_CASES:
        if not os.path.exists(tc["image"]):
            print(f"SKIP — image not found: {tc['image']}")
            continue
        path = run_iterative(tc["image"], tc["question"], tc["prefix"], tc.get("max_steps", MAX_STEPS))
        outputs.append(path)

    print(f"\n{'='*65}")
    print(f"Done. HTML canvases:")
    for p in outputs:
        print(f"  {p}")

    if outputs:
        import subprocess
        subprocess.run(["open"] + outputs)
