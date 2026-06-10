"""
test_memory_canvas.py — Chess reasoning with memory canvas.

Task: given a chess board image, determine who is winning and annotate
the key move / winning piece with arrows directly on the evidence image.

The model may:
  - Answer in one step if the position is clear.
  - Call zoom_crop to inspect a region before deciding.

Memory stores only the retained evidence with arrows showing the key move.

Run: conda run -n sketchpad python agent/test_memory_canvas.py
"""

import base64, json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
from openai import OpenAI
from PIL import Image as PILImage

from memory_agent import MemoryOrganizerAgent

API_KEY  = "sk-IEh7KU9A1qT3FYuKe8Ak0RNb3fc4AjHkguBZYwJN0OBxv3Sj"
BASE_URL = "https://api.kksj.org/v1"
MODEL    = "gpt-4o-2024-11-20"

AGENT_DIR = Path(__file__).parent
BOARD_IMG = AGENT_DIR / "board.png"

CHESS_SYSTEM = """You are a chess analyst with vision. You will be shown a chess board.

Your job:
1. Identify the current position — which side is winning (or checkmate/stalemate already).
2. If there is a forced checkmate or decisive move, identify it precisely.
3. Annotate your KEY finding on the board image using arrows:
   - arrows use normalized coordinates [0,1] from the TOP-LEFT of the image
   - A standard chessboard has 8×8 equal squares
   - Column a is leftmost (x=0..0.125), column h is rightmost (x=0.875..1.0)
   - Row 8 is at the top (y=0..0.125), row 1 is at the bottom (y=0.875..1.0)
   - Square center formula: x = (col_index + 0.5) / 8, y = (8 - row_num + 0.5) / 8
     where col_index: a=0, b=1, ..., h=7
   - Example: e1 center = ((4+0.5)/8, (8-1+0.5)/8) = (0.5625, 0.9375)
   - Example: e8 center = (0.5625, 0.0625)

When you have your answer, output a JSON memory object (do not wrap in markdown):
{
  "title": "Chess Position Analysis",
  "stage": "analysis_complete",
  "key_finding": "<one sentence: who wins and how>",
  "context": {
    "task": "Determine who is winning and annotate the key move.",
    "latest_observation": "<brief description of the position>"
  },
  "evidence": [
    {
      "id": "board",
      "type": "image",
      "path": "<board_image_path>",
      "caption": "Chess board — annotated",
      "notes": ["<key observation about the position>"],
      "arrows": [
        {
          "from_xy": [<x>, <y>],
          "to_xy": [<x>, <y>],
          "label": "<e.g. Rook e1 to e8 — checkmate>",
          "color": "#e11d48"
        }
      ]
    }
  ],
  "notes": {
    "open_checks": ["<any uncertainty about the position>"]
  }
}

If you need to zoom into a region first, say so briefly before the JSON.
You may include a zoom_crop request like:
ZOOM: x=0.0, y=0.0, w=0.5, h=0.5, label=top-left
(one crop only — then provide the full JSON analysis)
"""


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def image_msg(path: Path) -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{encode_image(path)}"},
    }


def do_crop(src: Path, x: float, y: float, w: float, h: float, out: Path) -> Path:
    img = PILImage.open(src)
    iw, ih = img.size
    crop = img.crop((int(x*iw), int(y*ih), int((x+w)*iw), int((y+h)*ih)))
    crop.save(out)
    return out


def run_chess_analysis(board_path: Path, tmp: Path) -> dict:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    agent  = MemoryOrganizerAgent(
        memory_format="html",
        llm_client=client,
        llm_model=MODEL,
        working_dir=tmp,
        enable_grounding=False,
    )

    messages = [
        {"role": "system", "content": CHESS_SYSTEM},
        {
            "role": "user",
            "content": [
                image_msg(board_path),
                {"type": "text", "text": (
                    f"Board image path: {board_path}\n"
                    "Analyze this chess position. Who is winning? "
                    "Mark the key piece or move with an arrow on the board. "
                    "Output your JSON memory object."
                )},
            ],
        },
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=1500,
        temperature=0.0,
    )
    raw = response.choices[0].message.content
    print("\n── Model response ────────────────────────────────")
    print(raw[:800])

    crop_path = None
    # Handle optional zoom request
    if raw.strip().startswith("ZOOM:"):
        zoom_line = raw.splitlines()[0]
        parts_kv = {}
        for part in zoom_line.replace("ZOOM:", "").split(","):
            k, v = part.strip().split("=")
            parts_kv[k.strip()] = v.strip()
        crop_path = tmp / "zoom_crop.png"
        do_crop(
            board_path,
            float(parts_kv["x"]), float(parts_kv["y"]),
            float(parts_kv["w"]), float(parts_kv["h"]),
            crop_path,
        )
        print(f"  → Cropped region: {crop_path}")
        # Second pass with crop
        rest_json = "\n".join(raw.splitlines()[1:]).strip()
        if not rest_json:
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": [
                    image_msg(crop_path),
                    {"type": "text", "text": (
                        f"Cropped region path: {crop_path}\n"
                        "Now provide your complete JSON memory object."
                    )},
                ],
            })
            response2 = client.chat.completions.create(
                model=MODEL, messages=messages, max_tokens=1500, temperature=0.0,
            )
            raw = response2.choices[0].message.content
            print("\n── Model response (after crop) ────────────────")
            print(raw[:800])

    # Parse JSON from response (may be embedded after prose)
    import re
    json_match = re.search(r'\{[\s\S]+\}', raw)
    payload = {}
    if json_match:
        try:
            payload = json.loads(json_match.group())
        except Exception:
            payload = {}

    if not payload:
        print("  ⚠ Could not parse JSON from model response")
        payload = {
            "title": "Chess Position Analysis",
            "stage": "analysis_complete",
            "key_finding": "Analysis complete — see model response",
            "context": {"task": "Who is winning?", "latest_observation": raw[:300]},
            "evidence": [{
                "id": "board", "type": "image",
                "path": str(board_path),
                "caption": "Chess board",
                "notes": [raw[:200]],
                "arrows": [],
            }],
        }

    generated = [str(board_path)]
    if crop_path:
        generated.append(str(crop_path))

    record = agent.update(
        task_prompt="Determine who is winning and annotate the key move.",
        assistant_message=raw,
        observation_text=f"Analysis complete. key_finding: {payload.get('key_finding', '')}",
        generated_files=generated,
        stage=payload.get("stage", "analysis_complete"),
    )
    return record


def run():
    if not BOARD_IMG.exists():
        print(f"SKIP — board.png not found at {BOARD_IMG}")
        return

    tmp = Path(tempfile.mkdtemp())
    print(f"Working dir: {tmp}")

    record = run_chess_analysis(BOARD_IMG, tmp)

    html = Path(record["path"]).read_text()
    canonical = record.get("canonical_payload") or {}

    print(f"\n── Memory output ────────────────────────────────")
    print(f"  Path        : {record['path']}")
    print(f"  key_finding : {canonical.get('key_finding', '')}")
    arrows_total = sum(
        len(e.get("arrows") or [])
        for e in canonical.get("evidence", [])
        if isinstance(e, dict)
    )
    print(f"  Arrow annotations : {arrows_total}")
    print(f"  SVG present : {'<svg' in html}")
    print(f"  Base64 imgs : {html.count('<image href=')}")

    import subprocess
    subprocess.run(["open", record["path"]])


if __name__ == "__main__":
    run()
