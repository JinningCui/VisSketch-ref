"""
sketch_canvas.py  —  Spatial HTML/SVG annotation canvas for VisualSketchpad agent.

Lets the agent place multiple images on a shared canvas, connect them with
arrows, draw bounding-box overlays, add labels and callout notes — closer to
a human sketch-pad workflow than plain image-text interleaving.

Quick-start
-----------
from sketch_canvas import sketch_canvas

path = sketch_canvas([
    # place original image with a panel border
    {"type": "image", "id": "orig", "path": "image_1.jpg",
     "x": 20, "y": 80, "width": 420,
     "panel": True, "panel_title": "Original"},

    # place a zoomed crop beside it
    {"type": "image", "id": "zoom", "path": "zoom_region.png",
     "x": 500, "y": 80, "width": 340,
     "panel": True, "panel_color": "#ffe4e1", "panel_title": "Zoom"},

    # dashed arrow from a point on the original to the zoomed panel
    {"type": "arrow", "x1": 440, "y1": 220, "x2": 500, "y2": 220,
     "style": "dashed", "color": "#e11d48"},

    # red bounding box in relative coords on the zoom image
    {"type": "bbox", "image_id": "zoom", "rel": [0.05, 0.35, 0.55, 0.30],
     "color": "#e11d48", "label": "target region"},

    # text note at the bottom
    {"type": "text", "x": 20, "y": 520,
     "content": "The 22 % figure is inside the left donut chart.",
     "size": 13, "bg": "#fef9c3"},
], output_path="sketch_step1.html")

print(f"SKETCH_PATH: {path}")
"""

import base64
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── image helpers ────────────────────────────────────────────────────────────

def _mime(path: str) -> str:
    return {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png",  ".gif": "image/gif",
            ".webp": "image/webp", ".svg": "image/svg+xml"
            }.get(Path(path).suffix.lower(), "image/png")


def _data_uri(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{_mime(path)};base64,{b64}"


def _img_natural_size(path: str):
    """Return (w, h) of image without PIL dependency (falls back to None)."""
    try:
        from PIL import Image as _PILImage
        with _PILImage.open(path) as im:
            return im.size
    except Exception:
        return None


# ── SVG primitives ───────────────────────────────────────────────────────────

def _esc(s: Any) -> str:
    import html
    return html.escape(str(s))


def _dash(style: str) -> str:
    return {"dashed": "stroke-dasharray='8 4'",
            "dotted": "stroke-dasharray='2 4'",
            "solid":  ""}.get(style, "")


def _arrow_marker(marker_id: str, color: str) -> str:
    return (
        f"<marker id='{marker_id}' markerWidth='10' markerHeight='7' "
        f"refX='10' refY='3.5' orient='auto'>"
        f"<polygon points='0 0, 10 3.5, 0 7' fill='{color}'/>"
        f"</marker>"
    )


def _curve_path(x1, y1, x2, y2, curve: float = 0.0) -> str:
    """Straight line (curve=0) or S-curve (curve != 0)."""
    if curve == 0:
        return f"M{x1},{y1} L{x2},{y2}"
    dx = abs(x2 - x1)
    cx1 = x1 + dx * curve
    cx2 = x2 - dx * curve
    return f"M{x1},{y1} C{cx1},{y1} {cx2},{y2} {x2},{y2}"


# ── main function ────────────────────────────────────────────────────────────

def sketch_canvas(
    items: List[Dict[str, Any]],
    output_path: str,
    width: int = 960,
    height: int = 660,
    bg_color: str = "#f8fafc",
    title: str = "",
    embed_images: bool = True,
) -> str:
    """
    Generate a spatial annotated HTML/SVG canvas and write it to *output_path*.

    Supported item types
    --------------------
    image
        id          str   unique identifier (used by bbox / arrow)
        path        str   file path to image
        x, y        int   top-left position on canvas
        width       int   display width in pixels (height auto-scaled)
        label       str   caption shown below the image  (default "")
        panel       bool  draw a coloured panel card behind the image
        panel_color str   panel background  (default "#e0e7ff")
        panel_title str   panel header text (default "")

    arrow
        x1,y1       int   start point
        x2,y2       int   end point
        style       str   "solid" | "dashed" | "dotted"  (default "solid")
        color       str   stroke colour  (default "#e11d48")
        width       int   stroke width  (default 2)
        label       str   optional mid-arrow label
        curve       float curvature 0–0.4  (default 0 = straight)

    bbox
        Canvas-absolute:
          x, y, w, h  int  top-left + size in pixels
        On-image relative (normalised 0–1):
          image_id    str  id of a previously declared image item
          rel         list [x, y, w, h] in [0,1] relative to image rect
        color         str  stroke colour  (default "#e11d48")
        label         str  optional label shown above the box
        line_width    int  (default 2)
        fill          str  fill colour (default "none"; try "rgba(255,0,0,0.08)")

    text
        x, y        int   anchor position
        content     str   text (newlines supported → tspan)
        size        int   font-size px  (default 13)
        color       str   (default "#1e293b")
        bold        bool  (default False)
        bg          str   background rect colour  (default "" = none)
        padding     int   bg rect padding  (default 6)

    circle
        cx, cy      int   centre
        r           int   radius  (default 18)
        color       str   stroke & label colour  (default "#e11d48")
        fill        str   (default "none")
        label       str   label inside the circle

    line
        x1,y1,x2,y2 int
        color       str   (default "#64748b")
        width       int   (default 1)
        style       str   "solid" | "dashed" | "dotted"

    Returns
    -------
    str  absolute path to the written HTML file.
    """
    output_path = str(Path(output_path).resolve())

    # ── pass 1: collect image registry (id → {x, y, w, h}) ─────────────────
    img_registry: Dict[str, Dict] = {}
    for item in items:
        if item.get("type") != "image":
            continue
        iid = item.get("id", "")
        path = item["path"]
        iw = int(item["width"])
        nat = _img_natural_size(path)
        if nat:
            scale = iw / nat[0]
            ih = int(nat[1] * scale)
        else:
            ih = int(iw * 0.75)
        img_registry[iid] = {"x": item["x"], "y": item["y"], "w": iw, "h": ih}

    # ── pass 2: build SVG elements ───────────────────────────────────────────
    defs_markers: Dict[str, str] = {}   # color → marker_id
    panels_svg = []
    images_svg = []
    draw_svg   = []      # arrows, bboxes, circles, lines
    labels_svg = []      # text annotations (rendered last → on top)

    marker_counter = [0]

    def _get_marker(color: str) -> str:
        if color not in defs_markers:
            mid = f"arr{marker_counter[0]}"
            marker_counter[0] += 1
            defs_markers[color] = mid
        return defs_markers[color]

    PANEL_PAD = 8   # px padding around image inside panel
    LABEL_H   = 22  # px for caption below image

    for item in items:
        t = item.get("type", "")

        # ── image ────────────────────────────────────────────────────────────
        if t == "image":
            iid   = item.get("id", "")
            ipath = item["path"]
            x, y  = int(item["x"]), int(item["y"])
            reg   = img_registry.get(iid, {})
            iw, ih = reg.get("w", int(item["width"])), reg.get("h", int(item["width"]) * 3 // 4)

            src = _data_uri(ipath) if embed_images else _esc(os.path.abspath(ipath))
            label_text = _esc(item.get("label", ""))

            if item.get("panel", False):
                pc    = _esc(item.get("panel_color", "#e0e7ff"))
                ptitle = _esc(item.get("panel_title", ""))
                px    = x - PANEL_PAD
                py    = y - (LABEL_H + PANEL_PAD) if ptitle else y - PANEL_PAD
                pw    = iw + PANEL_PAD * 2
                ph    = ih + PANEL_PAD * 2 + (LABEL_H if ptitle else 0)
                panels_svg.append(
                    f"<rect x='{px}' y='{py}' width='{pw}' height='{ph}' "
                    f"rx='8' fill='{pc}' stroke='#94a3b8' stroke-width='1.5'/>"
                )
                if ptitle:
                    panels_svg.append(
                        f"<text x='{px+pw//2}' y='{py+LABEL_H-5}' "
                        f"text-anchor='middle' font-size='13' font-weight='bold' "
                        f"fill='#334155'>{ptitle}</text>"
                    )

            images_svg.append(
                f"<image href='{src}' x='{x}' y='{y}' "
                f"width='{iw}' height='{ih}' preserveAspectRatio='xMidYMid meet'/>"
            )
            if label_text:
                labels_svg.append(
                    f"<text x='{x + iw // 2}' y='{y + ih + 16}' "
                    f"text-anchor='middle' font-size='12' fill='#475569'>{label_text}</text>"
                )

        # ── arrow ────────────────────────────────────────────────────────────
        elif t == "arrow":
            x1, y1 = int(item["x1"]), int(item["y1"])
            x2, y2 = int(item["x2"]), int(item["y2"])
            color  = _esc(item.get("color", "#e11d48"))
            style  = item.get("style", "solid")
            lw     = int(item.get("width", 2))
            curve  = float(item.get("curve", 0.0))
            mid    = _get_marker(color)
            dash   = _dash(style)
            path_d = _curve_path(x1, y1, x2, y2, curve)
            draw_svg.append(
                f"<path d='{path_d}' stroke='{color}' stroke-width='{lw}' "
                f"fill='none' {dash} marker-end='url(#{mid})'/>"
            )
            if item.get("label"):
                mx = (x1 + x2) // 2
                my = (y1 + y2) // 2 - 6
                labels_svg.append(
                    f"<text x='{mx}' y='{my}' text-anchor='middle' "
                    f"font-size='11' fill='{color}'>{_esc(item['label'])}</text>"
                )

        # ── bbox ─────────────────────────────────────────────────────────────
        elif t == "bbox":
            color = _esc(item.get("color", "#e11d48"))
            lw    = int(item.get("line_width", 2))
            fill  = _esc(item.get("fill", "none"))

            if "image_id" in item and "rel" in item:
                reg = img_registry.get(item["image_id"])
                if reg is None:
                    continue
                rx, ry, rw, rh = item["rel"]
                bx = int(reg["x"] + rx * reg["w"])
                by = int(reg["y"] + ry * reg["h"])
                bw = int(rw * reg["w"])
                bh = int(rh * reg["h"])
            else:
                bx = int(item["x"])
                by = int(item["y"])
                bw = int(item["w"])
                bh = int(item["h"])

            draw_svg.append(
                f"<rect x='{bx}' y='{by}' width='{bw}' height='{bh}' "
                f"fill='{fill}' stroke='{color}' stroke-width='{lw}' rx='2'/>"
            )
            if item.get("label"):
                lbl = _esc(item["label"])
                lbl_w = min(len(item["label"]) * 7 + 8, bw + 20)
                labels_svg.append(
                    f"<rect x='{bx}' y='{by-18}' width='{lbl_w}' "
                    f"height='16' rx='3' fill='{color}' opacity='0.85'/>"
                    f"<text x='{bx+4}' y='{by-5}' font-size='11' "
                    f"fill='white' font-weight='bold'>{lbl}</text>"
                )

        # ── text ─────────────────────────────────────────────────────────────
        elif t == "text":
            x, y    = int(item["x"]), int(item["y"])
            content = item.get("content", "")
            size    = int(item.get("size", 13))
            color   = _esc(item.get("color", "#1e293b"))
            bold    = "bold" if item.get("bold") else "normal"
            bg      = item.get("bg", "")
            pad     = int(item.get("padding", 6))
            lines   = content.split("\n")
            line_h  = size + 4

            if bg:
                bw = max(len(l) for l in lines) * (size * 0.6) + pad * 2
                bh = len(lines) * line_h + pad * 2
                labels_svg.append(
                    f"<rect x='{x - pad}' y='{y - size - pad}' "
                    f"width='{int(bw)}' height='{int(bh)}' rx='5' "
                    f"fill='{_esc(bg)}' stroke='#e2e8f0' stroke-width='1'/>"
                )
            tspans = "".join(
                f"<tspan x='{x}' dy='{0 if i == 0 else line_h}'>{_esc(l)}</tspan>"
                for i, l in enumerate(lines)
            )
            labels_svg.append(
                f"<text x='{x}' y='{y}' font-size='{size}' "
                f"font-weight='{bold}' fill='{color}'>{tspans}</text>"
            )

        # ── circle ───────────────────────────────────────────────────────────
        elif t == "circle":
            cx     = int(item["cx"])
            cy     = int(item["cy"])
            r      = int(item.get("r", 18))
            color  = _esc(item.get("color", "#e11d48"))
            fill   = _esc(item.get("fill", "none"))
            draw_svg.append(
                f"<circle cx='{cx}' cy='{cy}' r='{r}' "
                f"fill='{fill}' stroke='{color}' stroke-width='2'/>"
            )
            if item.get("label"):
                labels_svg.append(
                    f"<text x='{cx}' y='{cy + 5}' text-anchor='middle' "
                    f"font-size='12' font-weight='bold' fill='{color}'>"
                    f"{_esc(item['label'])}</text>"
                )

        # ── line ─────────────────────────────────────────────────────────────
        elif t == "line":
            color = _esc(item.get("color", "#64748b"))
            lw    = int(item.get("width", 1))
            dash  = _dash(item.get("style", "solid"))
            draw_svg.append(
                f"<line x1='{item['x1']}' y1='{item['y1']}' "
                f"x2='{item['x2']}' y2='{item['y2']}' "
                f"stroke='{color}' stroke-width='{lw}' {dash}/>"
            )

    # ── assemble SVG ─────────────────────────────────────────────────────────
    defs_block = "<defs>" + "".join(
        _arrow_marker(mid, color) for color, mid in defs_markers.items()
    ) + "</defs>"

    title_block = ""
    if title:
        title_block = (
            f"<text x='{width // 2}' y='28' text-anchor='middle' "
            f"font-size='16' font-weight='bold' fill='#0f172a'>{_esc(title)}</text>"
        )
        ty_offset = 44
    else:
        ty_offset = 0

    all_elements = (
        [f"<rect width='{width}' height='{height}' fill='{bg_color}'/>"]
        + ([title_block] if title_block else [])
        + panels_svg
        + images_svg
        + draw_svg
        + labels_svg
    )

    svg = (
        f"<svg xmlns='http://www.w3.org/2000/svg' "
        f"xmlns:xlink='http://www.w3.org/1999/xlink' "
        f"width='{width}' height='{height}' "
        f"style='font-family:Arial,sans-serif;'>"
        + defs_block
        + "\n".join(all_elements)
        + "</svg>"
    )

    html_out = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{_esc(title or "Sketch Canvas")}</title>
  <style>
    body {{ margin: 0; background: #e2e8f0; display: flex;
            justify-content: center; padding: 24px; }}
    .canvas-wrap {{ background: white; border-radius: 12px;
                    box-shadow: 0 4px 24px rgba(0,0,0,0.12);
                    padding: 16px; display: inline-block; }}
  </style>
</head>
<body>
  <div class="canvas-wrap">
    {svg}
  </div>
</body>
</html>"""

    Path(output_path).write_text(html_out, encoding="utf-8")

    # Save a JSON sidecar so memory_agent can read the canvas structure
    # without parsing HTML.  File: <output_path>.items.json
    import json as _json
    sidecar = {
        "canvas_path": output_path,
        "width": width,
        "height": height,
        "title": title,
        "items": items,
    }
    sidecar_path = output_path + ".items.json"
    Path(sidecar_path).write_text(_json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")

    return output_path
