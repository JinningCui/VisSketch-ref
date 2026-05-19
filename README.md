# html-sketch-main Task Execution README

This project extends the VisualSketchpad-style ReACT + Jupyter loop to compare intermediate visual draft formats for multimodal reasoning:

- `image`: a pure visual sketch baseline, typically generated or displayed by Python/Jupyter.
- `html`: a dynamic HTML visual draft that can organize text, structure, and one or more evidence images.
- `json`: a dynamic declarative JSON draft that can organize the same evidence in machine-readable fields.

The core execution loop remains close to `VisualSketchpad-main`: an LLM planner writes Python actions, the user proxy executes them in Jupyter, and observations are fed back until the planner returns `ANSWER: ... TERMINATE`.

## Core Idea

HTML and JSON do not create independent visual evidence by themselves. Images are still produced or transformed by Python in the Jupyter executor:

- input images loaded as `image_1`, `image_2`, ...
- derived crops, overlays, auxiliary lines, graph renderings, function plots, or chess boards saved by Python
- optional vision-tool outputs such as detection, segmentation, and depth maps

HTML/JSON then reference and organize these evidence images. This lets the experiment test whether a richer draft representation improves reasoning compared with pure image sketches.

## Execution Flow

```text
Task input
  |
  v
run_agent(...)
  |
  |-- select task prompt
  |     vision -> ReACTPrompt
  |     math   -> MathPrompt
  |     geo    -> GeoPrompt
  |     t2i_html -> HTMLVisualPrompt
  |     t2i_json -> JSONVisualPrompt
  |
  |-- optional draft wrapper
  |     --draft-format html -> dynamic HTML contract
  |     --draft-format json -> dynamic JSON contract
  |
  v
Planner LLM
  |
  |  writes REFLECTION / THOUGHT / ACTION
  |  ACTION must contain exactly one ```python code block
  v
Parser
  |
  |-- accepts exactly one Python block
  |-- rejects missing, empty, or multiple Python blocks
  v
Jupyter CodeExecutor
  |
  |-- executes Python in the task working directory
  |-- can generate images, HTML files, JSON files, plots, overlays
  v
Draft validation
  |
  |-- html mode: checks printed HTML_DRAFT_PATH and required HTML sections
  |-- json mode: checks printed JSON_DRAFT_PATH and required JSON keys
  |-- failure becomes feedback for the next loop
  v
Observation to Planner
  |
  |-- planner revises draft or answers
  v
Final answer extraction
  |
  |-- strips code blocks
  |-- extracts ANSWER: ... TERMINATE
  |-- filters placeholder answers
  v
Saved outputs
```

## Important Files

- `agent/main.py`: entrypoint for `run_agent`; selects task type, prompt, executor, and output saving.
- `agent/prompt.py`: prompt definitions for vision, math, geometry, HTML drafts, JSON drafts, and draft-format wrappers.
- `agent/agent.py`: `SketchpadUserAgent`; receives planner messages, parses code, executes code, validates drafts, and sends observations.
- `agent/parse.py`: extracts exactly one executable Python block.
- `agent/execution.py`: Jupyter-backed Python execution environment.
- `agent/draft_validation.py`: validates HTML/JSON draft files after execution.
- `agent/utils.py`: structured trace construction and final answer extraction.
- `agent/run_task.py`: batch runner for benchmark tasks.
- `agent/evaluate_tasks.py`: normalizes and evaluates final answers.

## Draft Modes

### Original VisualSketchpad-style mode

Without `--draft-format`, the agent follows the base task prompt. It can use Python/Jupyter to display or save visual sketches, plots, crops, overlays, and tool outputs.

This is closest to the original VisualSketchpad behavior.

### HTML draft mode

Run standard tasks with:

```bash
cd agent
python run_task.py --task geometry --draft-format html
```

Each non-final action must write a valid `.html` file and print:

```text
HTML_DRAFT_PATH: <path>
HTML_DRAFT_SUMMARY: <summary>
```

The HTML draft must contain visible sections labelled:

- Thinking
- Objects
- Relations
- State
- Revision
- Retained
- Referenced Images

HTML is intended to be the strongest visual draft format. It can organize text, tables, diagrams, inline SVG, board states, charts, and multiple evidence images in one inspectable workspace.

### JSON draft mode

Run standard tasks with:

```bash
cd agent
python run_task.py --task graph_maxflow --draft-format json
```

Each non-final action must write a valid `.json` file and print:

```text
JSON_DRAFT_PATH: <path>
JSON_DRAFT_SUMMARY: <summary>
```

The JSON root must include:

- `title`
- `original_prompt`
- `thinking_text`
- `objects_entities`
- `relations_constraints`
- `state_effects_view`
- `retained_context`
- `referenced_images`
- `open_issues_revision_targets`

JSON is intended to be a structured, machine-readable visual draft. It is weaker than HTML for direct visual layout, but better for parsing, automatic checks, and analysis.

## Dynamic Draft Memory

HTML/JSON drafts are treated as mutable memory across reasoning turns.

On each loop, the model should:

- keep useful content from the previous draft
- correct wrong content
- delete stale or irrelevant details
- add new evidence
- decide how many evidence images to reference, including zero, one, or many

Evidence images should come from the same task context and Jupyter toolchain, such as:

- original input image
- crop or zoomed region
- bounding-box or segmentation overlay
- auxiliary-line geometry sketch
- graph drawing or residual graph
- function plot
- chess board rendering

## Answer Format

The final answer must be plain text outside a code block:

```text
ANSWER: concrete_final_value TERMINATE
```

The extraction logic:

- ignores anything inside code blocks
- accepts only real answer text after `ANSWER:`
- filters placeholders such as `{answer}`, `<answer>`, `[answer]`, `final reasoning result`, and `concrete_final_value`

Task-specific prompts further constrain final answers, for example:

- `winner_id`: `white`, `black`, or `draw`
- `graph_isomorphism`: `true` or `false`
- `graph_connectivity`: starts with `yes` or `no`
- `math_convexity`: `convex` or `concave`
- `math_parity`: `even`, `odd`, or `neither`

## Suitable Task Types

This setup is most useful for tasks where intermediate visual state can be inspected or revised:

- geometry reasoning
- graph connectivity and maximum flow
- mathematical function properties
- chess or other board games
- real-scene image QA
- counting with detection/crop/overlay evidence

For real-scene QA and counting, HTML/JSON should not replace visual perception. They should organize evidence from original images, crops, detections, segmentations, and uncertainty notes.

## Experimental Notes

For fair comparison between image, HTML, and JSON draft formats, keep these variables aligned:

- same model
- same task set
- same max turns
- same Python/Jupyter tools
- same vision-tool availability
- same image-generation or evidence-image budget, if measuring cost-sensitive performance
- same final answer normalization and evaluation

HTML/JSON may organize multiple evidence images. This is a real representational advantage, not necessarily a confound, as long as the same evidence-generation tools and budget are available across conditions.

