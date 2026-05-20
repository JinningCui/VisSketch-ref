# html-sketch-main Task Execution README

This project follows the VisualSketchpad-style ReACT + Jupyter loop, with an added dynamic memory agent for comparing memory organizations:

- `image`: a single linear visual memory image.
- `html`: an inspectable HTML memory page that can organize text, structure, and evidence images.
- `json`: a structured JSON memory object.

The task-solving reasoner is kept close to the original VisualSketchpad behavior. It still writes Python actions, receives Jupyter observations, and eventually returns `ANSWER: ... TERMINATE`. HTML/JSON are no longer forced into the reasoner's action format.

## Core Idea

The experiment treats the visible multimodal reasoning process as dynamic external memory. A separate Memory Organizer Agent observes each visible step and maintains a memory artifact. The reasoner can use this artifact on later turns, but the reasoner is not required to write HTML or JSON itself.

This separates two variables:

- the task-solving prompt and tool loop
- the external memory representation

That makes the comparison between HTML, JSON, and image memory cleaner than rewriting the solver prompt with an HTML/JSON draft contract.

## Execution Flow

```text
Task input
  |
  v
run_agent(...)
  |
  |-- select original task prompt
  |     vision -> ReACTPrompt
  |     math   -> MathPrompt(task_name)
  |     geo    -> GeoPrompt
  |
  v
Reasoner LLM
  |
  |  writes REFLECTION / THOUGHT / ACTION
  |  ACTION contains exactly one ```python code block unless final answer
  v
Parser
  |
  |-- accepts exactly one Python block
  |-- rejects missing, empty, or multiple Python blocks
  v
Jupyter CodeExecutor
  |
  |-- executes Python in the task working directory
  |-- can generate images, plots, overlays, HTML, JSON, or text outputs
  v
Memory Organizer Agent
  |
  |-- reads visible assistant message, observation, generated files, and previous memory
  |-- writes memory_step_000.html / .json / .png
  |-- appends the memory artifact back to the next reasoner observation
  v
Reasoner continues
  |
  |-- uses task prompt, execution observation, and dynamic memory
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

- `agent/main.py`: entrypoint for `run_agent`; selects task prompt, executor, memory agent, and output saving.
- `agent/agent.py`: `SketchpadUserAgent`; parses code, executes code, calls the memory agent, and sends observations.
- `agent/memory_agent.py`: maintains external HTML/JSON/image memory artifacts.
- `agent/prompt.py`: original task prompts and legacy draft prompt classes.
- `agent/parse.py`: extracts exactly one executable Python block.
- `agent/execution.py`: Jupyter-backed Python execution environment.
- `agent/utils.py`: structured trace construction and final answer extraction.
- `agent/run_task.py`: batch runner for benchmark tasks.
- `agent/evaluate_tasks.py`: normalizes and evaluates final answers.

## Memory Modes

Run with the new argument:

```bash
cd agent
python run_task.py --task geometry --memory-format html
```

The old `--draft-format html/json` flag is kept as a backward-compatible alias for `--memory-format html/json`, but it no longer wraps or rewrites the reasoner's prompt.

Outputs are written under:

```text
outputs/.../memory_html
outputs/.../memory_json
outputs/.../memory_image
```

Each task instance also contains:

```text
memory_step_000.html/json/png
memory_step_001.html/json/png
memory_index.json
```

### HTML Memory

HTML memory is intended to be the richest dynamic memory format. It can organize:

- task state
- objects and variables
- relations and constraints
- derived facts
- generated images and captions
- revision history
- open issues
- next-action hints

The reasoner receives the HTML memory source and any image references embedded in it as part of the next observation.

### JSON Memory

JSON memory stores similar content in machine-readable fields:

- `objects_variables`
- `relations_constraints`
- `derived_facts`
- `retained_state`
- `revisions`
- `open_issues`
- `next_action_hints`
- `evidence_files`

This is useful for symbolic and graph-like tasks, where strict fields can reduce ambiguity.

### Image Memory

Image memory renders a single linear PNG summary board. It is a useful baseline for comparing against VisualSketchpad-style image-only memory, but it is weaker at preserving structured, revisable state.

## Answer Format

The final answer must be plain text outside a code block:

```text
ANSWER: concrete_final_value TERMINATE
```

The extraction logic:

- ignores anything inside code blocks
- requires both `ANSWER:` and `TERMINATE`
- filters placeholders such as `{answer}`, `<answer>`, `[answer]`, `final reasoning result`, and `concrete_final_value`

Task-specific prompts further constrain final answers, for example:

- `winner_id`: `white`, `black`, or `draw`
- `graph_isomorphism`: `true` or `false`
- `graph_connectivity`: starts with `yes` or `no`
- `graph_maxflow`: a concrete maximum-flow value
- `math_convexity`: `convex` or `concave`
- `math_parity`: `even`, `odd`, or `neither`

## Experimental Notes

For fair comparison between image, HTML, and JSON memory formats, keep these variables aligned:

- same model
- same task set
- same max turns
- same Python/Jupyter tools
- same vision-tool availability
- same memory update schedule
- same final answer normalization and evaluation

HTML memory should be expected to help most on multimodal tasks that require preserving heterogeneous evidence: geometry diagrams, generated auxiliary-line images, real-scene QA, counting with crops/detections, board states, and multi-image evidence. JSON may remain stronger on purely symbolic graph tasks.
