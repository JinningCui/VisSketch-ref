# VisualSketchpad Agent Architecture

本文档描述当前 `VisualSketchpad-main/agent` 的实际执行逻辑。当前版本不是旧文档中的“四模块视觉推理系统”，而是以 VisualSketchpad 原始 ReACT/code-execution agent 为主体，并增加一个可选的外部动态 memory organization layer。

## 1. 总体目标

当前 agent 支持四类任务入口：

- `vision`: 图像 QA 类任务，读取 `request.json`，加载图像并可调用视觉工具。
- `math`: 数学、图、棋盘等符号任务，读取 `example.json`。
- `geo`: 几何任务，读取 `ex.json`。
- `t2i_html`: 文生图/结构化 HTML 草稿任务，读取 `request.json`、`prompt.json`、`example.json` 或 `ex.json`。

核心设计原则是：

- 保持主 Reasoner 的 prompt、parser、execution loop 与 VisualSketchpad baseline 一致。
- 通过 `--memory-format` 可选加入外部动态 memory，不改写主 Reasoner 的任务 prompt。
- memory agent 只组织可见推理摘要、代码动作、执行 observation、证据文件和错误状态，不独立解题。
- 对简单符号/数学/图任务使用 gating，避免无条件注入长 memory 导致上下文噪声。

## 2. 主执行流程

入口文件是 `agent/run_task.py`。它负责解析命令行参数、选择任务、设置输出目录，并按任务实例调用 `main.run_agent(...)`。

```text
run_task.py
  |
  |-- resolve_tasks(...)
  |-- iter_task_instances(...)
  |-- _run_one_instance(...)
        |
        v
main.run_agent(...)
  |
  |-- load task input
  |-- choose prompt generator
  |-- build Parser and CodeExecutor
  |-- build LLM runtime
  |-- create SketchpadUserAgent
  |-- create MultimodalConversableAgent planner
  |-- user.initiate_chat(planner, ...)
  |-- save output.json / structured_trace.json / prediction_summary.json
  |-- optional reflection update
```

## 3. Agent Loop

核心交互发生在 `agent/agent.py` 的 `SketchpadUserAgent.receive(...)`。

每一轮流程：

1. 接收 planner 的消息。
2. 使用 `Parser.parse(...)` 抽取代码块。
3. 如果没有代码且消息包含 `TERMINATE`，停止。
4. 如果解析失败，返回 parsing feedback。
5. 如果解析成功，使用 `CodeExecutor.execute(...)` 执行代码。
6. 根据执行结果返回 execution feedback。
7. 如果启用了 memory agent，在 parsing/execution feedback 后追加 memory summary 或 full memory。

简化流程图：

```text
Planner message
  |
  v
Parser.parse(message)
  |
  |-- no code + TERMINATE --> stop
  |
  |-- parse error
  |      |
  |      v
  |   parsing feedback
  |      |
  |      v
  |   optional memory update
  |
  |-- code parsed
         |
         v
      CodeExecutor.execute(code)
         |
         |-- error --> execution error feedback + optional memory update
         |
         |-- success --> execution success feedback + optional memory update
```

终止逻辑仍然使用 VisualSketchpad 原始的 `TERMINATE` 检查，不使用答案抽取结果控制 loop 停止。

## 4. Prompt 组件

Prompt 生成器位于 `agent/prompt.py`：

- `ReACTPrompt`: vision task 的默认 ReACT prompt。
- `MathPrompt`: graph/math/chess 等任务 prompt。
- `GeoPrompt`: geometry task prompt。
- `HTMLVisualPrompt`: `t2i_html` 任务 prompt。
- `MULTIMODAL_ASSISTANT_MESSAGE`: 默认 planner system message。
- `HTML_VISUAL_ASSISTANT_MESSAGE`: HTML visual draft 任务的 system message。

当前版本已经移除旧的 JSON draft wrapper 执行路径：

- 不再支持 `t2i_json` task type。
- 不再使用 `JSONVisualPrompt`。
- 不再使用 `--draft-format`。
- 不再保留 `draft_validation.py` 和 `run_json_reasoning.py`。

因此，HTML/JSON/image 的对比实验现在不是通过改写主 Reasoner prompt 来实现，而是通过独立 memory agent 的 `--memory-format` 实现。

## 5. Dynamic Memory Organization Layer

动态 memory 由 `agent/memory_agent.py` 中的 `MemoryOrganizerAgent` 实现。它在 `main.py` 中被注入到 `SketchpadUserAgent`：

```python
memory_agent=MemoryOrganizerAgent(
    memory_format,
    llm_runtime.client,
    task_directory,
    task_name=task_name,
)
```

这意味着 memory agent 默认使用与主 Reasoner 相同的 LLM client、API 配置和模型。例如运行时指定 `--backend api --model gpt-4o`，则 Reasoner 和 memory agent 都使用 `gpt-4o`。

### 5.1 Memory 输入

每次更新 memory 时，memory agent 接收：

- 初始任务 prompt。
- 当前 planner 的可见消息。
- parser 或 executor 返回的 observation/feedback。
- 本轮代码执行生成或引用的文件路径。
- 当前阶段：`parsing_error`、`execution_error`、`execution_success` 等。

### 5.2 Memory 输出格式

通过 `--memory-format` 控制：

- `html`: 生成 `memory_step_XXX.html`。
- `json`: 生成 `memory_step_XXX.json`。
- `image`: 生成 `memory_step_XXX.png`。
- `None`: 禁用 memory layer。

每轮还会维护：

- `memory_index.json`: 当前 case 的 memory artifact 索引。
- feedback 中的 `DYNAMIC MEMORY SUMMARY` 或 `DYNAMIC MEMORY UPDATE`。

### 5.3 Memory 内容约束

memory agent 的职责不是重新解题，而是维护外部、可检查的动态状态：

- evidence
- variables / objects
- relations / constraints
- derived facts
- revisions
- open issues
- next-action hints

当最新 observation 与旧 memory 冲突时，memory agent 应修正旧 memory：把旧错误声明放入 revisions，把仍不确定的问题放入 open issues，不把未验证内容提升为事实。

## 6. Gating 机制

为了避免 memory 对简单符号任务造成上下文噪声，`MemoryOrganizerAgent.should_inject_full(...)` 对部分任务启用 gating。

当前 gated tasks：

```text
graph_connectivity
graph_maxflow
graph_isomorphism
math_breakpoint
math_convexity
math_parity
```

对这些任务，只有满足以下条件之一时才注入 full memory：

- 任务已经超过第一步，memory 开始具备跨步状态价值。
- 本轮生成了 evidence files。
- 本轮出现 parsing/execution error。
- memory 中存在 open issues。

否则只注入一行式 memory summary，避免把简单问题的上下文变长。

非 gated tasks 默认注入 full memory。

## 7. 文件与模块职责

```text
agent/
  run_task.py
    批量任务入口；支持 --task、--workers、--backend、--model、--memory-format。

  main.py
    单个 case 的主入口；加载数据、创建 prompt/parser/executor/planner/user agent、保存输出。

  agent.py
    SketchpadUserAgent；负责 parse-execute-feedback loop，并在反馈中接入 memory。

  memory_agent.py
    外部动态 memory 组织层；生成 HTML/JSON/image memory artifact，并实现 gating。

  prompt.py
    主 Reasoner prompt 定义；保持 VisualSketchpad baseline 的任务思考逻辑。

  parse.py
    解析 planner 输出中的代码块。

  execution.py
    Jupyter/code execution 和文件路径追踪。

  llm_backend.py
    API/local backend 抽象。

  config.py
    backend、model、API key/base URL 和视觉工具地址配置。

  task_registry.py
    任务名、输入文件、label key、输出目录规则。

  utils.py
    trace 构建、answer 抽取、JSON 保存等工具函数。
```

## 8. 输出结构

`run_task.py` 默认把结果写入：

```text
outputs/<backend>/<model>/
```

如果启用 memory：

```text
outputs/<backend>/<model>/memory_<format>/
```

单个 case 目录中常见文件：

```text
output.json
full_trajectory.json
structured_trace.json
prediction_summary.json
usage_summary.json
run.log
memory_index.json              # 启用 memory 时
memory_step_000.html/json/png  # 启用 memory 时
```

这些都是运行产物，不应作为代码提交。

## 9. 运行示例

API backend：

```bash
cd agent
python run_task.py \
  --task graph_connectivity graph_maxflow \
  --backend api \
  --model gpt-4o \
  --base-url https://api.kksj.org/v1 \
  --api-key "$VISUAL_SKETCHPAD_API_KEY" \
  --memory-format html \
  --workers 2
```

禁用 memory，运行 baseline：

```bash
cd agent
python run_task.py \
  --task graph_connectivity \
  --backend api \
  --model gpt-4o
```

JSON memory：

```bash
cd agent
python run_task.py \
  --task math_convexity \
  --backend api \
  --model gpt-4o \
  --memory-format json
```

## 10. 与 Baseline 的变量控制

当前实验设计中，`VisualSketchpad-main/agent` 的主推理逻辑与 baseline 的差异应集中在：

- 是否启用外部 dynamic memory organization layer。
- memory 格式：`html`、`json`、`image`。
- gated symbolic tasks 是否注入 full memory。

不应再通过 draft wrapper 改写主 Reasoner prompt 来比较 HTML/JSON。这样可以更直接地测试：

> 将多模态模型的思考过程视为动态记忆时，HTML/JSON/image 哪种外部组织形式更适合作为可检查、可更新、可复用的视觉/结构化草稿。

