import json
import os
import argparse, shutil

from agent import SketchpadUserAgent
from multimodal_conversable_agent import MultimodalConversableAgent
from prompt import (
    ReACTPrompt,
    MathPrompt,
    GeoPrompt,
    HTMLVisualPrompt,
    JSONVisualPrompt,
    DraftFormatPromptWrapper,
    python_codes_for_images_reading,
    MULTIMODAL_ASSISTANT_MESSAGE,
    HTML_VISUAL_ASSISTANT_MESSAGE,
    JSON_VISUAL_ASSISTANT_MESSAGE,
)
from parse import Parser
from execution import CodeExecutor
from reflection import build_memory_prompt, reflect_and_update_memory
from utils import build_structured_trace, custom_encoder, save_json
from config import MAX_REPLY, build_llm_runtime_config, validate_llm_config


def checks_terminate_message(msg):
    if isinstance(msg, str):
        return msg.find("TERMINATE") > -1
    elif isinstance(msg, dict) and 'content' in msg:
        return msg['content'].find("TERMINATE") > -1
    else:
        print(type(msg), msg)
        raise NotImplementedError


def _wrap_prompt_generator(prompt_generator, task_type, draft_format):
    if draft_format is None:
        return prompt_generator
    return DraftFormatPromptWrapper(prompt_generator, task_type=task_type, draft_format=draft_format)


def _resolve_system_message(task_type, draft_format):
    if draft_format == "html":
        return HTML_VISUAL_ASSISTANT_MESSAGE
    if draft_format == "json":
        return JSON_VISUAL_ASSISTANT_MESSAGE
    if task_type == "t2i_html":
        return HTML_VISUAL_ASSISTANT_MESSAGE
    if task_type == "t2i_json":
        return JSON_VISUAL_ASSISTANT_MESSAGE
    return MULTIMODAL_ASSISTANT_MESSAGE


def run_agent(
    task_input,
    output_dir,
    task_type="vision",
    draft_format=None,
    task_name=None,
    backend=None,
    model=None,
    base_url=None,
    api_key=None,
    local_model=None,
    local_dtype=None,
    device_map=None,
):
    """Run the Visual Sketchpad agent on one task instance.

    Args:
        task_input (str): a path to the task input directory
        output_dir (str): a path to the directory where the output will be saved
        task_type (str): Task type. Should be vision, math, geo, t2i_html, or t2i_json. Defaults to "vision".
        draft_format (str, optional): Intermediate draft format for standard tasks. One of None, "html", or "json".
        task_name (str, optional): Only needed for math tasks. Defaults to None.
    """
    
    # task type should be one of "vision", "math", "geo", "t2i_html", "t2i_json"
    assert task_type in ["vision", "math", "geo", "t2i_html", "t2i_json"]
    assert draft_format in [None, "html", "json"]
    
    # create a directory for the task
    task_input = task_input.rstrip('/')
    task_directory = os.path.join(output_dir, os.path.basename(task_input))
    
    # copy the task input to the output directory
    os.makedirs(output_dir, exist_ok=True)
    shutil.copytree(task_input, task_directory, dirs_exist_ok=True)
    
    
    if task_type == "vision":
        
        # test if vision tools are loaded
        try:
            from tools import som_client, gd_client, da_client
        except ImportError as e:
            raise ImportError("Vision tools are not loaded. Please install vision_experts.")
        
        task_metadata = json.load(open(os.path.join(task_input, "request.json")))
        query = task_metadata['query']
        images = task_metadata['images']
    
        prompt_generator = _wrap_prompt_generator(ReACTPrompt(), task_type="vision", draft_format=draft_format)
        parser = Parser()
        executor = CodeExecutor(working_dir=task_directory, use_vision_tools=True)
        
        # read all images, save them in image_1, image_2, ... as PIL images
        image_reading_codes = python_codes_for_images_reading(images)
        image_loading_result = executor.execute(image_reading_codes)
        if image_loading_result[0] != 0:
            raise Exception(f"Error loading images: {image_loading_result[1]}")
        
    elif task_type == "math":
        query = json.load(open(os.path.join(task_input, "example.json")))
        images = []
        prompt_generator = _wrap_prompt_generator(MathPrompt(task_name), task_type="math", draft_format=draft_format)
        parser = Parser()
        executor = CodeExecutor(working_dir=task_directory)
        
    elif task_type == "geo":
        query = json.load(open(os.path.join(task_input, "ex.json")))
        images = []
        prompt_generator = _wrap_prompt_generator(GeoPrompt(), task_type="geo", draft_format=draft_format)
        parser = Parser()
        executor = CodeExecutor(working_dir=task_directory)

    elif task_type in {"t2i_html", "t2i_json"}:
        query_file = None
        for candidate in ("request.json", "prompt.json", "example.json", "ex.json"):
            candidate_path = os.path.join(task_input, candidate)
            if os.path.exists(candidate_path):
                query_file = candidate_path
                break
        if query_file is None:
            raise FileNotFoundError(
                f"{task_type} tasks require request.json, prompt.json, example.json, or ex.json."
            )

        task_payload = json.load(open(query_file))
        if isinstance(task_payload, dict):
            query = (
                task_payload.get("query")
                or task_payload.get("prompt")
                or task_payload.get("Prompt")
                or task_payload.get("text")
                or json.dumps(task_payload, ensure_ascii=False)
            )
        else:
            query = str(task_payload)
        images = []
        prompt_generator = HTMLVisualPrompt() if task_type == "t2i_html" else JSONVisualPrompt()
        parser = Parser()
        executor = CodeExecutor(working_dir=task_directory)
    
    validate_llm_config(backend=backend)
    llm_runtime = build_llm_runtime_config(
        backend=backend,
        model=model,
        base_url=base_url,
        api_key=api_key,
        local_model=local_model,
        local_dtype=local_dtype,
        device_map=device_map,
    )

    user = SketchpadUserAgent(
        name="multimodal_user_agent",
        human_input_mode='NEVER',
        max_consecutive_auto_reply=MAX_REPLY,
        is_termination_msg=checks_terminate_message,
        prompt_generator = prompt_generator,
        parser = parser,
        executor = executor
    )
    
    # running the planning experiment
    all_messages = {}
    
    planner = MultimodalConversableAgent(
        name="planner",
        human_input_mode='NEVER',
        max_consecutive_auto_reply=MAX_REPLY,
        is_termination_msg = lambda x: False,
        system_message=_resolve_system_message(task_type, draft_format) + build_memory_prompt(task_name),
        llm_config=False if llm_runtime.client is not None else None,
        llm_client=llm_runtime.client,
    )
    
    # run the agent
    try:
        user.initiate_chat(
            planner,
            n_image=len(images),
            task_id = "testing_case",
            message = query,
            log_prompt_only = False,
        )
        all_messages = planner.chat_messages[user]
        
    except Exception as e:
        print(e)
        all_messages = {'error': e.message if hasattr(e, 'message') else f"{e}"}
        
    
    # save the results
    os.makedirs(task_directory, exist_ok=True)
    structured_trace = build_structured_trace(all_messages)
    with open(os.path.join(task_directory, "output.json"), "w") as f:
        json.dump(all_messages, f, indent=4, default=custom_encoder)
    save_json(
        os.path.join(task_directory, "full_trajectory.json"),
        {
            "messages": structured_trace.get("messages", []),
        },
    )
    save_json(os.path.join(task_directory, "structured_trace.json"), structured_trace)
    save_json(
        os.path.join(task_directory, "prediction_summary.json"),
        {
            "task_type": task_type,
            "draft_format": draft_format,
            "task_name": task_name,
            "task_input": task_input,
            "final_answer": structured_trace.get("final_answer"),
            "num_messages": len(all_messages) if isinstance(all_messages, list) else 0,
            "num_turns": len(structured_trace.get("turns", [])),
            "status": structured_trace.get("status"),
            "error": structured_trace.get("error"),
        },
    )
        
    usage_summary = {'total': planner.client.total_usage_summary, 'actual': planner.client.actual_usage_summary}
    with open(os.path.join(task_directory, "usage_summary.json"), "w") as f:
        json.dump(usage_summary, f, indent=4)

    reflection_record = None
    try:
        if task_name:
            reflection_record = reflect_and_update_memory(
                task_name,
                task_input,
                structured_trace,
                task_directory,
                llm_client=llm_runtime.client,
            )
    except Exception as reflection_error:
        save_json(
            os.path.join(task_directory, "reflection_error.json"),
            {"error": str(reflection_error)},
        )
    if reflection_record is not None:
        save_json(os.path.join(task_directory, "reflection_summary.json"), reflection_record)
        
    # turn off server
    user.executor.cleanup()
        
    user.reset()
    planner.reset()
    
    
