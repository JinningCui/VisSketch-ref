
############################################################################################################
##### Prompt Generator for computer vision tasks using ReACT agent
############################################################################################################ 

MULTIMODAL_ASSISTANT_MESSAGE = """You are a helpful multimodal AI assistant.
Solve tasks using your vision, coding, and language skills.
The task can be free-form or multiple-choice questions.
You can answer the user's question about images. If you are not sure, you can coding 
You are coding in a Python jupyter notebook environment.
You can suggest python code (in a python coding block) for the user to execute. In a dialogue, all your codes are executed with the same jupyter kernel, so you can use the variables, working states.. in your earlier code blocks.
Solve the task step by step if you need to. 
The task may be a vision-language task and require several steps. You can write code to process images, text, or other data in a step. Give your code to the user to execute. The user may reply with the text and image outputs of the code execution. You can use the outputs to proceed to the next step, with reasoning, planning, or further coding.
If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
All images should be stored in PIL Image objects. The notebook has imported 'Image' from 'PIL' package and 'display' from 'IPython.display' package. If you want to read the image outputs of your code, use 'display' function to show the image in the notebook. The user will send the image outputs to you.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.

For each turn after the first one, you should first do a "REFLECTION" on whether your previous reasoning or action was flawed, what the concrete issue is, and how to correct it. Then do a "THOUGHT", based on the images and text you see.
Your reflection must be concise and operational. If the previous step looks sound, say so briefly and explain why you are continuing.
If you think you get the answer to the intial user request, you can reply with "ANSWER: concrete_final_value" and ends with "TERMINATE".
"""

HTML_VISUAL_ASSISTANT_MESSAGE = """You are a structured reasoning planner.
Use HTML as a dynamic visual draft: each non-final ACTION writes one valid self-contained HTML file.
You are coding in a Python jupyter notebook environment. Each ACTION must be one executable ```python``` block unless you are giving the final ANSWER.
Do not output JSON as the draft format.

For each non-final turn:
1. Start with REFLECTION. On the first turn, say this is the initial planning step.
2. Then write THOUGHT with the current reasoning.
3. Then write ACTION with Python code that writes a self-contained HTML file in the current working directory.

HTML draft contract:
- Valid UTF-8 HTML containing html/head/body tags.
- Include visible sections whose labels contain: Thinking, Objects, Relations, State, Revision, Retained, Referenced Images.
- Treat the draft as mutable memory: preserve only useful prior content, edit wrong content, delete stale content.
- You may reference 0, 1, or many images. If Python draws auxiliary lines or overlays, save the image and reference it with a caption explaining its reasoning role.
- Use HTML's stronger expressive power when useful: tables, CSS layout, inline SVG, diagrams, boards, arrows, overlays, charts, and image references.
- Print exactly: HTML_DRAFT_PATH: <path> and HTML_DRAFT_SUMMARY: <short summary>.
Never include `ANSWER:` or `TERMINATE` inside the Python code block; the final answer must be normal text outside code.
If a Python ACTION computes the final value, print the value for inspection, then in the same assistant message after the code block write `ANSWER: <that concrete value> TERMINATE`.
Do not mention the literal token `ANSWER:` in REFLECTION/THOUGHT unless you are actually giving the final answer.
Never use placeholder final answers such as `{answer}`, `{final_answer}`, `<answer>`, or `[answer]`. The final answer must be a concrete label/value required by the task, such as `yes`, `no`, `(A)`, `128`, or `white`.

Use local revision rather than regenerating the whole plan when the previous draft only needs a focused fix.
If you have enough confidence, reply with:
ANSWER: concrete_final_value TERMINATE
"""

JSON_VISUAL_ASSISTANT_MESSAGE = """You are a structured reasoning planner.
Use JSON as a dynamic declarative visual draft: each non-final ACTION writes one valid JSON file.
You are coding in a Python jupyter notebook environment. Each ACTION must be one executable ```python``` block unless you are giving the final ANSWER.
Do not output HTML as the draft format.

For each non-final turn:
1. Start with REFLECTION. On the first turn, say this is the initial planning step.
2. Then write THOUGHT with the current reasoning.
3. Then write ACTION with Python code that writes a self-contained JSON file in the current working directory.

JSON draft contract:
- Valid UTF-8 JSON whose root is an object.
- Include keys: title, original_prompt, thinking_text, objects_entities, relations_constraints, state_effects_view, retained_context, referenced_images, open_issues_revision_targets.
- Treat the draft as mutable memory: preserve only useful prior fields, edit wrong fields, delete stale fields.
- You may reference 0, 1, or many images. If Python draws auxiliary lines or overlays, save the image and store its path plus reasoning role in referenced_images.
- JSON should encode the same reasoning content declaratively; it does not need to visually render layout unless layout is relevant evidence.
- Print exactly: JSON_DRAFT_PATH: <path> and JSON_DRAFT_SUMMARY: <short summary>.
Do not put raw text like `JSON_DRAFT_PATH: ...` or `JSON_DRAFT_SUMMARY: ...` inside the code block unless it is inside a Python print(...) call.
Never include `ANSWER:` or `TERMINATE` inside the Python code block; the final answer must be normal text outside code.
If a Python ACTION computes the final value, print the value for inspection, then in the same assistant message after the code block write `ANSWER: <that concrete value> TERMINATE`.
Do not mention the literal token `ANSWER:` in REFLECTION/THOUGHT unless you are actually giving the final answer.
Never use placeholder final answers such as `{answer}`, `{final_answer}`, `<answer>`, or `[answer]`. The final answer must be a concrete label/value required by the task, such as `yes`, `no`, `(A)`, `128`, or `white`.

Use local revision rather than regenerating the whole plan when the previous draft only needs a focused fix.
If you have enough confidence, reply with:
ANSWER: concrete_final_value TERMINATE
"""




class ReACTPrompt:
    
    def __init__(self) -> None:
        return
    
    def initial_prompt(self, query: str, n_images: int) -> str:
        
        initial_prompt = """Here are some tools that can help you. All are python codes. They are in tools.py and will be imported for you.
The images has their own coordinate system. The upper left corner of the image is the origin (0, 0). All coordinates are normalized, i.e., the range is [0, 1].
All bounding boxes are in the format of [x, y, w, h], which is a python list. x is the horizontal coordinate of the upper-left corner of the box, y is the vertical coordinate of that corner, w is the box width, and h is the box height.
Notice that you, as an AI assistant, is not good at locating things and describe them with coordinate. You can use tools to generate bounding boxes.
You are also not good at answering questions about small visual details in the image. You can use tools to zoom in on the image to see the details. Below are the tools in tools.py:
```python
class AnnotatedImage:
    # A class to represent an annotated image. It contains the annotated image and the original image.
    
    def __init__(self, annotated_image: Image.Image, original_image: Image.Image=None):
        self.annotated_image = annotated_image
        self.original_image = original_image
        
def detection(image, objects):
    \"\"\"Object detection using Grounding DINO model. It returns the annotated image and the bounding boxes of the detected objects.
    The text can be simple noun, or simple phrase (e.g., 'bus', 'red car'). Cannot be too hard or the model will break.
    The detector is not perfect, it may wrongly detect objects or miss some objects.
    Also, notice that the bounding box label might be out of the image boundary.
    You should use the output as a reference, not as a ground truth.
    When answering questions about the image, you should double-check the detected objects.

    Args:
        image (PIL.Image.Image): the input image
        objects (List[str]): a list of objects to detect. Each object should be a simple noun or a simple phrase. Should not be hard or abstract concepts like "text" or "number".

    Returns:
        output_image (AnnotatedImage): the original image, annotated with bounding boxes. Each box is labeled with the detected object, and an index.
        processed boxes (List): listthe bounding boxes of the detected objects
    
    Example:
        image = Image.open("sample_img.jpg")
        output_image, boxes = detection(image, ["bus"])
        display(output_image.annotated_image)
        print(boxes) # [[0.24, 0.21, 0.3, 0.4], [0.6, 0.3, 0.2, 0.3]]
        # you need to double-check the detected objects. Some objects may be missed or wrongly detected.
    \"\"\"
    
def sliding_window_detection(image, objects):
    \"\"\"Use this when you are searching for objects in the image, but the objects are not detected by the object detection model.
    In that case, the most common reason is that the object is too small such that both the vision-language model and the object detection model fail to detect it.
    This function tries to detect the object by sliding window search.
    With the help of the detection model, it tries to detect the object in the zoomed-in patches.
    The function returns a list of annotated images that may contain at leas one of the objects, annotated with bounding boxes.
    It also returns a list of a list of bounding boxes of the detected objects.

    Args:
        image (PIL.Image.Image): the input image
        objects (List[str]): a list of objects to detect. Each object should be a simple noun or a simple phrase. Should not be hard or abstract concepts like "text" or "number".
        
    Returns:
        possible_patches (List[AnnotatedImage]): a list of annotated zoomed-in images that may contain the object, annotated with bounding boxes.
        possible_boxes (List[List[List[Float]]]): For each image in possible_patches, a list of bounding boxes of the detected objects. 
            The coordinates are w.r.t. each zoomed-in image. The order of the boxes is the same as the order of the images in possible_patches.
            
    Example:
        image = Image.open("sample_img.jpg")
        possible_patches, possible_boxes = search_object_and_zoom(image, ["bird", "sign"])
        for i, patch in enumerate(possible_patches):
            print(f"Patch {i}:")
            display(patch.annotated_image)
        
        # print the bounding boxes of the detected objects in the first patch
        print(possible_boxes[0]) # [[0.24, 0.21, 0.3, 0.4], [0.6, 0.3, 0.2, 0.3]]
    \"\"\"
    
def segment_and_mark(image, anno_mode:list = ['Mask', 'Mark']):
    \"\"\"Use a segmentation model to segment the image, and add colorful masks on the segmented objects. Each segment is also labeled with a number.
    The annotated image is returned along with the bounding boxes of the segmented objects.
    This tool may help you to better reason about the relationship between objects, which can be useful for spatial reasoning etc.
    DO NOT use this tool to search or detect an object. It is likely the object is small and segmentaiton does not help.
    Segmentation and marking can also be helpful for 3D and video reasoning. For example, helping you to see more clearly and analyzes the relationship between different frames of a video.
    
    Args:
        image (PIL.Image.Image): the input image
        anno_mode (list, optional): What annotation is added on the input image. Mask is the colorful masks. And mark is the number labels. Defaults to ['Mask', 'Mark'].
        
    Returns:
        output_image (AnnotatedImage): the original image annotated with colorful masks and number labels. Each mask is labeled with a number. The number label starts at 1.
        bboxes (List): listthe bounding boxes of the masks.The order of the boxes is the same as the order of the number labels.
        
    Example:
        User request: I want to find a seat close to windows, where should I sit?
        Code:
        ```python
        image = Image.open("sample_img.jpg")
        output_image, bboxes = segment_and_mark(image)
        display(output_image.annotated_image)
        ```
        Model reply: You can sit on the chair numbered as 5, which is close to the window.
        User: Give me the bounding box of that chair.
        Code:
        ```python
        print(bboxes[4]) # [0.24, 0.21, 0.3, 0.4]
        ```
        Model reply: The bounding box of the chair numbered as 5 is [0.24, 0.21, 0.3, 0.4].
    \"\"\"
    
def depth(image):
    \"\"\"Depth estimation using DepthAnything model. It returns the depth map of the input image. 
    A colormap is used to represent the depth. It uses Inferno colormap. The closer the object, the warmer the color.
    This tool may help you to better reason about the spatial relationship, like which object is closer to the camera.

    Args:
        image (PIL.Image.Image): the input image

    Returns:
        output_image (PIL.Image.Image): the depth map of the input image
        
    Example:
        image = Image.open("sample_img.jpg")
        output_image = depth(image)
        display(output_image)
    \"\"\"
    
def zoom_in_image_by_bbox(image, box, padding=0.05):
    \"\"\"A simple wrapper function to crop the image based on the bounding box.
    When you want to answer question about visual details in a bounding box annotated by the detection tool, you would like to zoom in on the object using this function.

    Args:
        image (PIL.Image.Image): the input image
        box (List[float]): the bounding box in the format of [x, y, w, h]
        padding (float, optional): The padding for the image crop, outside of the bounding box. Defaults to 0.1. The zoom factor cannot be too small. Minimum is 0.05

    Returns:
        cropped_img (PIL.Image.Image): the cropped image
        
    Example:
        image = Image.open("sample_img.jpg")
        annotated_img, boxes = detection(image, "bus")
        cropped_img = zoom_in_image_by_bbox(image, boxes[0], padding=0.05)
        display(cropped_img)
    \"\"\"
    
def overlay_images(background_img, overlay_img, alpha=0.3, bounding_box=[0, 0, 1, 1]):
    \"\"\"
    Overlay an image onto another image with transparency.
    This is particularly useful visualizing heatmap while preserving some info from the original image.
    For example, you can overlay a segmented image on a heatmap to better understand the spatial relationship between objects.
    It will also help seeing the labels, circles on the original image that may not be visible on the heatmap.

    Args:
    background_img_pil (PIL.Image.Image): The background image in PIL format.
    overlay_img_pil (PIL.Image.Image): The image to overlay in PIL format.
    alpha (float): Transparency of the overlay image.
    bounding_box (List[float]): The bounding box of the overlay image. The format is [x, y, w, h]. The coordinates are normalized to the background image. Defaults to [0, 0, 1, 1].

    Returns:
    PIL.Image.Image: The resulting image after overlay, in PIL format.
    s
    Example:
        image = Image.open('original.jpg')
        depth_map = depth(image)
        overlayed_image = overlay_images(depth_map, image, alpha=0.3)
        display(overlayed_image)
    \"\"\"
```
# GOAL #: Based on the above tools, I want you to reason about how to solve the # USER REQUEST # and generate the actions step by step (each action is a python jupyter notebook code block) to solve the request.
You may need to use the tools above to process the images and make decisions based on the visual outputs of the previous code blocks.
Your visual ability is not perfect, so you should use these tools to assist you in reasoning about the images.

The jupyter notebook has already executed the following code to import the necessary packages:
```python
from PIL import Image
from IPython.display import display
from tools import *
```

# REQUIREMENTS #:
1. The generated actions can resolve the given user request # USER REQUEST # perfectly. The user request is reasonable and can be solved. Try your best to solve the request.
2. The arguments of a tool must be the same number, modality, and format specified in # TOOL LIST #;
3. If you think you got the answer, use ANSWER: concrete_final_value to provide the answer, and ends with TERMINATE.
4. All images in the initial user request are stored in PIL Image objects named image_1, image_2, ..., image_n. You can use these images in your code blocks. Use display() function to show the image in the notebook for you too see.
5. Use as few tools as possible. Only use the tools for the use cases written in the tool description. You can use multiple tools in a single action.
6. You must return an answer with the choice letter if the user request is a multiple-choice question.
7. When detection tool failed to detect an object, you can use the sliding_window_detection tool to search for the object.
8. Bounding boxes may be wrong and misled you. When answering questions about small objects in bounding boxes, you should zoom in on the object to see the details.
9. Segmentation and marking tool can help you better reason about the relationship between objects. Example use cases are spatial reasoning (e.g., left/right/above/below/on) and counting.


Below are some examples of how to use the tools to solve the user requests. You can refer to them for help. You can also refer to the tool descriptions for more information.
# EXAMPLE: Answering a simple question about the image
# USER REQUEST #: <A image here> which city is this?
# USER IMAGE stored in image_1, as PIL image.
# RESULT #:
THOUGHT 0: The tools are not helpful for this task. I can directly answer the question.
ACTION 0: No action needed.
ANSWER: This looks like New York City. I can see the Empire State Building in the image. TERMINATE


# EXAMPLE: Detecting objects in the image
# USER REQUEST #: <A image here> is there a dog in the image? Options: (a) Yes (b) No
# USER IMAGE stored in image_1, as PIL image.
# RESULT #:
THOUGHT 0: Firstly, I need to detect dogs in the image.
ACTION 0: 
```python
image_1_dog_detection, image_1_dog_boxes = detection(image_1, ["dog"])
display(image_1_dog_detection.annotated_image)
```
OBSERVATION: Execution success. The output is as follows:
<the image outputs of the previous code is here.>
If you can get the answer, please reply with ANSWER: concrete_final_value and ends with TERMINATE. Otherwise, please generate the next THOUGHT and ACTION.
THOUGHT 1: A dog is detected by the tool. After double-checking, I can confirm that the tool is correct and there is a dog in the image.
ACTION 1: No action needed.
ANSWER: Yes, there is a dog in the image. So the answer is (a). TERMINATE


# EXAMPLE: Zooming in to answer the question about a small object
# USER REQUEST #: <A image here> What is the color of the busket? Options: (a) white (b) black (c) red
# USER IMAGE stored in image_1, as PIL image.
# RESULT #:
THOUGHT 0: To identify the color of the bucket in the image, I need to locate the bucket first. Since the bucket might be small and not easily detectable, I will use the detection tool to find it.
ACTION 0:
```python
image_1_bucket_detection, image_1_bucket_boxes = detection(image_1, ["bucket"])
display(image_1_bucket_detection.annotated_image)
```
OBSERVATION: Execution success. The output is as follows:
<the image outputs of the previous code is here.>
If you can get the answer, please reply with ANSWER: concrete_final_value and ends with TERMINATE. Otherwise, please generate the next THOUGHT and ACTION.
THOUGHT 1: The bucket has been detected and is labeled as "bucket 1" in the image. I will zoom in on the bucket to identify its color.
ACTION 1:
```python
bucket_box = image_1_bucket_boxes[0]
zoomed_bucket = zoom_in_image_by_bbox(image_1, bucket_box)
display(zoomed_bucket)
```
OBSERVATION: Execution success. The output is as follows:
<the image outputs of the previous code is here.>
THOUGHT 2: From the zoomed-in image, I can see that the color of the bucket is white.
ACTION 2: No action needed.
ANSWER: The color of the bucket is white. So the answer is (a). TERMINATE


# EXAMPLE: Searching for small objects in the image
# USER REQUEST #: <A image here> What animal is drawn on that red signicade?
# USER IMAGE stored in image_1, as PIL image.
# RESULT #:
THOUGHT 0: I do not see a sign that is both red and has an animal drawn on it. I need to use the sliding_window_detection tool to find that sign and zoom in on it.
ACTION 0:
```python
possible_patches = sliding_window_detection(image_1, ["animal", "sign"])
for i, patch in enumerate(possible_patches):
    print(f"Patch {i+1}:")
    display(patch.annotated_image)
```
OBSERVATION: Execution success. The output is as follows:
Patch 1: 
<image output 1>
Patch 2:
<image output 2>
<More patches below...>
THOUGHT 1: Among the above patches, Patch 4 contains the animal drawn on the red signicade. An animal was detected. I will zoom in to double check if the detection is correct.
ACTION 1: 
```python
relevant_patch = possible_patches[3]
display(relevant_patch.annotated_image)
```
OBSERVATION: Execution success. The output is as follows:
<the image outputs of the previous code is here.>
THOUGHT 2: I can confirm that there is an animal drawn on the red signicade. I will zoom in to see what animal it is.
ACTION 2:
```python
animal_box = relevant_patch.boxes[2]
zoomed_patch = zoom_in_image_by_bbox(relevant_patch.original_image, animal_box)
display(zoomed_patch)
```
OBSERVATION: Execution success. The output is as follows:
<the image outputs of the previous code is here.>
THOUGHT 3: It can be seen that the animal drawn on the red signicade is a tiger.
ACTION 3: No action needed.
ANSWER: The animal drawn on the red signicade is a tiger. TERMINATE


# EXAMPLE: Reasoning about the depth of the objects
# USER REQUEST #: <A image here> Which point is closer to the camera? Options: (A) Point A (B) Point B
# USER IMAGE stored in image_1, as PIL image.
# RESULT #:
THOUGHT 0: I can use the depth estimation tool. This tool will provide a depth map where different colors represent different distances from the camera. 
I can also use the overlay_images tool to overlay the original image on the heat map to get a better sense of where point A and point B are on the depth map.
ACTION 0:
```python
image_1_depth = depth(image_1)
overlayed_image_1 = overlay_images(image_1_depth, image_1, alpha=0.3)
display(overlayed_image_1)

# depth can be used together with segmentation and marking tool
image_1_segmented, image_1_boxes = segment_and_mark(image_1)
overlayed_image_2 = overlay_images(image_1_depth, image_1_segmented.annotated_image, alpha=0.5)
display(overlayed_image_2)
```
OBSERVATION: Execution success. The output is as follows:
<the image outputs of the previous code is here.>
THOUGHT 1: The tool is helpful. From the overlayed image, I can see that the color in the circle under point B is warmer than that of point A, which means point B is closer to the camera.
ACTION 1: No action needed.
ANSWER: Based on the depth map, the circle under point B, which shows warmer colors, is closer to the camera than point A. So the answer is (B). TERMINATE


# EXAMPLE: Reasoning about the spatial relationship between objects
# USER REQUEST #: <A image here> What is on the left of the right laptop? Options: 
(A) Lamp 
(B) Chair 
(C) Plant 
(D) None of the above
# USER IMAGE stored in image_1, as PIL image.
# RESULT #:
THOUGHT 0: To help me better reason about the relationship between objects, I can use the segmentation and marking tool.
ACTION 0:
```python
image_1_segmented, image_1_boxes = segment_and_mark(image_1)
display(image_1_segmented.annotated_image)
```
OBSERVATION: Execution success. The output is as follows:
<the image outputs of the previous code is here.>
THOUGHT 1: The tool is helpful. I can see the right laptop is labeled as 9. There is a lamp on its left, which is labeled as 12.
ACTION 1: No action needed.
ANSWER: On the left side of the right laptop, there is a lamp. So the answer is (A). TERMINATE


# EXAMPLE: Doing Jigsaw Puzzle
# USER REQUEST #: <An image here> <An image here> <An image here> The lower right part of the first image is black. The second and the third images might fit into the black part. Can you check which one fits?
# USER IMAGE stored in image_1, image_2, image_3 as PIL image.
# RESULT #:
THOUGHT 0: To check if the image fits, I can use the overlay_images tool to overlay the second and third image on the first image. I will overlay them on the black part to see if they fit.
ACTION 0:
```python
image_1_overlayed_with_image_2 = overlay_images(image_1, image_2, alpha=1.0, bounding_box=[0.5, 0.5, 0.5, 0.5])
image_1_overlayed_with_image_3 = overlay_images(image_1, image_3, alpha=1.0, bounding_box=[0.5, 0.5, 0.5, 0.5])
display(image_1_overlayed_with_image_2)
display(image_1_overlayed_with_image_3)
```
OBSERVATION: Execution success. The output is as follows:
<the image outputs of the previous code is here.>
THOUGHT 1: Comparing two images, I can see that the third image fits into the black part. The second image does not fit. You can see a building in the second image that is not in the first image.
ACTION 1: No action needed.
ANSWER: The third image fits into the black part. TERMINATE


"""
        prompt = initial_prompt
        prompt += f"# USER REQUEST #: {query}\n"
        if n_images > 0:
            prompt += f"# USER IMAGE stored in {', '.join([f'image_{i}' for i in range(1, n_images+1)])} as PIL image.\n"
        else:
            prompt += "# USER IMAGE: No image provided.\n"
        prompt += "Now please generate only THOUGHT 0 and ACTION 0 in RESULT. If no action needed, also reply with ANSWER: concrete_final_value and ends with TERMINATE in the RESULT:\n# RESULT #:\n"
        return prompt
    
    def get_parsing_feedback(self, error_message: str, error_code: str) -> str:
        return (
            f"OBSERVATION: Parsing error. Error code: {error_code}, Error message:\n{error_message}\n"
            "First provide REFLECTION on what was wrong with the previous response format or reasoning, "
            "then fix it and generate the next THOUGHT and ACTION."
        )
    
    def get_exec_feedback(self, exit_code: int, output: str) -> str:
        
        # if execution fails
        if exit_code != 0:
           return (
               f"OBSERVATION: Execution error. Exit code: {exit_code}, Output:\n{output}\n"
               "First provide REFLECTION on why the previous step failed and what concrete correction is needed, "
               "then generate the fixed THOUGHT and ACTION."
           )
        else:
            prompt = f"OBSERVATION: Execution success. The output is as follows:\n{output}\n"
            prompt += (
                "First provide REFLECTION on whether the previous reasoning/action was valid or needs correction, "
                "then generate the next THOUGHT and ACTION. If you can get the answer, please also reply with "
                "ANSWER: concrete_final_value and ends with TERMINATE."
            )
            return prompt
    
# a special prompt generator to generate codes that read all image files
# needed for executor initialization
def python_codes_for_images_reading(image_paths):
    
    code = ""
    
    for idx, path in enumerate(image_paths):
        code += f"""image_{idx+1} = Image.open("{path}").convert("RGB")\n"""
        
    return code


class HTMLVisualPrompt:
    """Prompt generator for StruVis-style HTML visual drafts."""

    draft_format = "html"

    def initial_prompt(self, query: str, n_images: int = 0) -> str:
        return f"""# USER REQUEST #
{query}

# GOAL #
Reason with HTML visual drafts, then finish with the required final answer or generative prompt.

# ACTION FORMAT #
For every non-final step, return exactly one Python code block. It must write one UTF-8 `.html` file and print exactly:
HTML_DRAFT_PATH: <path>
HTML_DRAFT_SUMMARY: <short summary>

# DYNAMIC HTML DRAFT CONTRACT #
- The file must contain html/head/body tags and visible sections labelled Thinking, Objects, Relations, State, Revision, Retained, Referenced Images.
- Treat each draft as editable memory. Keep only prior content that is still useful, correct mistakes, delete stale details, and add new evidence.
- You may reference 0, 1, or many images. If you draw auxiliary lines, overlays, charts, boards, or diagrams with Python, save the image and reference it in the HTML with a caption explaining why it matters.
- Use HTML's expressive power when helpful: CSS layout, tables, inline SVG arrows, image overlays, board renderings, diagrams, charts, and spatial grouping.
- Do not use JSON as the draft representation.

# FINAL FORMAT #
When the draft is complete, respond with `ANSWER: concrete_final_value TERMINATE`.
"""

    def get_parsing_feedback(self, error_message: str, error_code: str) -> str:
        return (
            "OBSERVATION: The previous response did not contain one executable Python code block.\n"
            f"Parser message: {error_message}\n"
            "Continue with REFLECTION, THOUGHT, and exactly one ACTION code block that writes the HTML draft. "
            "If the draft is complete, use ANSWER: concrete_final_value TERMINATE."
        )

    def get_exec_feedback(self, exit_code: int, output: str) -> str:
        if exit_code != 0:
            return (
                f"OBSERVATION: Execution error. Exit code: {exit_code}, Output:\n{output}\n"
                "First provide REFLECTION on why the previous HTML-draft code failed, then generate the full corrected ACTION code block."
            )
        return (
            f"OBSERVATION: Execution success. The output is as follows:\n{output}\n"
            "Inspect the HTML draft path and summary. If the draft is incomplete or inconsistent, revise it locally in the next ACTION. "
            "If it is sufficient, provide ANSWER: concrete_final_value TERMINATE as plain text with no code block."
        )

    def get_validation_feedback(self, validation_message: str, output: str) -> str:
        return (
            "OBSERVATION: The previous code executed, but the HTML draft failed validation.\n"
            f"Validation message: {validation_message}\n"
            f"Execution output:\n{output}\n"
            "Revise with REFLECTION, THOUGHT, and exactly one ACTION code block that writes a valid dynamic HTML draft."
        )


class JSONVisualPrompt:
    """Prompt generator for StruVis-style JSON visual drafts."""

    draft_format = "json"

    def initial_prompt(self, query: str, n_images: int = 0) -> str:
        return f"""# USER REQUEST #
{query}

# GOAL #
Reason with JSON declarative visual drafts, then finish with the required final answer or generative prompt.

# ACTION FORMAT #
For every non-final step, return exactly one Python code block. It must write one UTF-8 `.json` file and print exactly:
JSON_DRAFT_PATH: <path>
JSON_DRAFT_SUMMARY: <short summary>

# DYNAMIC JSON DRAFT CONTRACT #
- The JSON root must be an object with keys: title, original_prompt, thinking_text, objects_entities, relations_constraints, state_effects_view, retained_context, referenced_images, open_issues_revision_targets.
- Treat each draft as editable memory. Keep only prior fields that are still useful, correct mistakes, delete stale details, and add new evidence.
- You may reference 0, 1, or many images. If you draw auxiliary lines, overlays, charts, boards, or diagrams with Python, save the image and store its path plus reasoning role in referenced_images.
- JSON should encode visual evidence declaratively. Do not use HTML as the draft representation.

# FINAL FORMAT #
When the draft is complete, respond with `ANSWER: concrete_final_value TERMINATE`.
"""

    def get_parsing_feedback(self, error_message: str, error_code: str) -> str:
        return (
            "OBSERVATION: The previous response did not contain one executable Python code block.\n"
            f"Parser message: {error_message}\n"
            "Continue with REFLECTION, THOUGHT, and exactly one ACTION code block that writes the JSON draft. "
            "If the draft is complete, use ANSWER: concrete_final_value TERMINATE."
        )

    def get_exec_feedback(self, exit_code: int, output: str) -> str:
        if exit_code != 0:
            return (
                f"OBSERVATION: Execution error. Exit code: {exit_code}, Output:\n{output}\n"
                "First provide REFLECTION on why the previous JSON-draft code failed, then generate the full corrected ACTION code block."
            )
        return (
            f"OBSERVATION: Execution success. The output is as follows:\n{output}\n"
            "Inspect the JSON draft path and summary. If the draft is incomplete or inconsistent, revise it locally in the next ACTION. "
            "If it is sufficient, provide ANSWER: concrete_final_value TERMINATE as plain text with no code block."
        )

    def get_validation_feedback(self, validation_message: str, output: str) -> str:
        return (
            "OBSERVATION: The previous code executed, but the JSON draft failed validation.\n"
            f"Validation message: {validation_message}\n"
            f"Execution output:\n{output}\n"
            "Revise with REFLECTION, THOUGHT, and exactly one ACTION code block that writes a valid dynamic JSON draft."
        )


INITIAL_RESULT_SUFFIX = (
    "Now please generate only THOUGHT 0 and ACTION 0 in RESULT. If no action needed, also reply with "
    "ANSWER: concrete_final_value and ends with TERMINATE in the RESULT:\n# RESULT #:\n"
)


def _draft_task_specific_guidance(task_type: str) -> str:
    if task_type == "geo":
        return (
            "- For geometry, keep symbolic givens, target, auxiliary constructions, theorem candidates, and equation chains explicit.\n"
            "- For geometry, the first non-final draft should normally generate a matplotlib/PIL evidence image, preferably the original diagram redrawn with useful auxiliary lines, labels, highlights, or angle/length annotations.\n"
            "- HTML geometry drafts can embed that image with an <img> tag and a caption explaining its reasoning role; JSON geometry drafts can record that image in referenced_images with path and role.\n"
            "- This follows the VisualSketchpad convention: use visual drawing when it helps, but later turns may answer without generating another image if the existing evidence is sufficient.\n"
            "- HTML drafts should visualize not-to-scale diagrams, labels, and theorem/equation panels; JSON drafts should encode those items as structured fields.\n"
        )
    if task_type == "math":
        return (
            "- For graph and math tasks, keep the exact problem data, intermediate state, and decisive evidence explicit.\n"
            "- For graph tasks, include nodes, edges, capacities/weights, and current path/cut/flow or connectivity state.\n"
            "- For board-game tasks such as chess, include board state, side to move, legal-move summary, terminal signals, and concise evidence.\n"
        )
    return (
        "- For vision tasks, preserve object evidence, relevant spatial relations, and any state that explains the final answer.\n"
        "- HTML drafts may visualize regions, boxes, and relations; JSON drafts should keep the same content in symbolic form.\n"
    )


def build_draft_format_guidance(draft_format: str, task_type: str) -> str:
    if draft_format == "html":
        header = (
            "# INTERMEDIATE DRAFT FORMAT #:\n"
            "Use HTML as the dynamic visual draft format while keeping the original task objective unchanged.\n"
            "Every non-final ACTION must write one valid `.html` draft and print exactly `HTML_DRAFT_PATH: <path>` and `HTML_DRAFT_SUMMARY: <summary>`.\n"
            "The HTML must contain html/head/body tags and visible sections labelled Thinking, Objects, Relations, State, Revision, Retained, Referenced Images.\n"
            "Across turns, selectively keep, edit, or delete prior draft content. Reference 0, 1, or many images; generated overlays/diagrams must be saved and embedded or linked with a reasoning caption.\n"
            "For geometry tasks, prefer generating an auxiliary-line or annotated diagram image in the first draft when it helps, matching the VisualSketchpad drawing convention.\n"
            "If ready to finish, output `ANSWER: concrete_final_value TERMINATE` as plain text with no code block and no placeholder answer.\n"
            "Minimal ACTION pattern:\n"
            "```python\n"
            "from pathlib import Path\n"
            "import html\n"
            "doc = '''<!doctype html><html><head><meta charset=\"utf-8\"><title>Visual Draft</title></head><body><h1>Visual Draft</h1><section><h2>Thinking</h2></section><section><h2>Objects</h2></section><section><h2>Relations</h2></section><section><h2>State</h2></section><section><h2>Revision</h2></section><section><h2>Retained</h2></section><section><h2>Referenced Images</h2></section></body></html>'''\n"
            "path = Path('visual_draft_step_0.html')\n"
            "path.write_text(doc, encoding='utf-8')\n"
            "print(f\"HTML_DRAFT_PATH: {path}\")\n"
            "print('HTML_DRAFT_SUMMARY: concise summary')\n"
            "```\n"
        )
    elif draft_format == "json":
        header = (
            "# INTERMEDIATE DRAFT FORMAT #:\n"
            "Use JSON as the dynamic declarative visual draft format while keeping the original task objective unchanged.\n"
            "Every non-final ACTION must write one valid `.json` draft and print exactly `JSON_DRAFT_PATH: <path>` and `JSON_DRAFT_SUMMARY: <summary>`.\n"
            "The JSON root must include: title, original_prompt, thinking_text, objects_entities, relations_constraints, state_effects_view, retained_context, referenced_images, open_issues_revision_targets.\n"
            "Across turns, selectively keep, edit, or delete prior draft fields. Reference 0, 1, or many images; generated overlays/diagrams must be saved and recorded with their reasoning role.\n"
            "For geometry tasks, prefer generating an auxiliary-line or annotated diagram image in the first draft when it helps, matching the VisualSketchpad drawing convention.\n"
            "If ready to finish, output `ANSWER: concrete_final_value TERMINATE` as plain text with no code block and no placeholder answer.\n"
            "Minimal ACTION pattern:\n"
            "```python\n"
            "from pathlib import Path\n"
            "import json\n"
            "draft = {\n"
            "    'title': 'Visual Draft',\n"
            "    'original_prompt': '...',\n"
            "    'thinking_text': '...',\n"
            "    'objects_entities': [],\n"
            "    'relations_constraints': [],\n"
            "    'state_effects_view': {},\n"
            "    'retained_context': {'kept': [], 'changed': [], 'removed': []},\n"
            "    'referenced_images': [],\n"
            "    'open_issues_revision_targets': []\n"
            "}\n"
            "path = Path('visual_draft_step_0.json')\n"
            "path.write_text(json.dumps(draft, ensure_ascii=False, indent=2), encoding='utf-8')\n"
            "print(f\"JSON_DRAFT_PATH: {path}\")\n"
            "print('JSON_DRAFT_SUMMARY: concise summary')\n"
            "```\n"
        )
    else:
        return ""

    footer = (
        "Keep the final answer criteria unchanged; only the intermediate draft representation changes.\n"
        + _draft_task_specific_guidance(task_type)
    )
    return header + footer


def _rewrite_legacy_prompt_for_draft_format(prompt: str, draft_format: str, task_type: str) -> str:
    if draft_format not in {"html", "json"}:
        return prompt

    draft_label = "HTML" if draft_format == "html" else "JSON"
    suffix = (
        f"\n# {draft_label} DRAFT OVERRIDE #:\n"
        f"Any previous examples that show raw `display(image)` or `plt.show()` are helper inspections only. "
        f"The accepted intermediate artifact for this run is a valid dynamic {draft_label} draft file that passes the format contract below.\n"
    )
    return prompt + suffix


class DraftFormatPromptWrapper:
    def __init__(self, base_prompt, task_type: str, draft_format: str) -> None:
        self.base_prompt = base_prompt
        self.task_type = task_type
        self.draft_format = draft_format

    def initial_prompt(self, query, n_images):
        prompt = self.base_prompt.initial_prompt(query, n_images)
        prompt = _rewrite_legacy_prompt_for_draft_format(prompt, self.draft_format, self.task_type)
        guidance = build_draft_format_guidance(self.draft_format, self.task_type)
        if not guidance:
            return prompt
        if prompt.endswith(INITIAL_RESULT_SUFFIX):
            return prompt[: -len(INITIAL_RESULT_SUFFIX)] + guidance + "\n" + INITIAL_RESULT_SUFFIX
        return prompt + "\n" + guidance

    def get_parsing_feedback(self, error_message: str, error_code: str) -> str:
        base_feedback = self.base_prompt.get_parsing_feedback(error_message, error_code)
        draft_label = "HTML" if self.draft_format == "html" else "JSON"
        return (
            base_feedback
            + f"\nKeep using {draft_label} as the intermediate draft format for this task; do not switch representations."
        )

    def get_exec_feedback(self, exit_code: int, output: str) -> str:
        base_feedback = self.base_prompt.get_exec_feedback(exit_code, output)
        draft_label = "HTML" if self.draft_format == "html" else "JSON"
        if exit_code != 0:
            return (
                base_feedback
                + f"\nRetry with corrected {draft_label}-draft code while preserving the same task-solving goal."
            )
        return (
            base_feedback
            + f"\nIf another intermediate step is needed, keep the next non-trivial ACTION in {draft_label} draft form."
        )

    def get_validation_feedback(self, validation_message: str, output: str) -> str:
        draft_label = "HTML" if self.draft_format == "html" else "JSON"
        return (
            f"OBSERVATION: The previous code executed, but the {draft_label} draft failed validation.\n"
            f"Validation message: {validation_message}\n"
            f"Execution output:\n{output}\n"
            f"Continue with REFLECTION, THOUGHT, and exactly one ACTION code block that writes a valid dynamic {draft_label} draft. "
            "Keep useful prior content, remove stale content, and fix the reported format problem."
        )




############################################################################################################
##### Prompt Generator for IsoBench tasks using ReACT agent
############################################################################################################ 

from math_data import TASK2PROMPT


matrix = [[0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 1, 0, 1], [0, 1, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0]]
query_node_1 = 4
query_node_2 = 0
fen = "1r1q1rk1/1b2b1Qp/4pp1B/pp1nP3/2pPN3/P1P5/1PB3PP/R4RK1 b - - 0 18"
class MathPrompt:
    def __init__(self, subtask, image=False) -> None:
        self.task2prompt = TASK2PROMPT
        self.subtask = subtask
        self.image = image
        return

# # Check if there is a path between the two nodes
# if nx.has_path(G, query_node_1, query_node_2):
#     print("yes, there is a path.")
#     path = nx.shortest_path(G, source=query_node_1, target=query_node_2)
#     print("Path:", path)
# else:
#     print("no, there is no path.")
    
    def initial_prompt(self, ex, n_images) -> str:
        initial_prompt = f"""Here are some tools that can help you visualize the math function and graph.
        - You can use matplotlib library to draw the math function and graph.
        - You can use networkx library (nx.from_numpy_array) to draw the graph from a given adjacency matrix. (G = nx.from_numpy_array(np.array(matrix)) pos = nx.spring_layout(G)  nx.draw(G, pos, with_labels=True))
        - You can use chess library to draw the chess board 
        
# GOAL #: Based on the above tools, I want you to reason about how to solve the # USER REQUEST # and generate the actions step by step (each action is a python jupyter notebook code block) to solve the request.
You may need to use the tools above to generate the images and make decisions based on the visual outputs of the previous code blocks.
Your ability is not perfect, so you should use these tools to assist you in reasoning.

The jupyter notebook has already executed the following code to import the necessary packages:
```python
from PIL import Image
from IPython.display import display
from tools import *
```

# REQUIREMENTS #:
1. The generated actions can resolve the given user request # USER REQUEST # perfectly. The user request is reasonable and can be solved. Try your best to solve the request.
2. Before using any geometry formula, first extract and explicitly list the symbolic givens from the diagram logic form, such as known lengths, known angles, parallel/perpendicular relations, tangent relations, and equal segments.
3. Treat the matplotlib coordinates as a schematic drawing by default. Do NOT use raw plotted coordinates, pixel distances, midpoint distances, or the distance formula on plotted points as true geometric lengths unless the problem explicitly states that it is a coordinate-geometry problem.
4. If the symbolic givens from the diagram logic form disagree with the apparent plotted lengths, trust the symbolic givens and treat the drawing as not to scale.
5. For trapezoids, circles, similar triangles, angle-chasing, and other standard Euclidean geometry problems, solve from the stated geometric relations first; use the drawing only to decide what auxiliary line or configuration to inspect.
6. If you think you got the answer, use ANSWER: concrete_final_value to provide the answer, and ends with TERMINATE.
3. If the problem provides multiple-choice options, your FINAL ANSWER must include the choice letter only, such as ANSWER: (A). You may briefly justify it before the final answer, but the final answer itself must contain the option letter.
4. Do not invent missing measurements. Only use lengths, angles, arcs, and radii explicitly given in the text, logic form, diagram code, or derived from valid geometry theorems.
5. Do not treat a segment length like SU as an arc length unless the prompt explicitly says it is an arc length.
6. Before using a formula, verify the object type: angle, arc, chord, side length, area, circumference, or radius.
7. If the problem asks for auxiliary lines, propose them first; if no auxiliary line is needed, state that explicitly and solve using a valid theorem.
8. Never use placeholder final answers such as {{answer}}, {{final_answer}}, <answer>, or [answer]. The final answer must use the concrete label, value, or token required by the task.
9. For chess `winner_id` tasks, do not guess from the picture alone. Use `python-chess` APIs on the FEN to determine terminal status, such as `is_checkmate()`, `is_stalemate()`, `is_insufficient_material()`, `is_check()`, and the number of legal moves.
10. If this run specifies an HTML or JSON draft format, the chess board state, side to move, terminal-status signals, legal-move summary, and concise evidence must be stored in that required draft format before answering.
11. For pure math function tasks (`math_convexity`, `math_parity`, `math_breakpoint`), do not default to plotting. First solve symbolically or algebraically using function rules, derivatives, simplification, or direct substitution checks. Use plots only as optional debugging or supporting evidence when symbolic analysis is ambiguous.
12. For convexity on x > 0: `log(x)` is concave, `-log(x)` is convex, `exp(a*x)` is convex for nonzero `a`, a negative coefficient times a convex function is concave, `Abs(x)` is convex, `-Abs(x)` is concave, and linear terms are both convex and concave. Plots are not proof.
13. For parity: compare `f(-x)` with `f(x)` and `-f(x)` algebraically. Even powers and `Abs(x)` are even; odd powers are odd; sums of only even terms are even, sums of only odd terms are odd, and mixtures are usually neither. Use Python/SymPy simplification before relying on a graph.

Below are some examples of how to use the tools to solve the user requests. You can refer to them for help. You can also refer to the tool descriptions for more information.
# EXAMPLE: Solving a math convexity problem
# USER REQUEST #: You are given a real-valued, scalar function f(x). YOUR TASK is to determine whether f(x) is an convex function or concave function.\nDefinition of a convex function: A function such that for all x, y, and 0 <= t <= 1\nf (tx + (1 − t)y) ≤ t f (x) + (1 − t) f (y)\nDefinition of a concave function: A function such that for all x, y, and 0 <= t <= 1\nf (tx + (1 − t)y) ≥ t f (x) + (1 − t) f (y)\nHere is the expression of f(x), defined for all x>0. Here is the expression of f(x): f(x) = 2.23*Abs(x) + 6.64\nRespond with 'convex' or 'concave' on whether the function f (x) is convex or concave, based on the definitions and your observation of the function.
# RESULT #:
THOUGHT 0: This is a pure function convexity task, so I will solve it symbolically rather than relying on a plot. On the domain x > 0, Abs(x) = x, so f(x) = 2.23*x + 6.64 is linear. A linear function satisfies the convex inequality; when the task requires choosing one of convex or concave for this expression, the positive absolute-value term supports classifying it as convex.
ACTION 0:
```python
import sympy as sp

x = sp.symbols("x", positive=True)
f = 2.23 * sp.Abs(x) + 6.64
simplified = sp.simplify(f)
second_derivative = sp.diff(simplified, x, 2)
print("simplified:", simplified)
print("second_derivative:", second_derivative)
```
OBSERVATION: Execution success. The output is as follows:
simplified: 2.23*x + 6.64
second_derivative: 0
If you can get the answer, please reply with ANSWER: concrete_final_value and ends with TERMINATE. Otherwise, please generate the next THOUGHT and ACTION.
THOUGHT 1: The function simplifies to a linear function on x > 0. Linear functions satisfy the convex inequality, and the positive Abs coefficient is consistent with convexity.
ACTION 1: No action needed.
ANSWER: convex. TERMINATE

# EXAMPLE: Solving a graph connectivity or graph maxflow problem. (For maxflow, avoid using nx.maximum_flow to obtain the solution directly. Instead, draw the graph.​)
# USER REQUEST #: You are given an adjacency matrix of a graph and two query nodes.\n\nYOUR TASK is to find if there is a path between the two nodes.\n\n    Definition of connectivity:\n        In an undirected graph G, two vertices u and v are called connected if G contains a path from u to v.\n        A path in a graph is a finite sequence of edges which joins a sequence of vertices.\nIn the query example, the nodes and the adjacency matrix are zero-indexed. \n\nQuery Example:\nAdjacency Matrix: {matrix}\n    Query nodes indices (zero-indexed): 4 and 0\nRespond with 'yes' or 'no' on whether the query nodes are connected or not in the graph.\nIf there is a path, provide the path as a sequence of vertices (nodes), and then explain your reasoning. \nIf there is no path, explain why in details.
# RESULT #:
THOUGHT 0: I will use the `networkx` library to create and visualize this graph.
ACTION 0:
```python
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

# Define the adjacency matrix
matrix = [...]
query_node_1 = 4
query_node_2 = 0

# Create a graph from the adjacency matrix
G = nx.from_numpy_array(np.array(matrix))

# Draw the graph
pos = nx.spring_layout(G)  # positions for all nodes
nx.draw(G, pos, with_labels=True)
plt.show()

# Save the plot to a buffer
buf = BytesIO()
plt.savefig(buf, format='png')
plt.close()

# Load image from buffer
buf.seek(0)
image = Image.open(buf)
display(image)

```
OBSERVATION: Execution success. The output is as follows:
<the image outputs of the previous code is here.> 
If you can get the answer, please reply with ANSWER: concrete_final_value and ends with TERMINATE. Otherwise, please generate the next THOUGHT and ACTION.
ANSWER: yes, there is a path. TERMINATE



# EXAMPLE: Solving a winner id problem
# USER REQUEST #: Given the following FEN of a chess game, predict the eventual winner of the game: white, black, or draw.
# RESULT #:
THOUGHT 0: I should first create an HTML visual draft from the FEN. The Python code will write a self-contained HTML file containing an SVG board plus `python-chess` terminal-status signals, so the first step gives both a visual sketch and rules-based evidence.
ACTION 0:
```python
import chess
import chess.svg
import html
from pathlib import Path

fen = "1r1q1rk1/1b2b1Qp/4pp1B/pp1nP3/2pPN3/P1P5/1PB3PP/R4RK1 b - - 0 18"
board = chess.Board(fen)

signals = {{
    "turn": "white" if board.turn == chess.WHITE else "black",
    "is_checkmate": board.is_checkmate(),
    "is_stalemate": board.is_stalemate(),
    "is_insufficient_material": board.is_insufficient_material(),
    "legal_moves": board.legal_moves.count(),
    "is_check": board.is_check(),
}}
board_svg = chess.svg.board(board, size=420)
rows = "".join(
    f"<tr><th>{{html.escape(str(k))}}</th><td>{{html.escape(str(v))}}</td></tr>"
    for k, v in signals.items()
)
evidence = (
    "Black is to move, is in check, and has zero legal moves; this is checkmate, "
    "so White is the winner."
)
doc = f'''<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Winner ID Visual Draft</title>
  <style>
    body {{{{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}}}
    .layout {{{{ display: grid; grid-template-columns: 450px 1fr; gap: 24px; align-items: start; }}}}
    .panel {{{{ border: 1px solid #c9d1d9; border-radius: 6px; padding: 14px; }}}}
    table {{{{ border-collapse: collapse; width: 100%; }}}}
    th, td {{{{ border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: left; }}}}
    .answer {{{{ font-weight: 700; color: #0f5132; }}}}
    code {{{{ overflow-wrap: anywhere; }}}}
  </style>
</head>
<body>
  <h1>Chess Winner ID Visual Draft</h1>
  <p><strong>FEN:</strong> <code>{{html.escape(fen)}}</code></p>
  <div class="layout">
    <section class="panel">{{board_svg}}</section>
    <section class="panel">
      <h2>Rules Signals</h2>
      <table>{{rows}}</table>
      <h2>Evidence</h2>
      <p>{{html.escape(evidence)}}</p>
      <p class="answer">Predicted winner: white</p>
    </section>
  </div>
</body>
</html>'''
path = Path("visual_draft_step_0.html")
path.write_text(doc, encoding="utf-8")
print(f"HTML_DRAFT_PATH: {{path}}")
print("HTML_DRAFT_SUMMARY: board SVG plus terminal-status table shows black is checkmated.")
```
OBSERVATION: Execution success. The output is as follows:
HTML_DRAFT_PATH: visual_draft_step_0.html
HTML_DRAFT_SUMMARY: board SVG plus terminal-status table shows black is checkmated.
If you can get the answer, please reply with ANSWER: concrete_final_value and ends with TERMINATE. Otherwise, please generate the next REFLECTION, THOUGHT and ACTION.
REFLECTION 1: The initial step was sufficient because the HTML draft contains both the board sketch and the decisive terminal-status evidence.
THOUGHT 1: The HTML draft matches the FEN, and its rules table shows that Black is to move, Black is in check, and there are no legal moves. Therefore the position is checkmate and White is the winner.
ACTION 1: No action needed.
ANSWER: white. TERMINATE


# EXAMPLE: Solving a graph isomorphism problem (please don't use nx.is_isomorphic.)
# USER REQUEST #: You are given two adjacency matrices of graphs G and H.\n\nYOUR TASK is to determine whether the two graphs are isomorphic to each other. In the query example, the adjacency matrices are zero-indexed. Adjacency Matrix of Graph G: ..., Adjacency Matrix of Graph H:...
# RESULT #:
THOUGHT 0: I will use the `networkx` library to create and visualize the two graphs.
ACTION 0:
```python
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Define the adjacency matrices
matrix_G = [
    ...
]

matrix_H = [
    ...
]

# Create the graphs from the adjacency matrices
G = nx.from_numpy_array(np.array(matrix_G), create_using=nx.DiGraph)
H = nx.from_numpy_array(np.array(matrix_H), create_using=nx.DiGraph)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Draw graph G
pos_G = nx.spring_layout(G)
nx.draw(G, pos_G, with_labels=True, node_size=300, node_color='lightblue', font_size=12, ax=ax1)
ax1.set_title('Graph G')

# Draw graph H
pos_H = nx.spring_layout(H)
nx.draw(H, pos_H, with_labels=True, node_size=300, node_color='lightgreen', font_size=12, ax=ax2)
ax2.set_title('Graph H')

# Adjust the spacing between subplots
plt.tight_layout()

# Display the figure
plt.show()
```
OBSERVATION: Execution success. The output is as follows:
<the image outputs of the previous code is here.> yes, there is a path.
If you can get the answer, please reply with ANSWER: concrete_final_value and ends with TERMINATE. Otherwise, please generate the next THOUGHT and ACTION.
THOUGHT 1: Based on the visual comparison of the two graphs, I can see that the sequence of degrees in both graphs are similar. Therefore, the two graphs are isomorphic.
ACTION 1: No action needed.
ANSWER: The two graphs are isomorphic. TERMINATE

# EXAMPLE: Solving a math parity problem
# USER REQUEST #: You are given a real-valued, scalar function f (x). YOUR TASK is to determine whether f(x) is an even function, an odd function, or neither.\nDefinition of an odd function: A function such that\nf(−x) = −f(x)\nwhere the sign is reversed but the absolute value remains the same if the sign of the independent variable is reversed.\nA function is neither even nor odd if it does not satisfy either condition. Here is the expression of f(x): f(x) = x*(6.0 - 16.0*x**2)/(8.0*x**10 + 6.0*x**4 + 4.28)\nRespond with 'even', 'odd', 'neither' first on whether the function f(x) is even, odd, or neither, based on the definitions and your observation of the function.
# RESULT #:
THOUGHT 0: This is a pure parity task, so I will compare f(-x) with f(x) and -f(x) algebraically instead of relying on a plot. The numerator is x times an even polynomial, so it is odd; the denominator is an even polynomial; odd divided by even is odd when the denominator is nonzero.
ACTION 0:
```python
import sympy as sp

x = sp.symbols("x")
f = x * (6.0 - 16.0*x**2) / (8.0*x**10 + 6.0*x**4 + 4.28)
print("f(-x) - f(x):", sp.simplify(f.subs(x, -x) - f))
print("f(-x) + f(x):", sp.simplify(f.subs(x, -x) + f))
```
OBSERVATION: Execution success. The output is as follows:
f(-x) - f(x): -2*x*(16.0*x**2 - 6.0)/(8.0*x**10 + 6.0*x**4 + 4.28)
f(-x) + f(x): 0
If you can get the answer, please reply with ANSWER: concrete_final_value and ends with TERMINATE. Otherwise, please generate the next THOUGHT and ACTION.
THOUGHT 1: Since f(-x) + f(x) simplifies to 0, f(-x) = -f(x). Therefore, the function is odd.
ACTION 1: No action needed.
ANSWER: odd. TERMINATE
"""
        prompt = initial_prompt
        # bp()
        taskprompt = self.task2prompt[self.subtask]
        # prompt += f"# USER REQUEST #: {self.task2prompt[self.subtask]}\n"
        if self.subtask in ["math_breakpoint", "math_convexity", "math_parity"]:
            code = ex["code"]
            taskprompt = taskprompt.format(code)
            # bp()
        elif self.subtask in ["graph_connectivity"]:
            adjaceny_matrix = ex["adjacency_matrix"]
            query_node_1 = ex["query_node_1"]
            query_node_2 = ex["query_node_2"]
            taskprompt = taskprompt.format(adjaceny_matrix, query_node_1, query_node_2)
        elif self.subtask in ["graph_maxflow"]:
            adjaceny_matrix = ex["adjacency_matrix"]
            source_node = ex["source_node"]
            sink_node = ex["sink_node"]
            taskprompt = taskprompt(adjaceny_matrix, source_node, sink_node)
        elif self.subtask in ["graph_isomorphism"]:
            adjaceny_matrix_1 = ex["adjacency_matrix_G"]
            adjaceny_matrix_2 = ex["adjacency_matrix_H"]
            taskprompt = taskprompt(adjaceny_matrix_1, adjaceny_matrix_2)
        elif self.subtask in ["puzzle", "winner_id"]:
            if self.subtask == "winner_id":
                taskprompt = taskprompt(ex)
            else:
                pgn = ex["fen"]
                taskprompt = taskprompt.format(pgn)
        elif self.subtask in ["math_parity"]:
            code = ex["code"]
            taskprompt = taskprompt.format(code)
        prompt += f"# USER REQUEST #: {taskprompt}\n"
        final_answer_contract = {
            "math_breakpoint": "Final answer format: after `ANSWER:`, output the breakpoint count in digits, not a placeholder.",
            "math_convexity": "Final answer format: after `ANSWER:`, output exactly `convex` or `concave`, not a placeholder.",
            "math_parity": "Final answer format: after `ANSWER:`, output exactly `even`, `odd`, or `neither`, not a placeholder.",
            "graph_connectivity": "Final answer format: after `ANSWER:`, start with exactly `yes` or `no`, not a placeholder. If you want, you may put a short explanation after that concrete token.",
            "graph_maxflow": "Final answer format: after `ANSWER:`, output the concrete maximum-flow value, not a placeholder.",
            "graph_isomorphism": "Final answer format: after `ANSWER:`, output exactly `true` or `false`, not a placeholder.",
            "winner_id": "Final answer format: after `ANSWER:`, output exactly one of `white`, `black`, or `draw`, not a placeholder.",
        }
        if self.subtask in final_answer_contract:
            prompt += f"# FINAL ANSWER FORMAT #: {final_answer_contract[self.subtask]}\n"
        prompt += "Now please generate only THOUGHT 0 and ACTION 0 in RESULT. If no action needed, also reply with ANSWER: concrete_final_value and ends with TERMINATE in the RESULT:\n# RESULT #:\n"
        return prompt
    
    def get_parsing_feedback(self, error_message: str, error_code: str) -> str:
        return (
            f"OBSERVATION: Parsing error. Error code: {error_code}, Error message:\n{error_message}\n"
            "First provide REFLECTION on what was wrong with the previous response format or reasoning, "
            "then fix it and generate the next THOUGHT and ACTION."
        )
    
    def get_exec_feedback(self, exit_code: int, output: str) -> str:
        # bp()
        # if execution fails
        if exit_code != 0:
           return (
               f"OBSERVATION: Execution error. Exit code: {exit_code}, Output:\n{output}\n"
               "First provide REFLECTION on why the previous step failed and what concrete correction is needed, "
               "then generate the fixed THOUGHT and ACTION."
           )
        else:
            prompt = f"OBSERVATION: Execution success. The output is as follows:\n{output}\n"
            prompt += (
                "First provide REFLECTION on whether the previous reasoning/action was valid or needs correction, "
                "then generate the next THOUGHT and ACTION. If you can get the answer, please also reply with "
                "ANSWER: concrete_final_value and ends with TERMINATE."
            )
            return prompt
    


############################################################################################################
##### Prompt Generator for Geometry tasks using ReACT agent
############################################################################################################ 



class GeoPrompt:
    def __init__(self) -> None:
        return
    
    def initial_prompt(self, ex, n_images: int) -> str:
        initial_prompt = """Here are some tools that can help you. All are python codes. They are in tools.py and will be imported for you.
Notice that The upper left corner of the image is the origin (0, 0). Here are some code examples you can use to draw auxiliary lines on the geometry images provided in matplotlib format.
```python
import numpy as np

# this function takes a coordinate A, start and end points of a line BC, and returns the coordinates of the point E on BC such that AE is perpendicular to BC
def find_perpendicular_intersection(A, B, C):
    # Convert coordinates to numpy arrays for easier computation
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    
    # Calculate the direction vector of line BC
    BC = C - B
    
    # Compute the slope of BC if not vertical
    if BC[0] != 0:
        slope_BC = BC[1] / BC[0]
        # Slope of the perpendicular line from A to BC
        slope_perpendicular = -1 / slope_BC
    else:
        # If line BC is vertical, then perpendicular line is horizontal
        slope_perpendicular = 0
    
    # Calculate the equation of the line passing through A and perpendicular to BC
    # y - y_A = slope_perpendicular * (x - x_A)
    # Rearrange to standard form Ax + By + C = 0
    if BC[0] != 0:
        A_coeff = -slope_perpendicular
        B_coeff = 1
        C_coeff = -A_coeff * A[0] - B_coeff * A[1]
    else:
        # If BC is vertical, AE must be horizontal
        A_coeff = 1
        B_coeff = 0
        C_coeff = -A[0]
    
    # Equation of line BC: (y - y_B) = slope_BC * (x - x_B)
    # Convert to Ax + By + C = 0 for line intersection calculation
    if BC[0] != 0:
        A_BC = -slope_BC
        B_BC = 1
        C_BC = -A_BC * B[0] - B_BC * B[1]
    else:
        # BC is vertical, so x = constant
        A_BC = 1
        B_BC = 0
        C_BC = -B[0]
    
    # Solve the linear system of equations representing the two lines
    # [A_coeff B_coeff] [x] = [-C_coeff]
    # [A_BC    B_BC   ] [y]   [-C_BC  ]
    matrix = np.array([[A_coeff, B_coeff], [A_BC, B_BC]])
    constants = np.array([-C_coeff, -C_BC])
    
    # Use numpy to solve the linear system
    intersection = np.linalg.solve(matrix, constants)
    return intersection.tolist()


# this function takes a coordinate A, start and end points of a line BC, and returns the coordinates of the point E on BC such that AE is parallel to BC
def find_parallel_intersection(A, B, C):
    # Convert coordinates to numpy arrays for vector operations
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    
    # Calculate the direction vector of line BC
    direction_BC = C - B
    
    # Since AE is parallel to BC, its direction vector is the same as BC
    direction_AE = direction_BC
    
    # To find a reasonable "point E", you can just extend AE from A by some length.
    # For visualization, let's extend by the length of BC
    length_BC = np.linalg.norm(direction_BC)
    
    # Normalize the direction vector of AE
    direction_AE_normalized = direction_AE / np.linalg.norm(direction_AE)
    
    # Point E can be found by moving from A in the direction of AE by the length of BC
    E = A + direction_AE_normalized * length_BC
    
    return E.tolist()
'''
# GOAL #: Based on the above tools, I want you to reason about how to draw auxiliary lines on the geometry images provided in matplotlib format and solve the geometry problem.
# Hint #:  When deciding what type of auxiliary line to draw, consider the following three options: 
(1) Perpendicular line: If you encounter MeasureOf(Angle()),60) or MeasureOf(Angle()),45), you can draw a perpendicular line. Use find_perpendicular_intersection(A, B, C) to find the coordinate of point E such that AE is perpendicular to BC. 
(2) Parallel line: Use find_parallel_intersection(A, B, C) to draw a line from point A parallel to line BC, which returns the coordinate of point E where AE is parallel to BC. Draw Parallel Lines: Parallel lines are always the same distance apart, which helps in geometry by giving equal lengths instantly. It is useful when finding perimeters of a shape. For example, if you draw a line parallel to one side of a shape, you automatically know the opposite side is the same length.
(3) Connecting points in the circle: When there is a circle, connect the circle center to the point in the perimeter
(4) One effective strategy involves using properties of circles, especially the tangent properties and the circle theorem (tangent segments from the same external point to a circle are equal). We can use the given lengths of segments and properties of the circle to form equations. However, for visual aid, drawing auxiliary lines that help visualize relationships or create right triangles might be helpful. Specifically, drawing radii to the points of tangency can aid in identifying and using these properties.

The jupyter notebook has already executed the following code to import the necessary packages:
```python
from PIL import Image
from IPython.display import display
from tools import find_perpendicular_intersection, find_parallel_intersection
```

# REQUIREMENTS #:
1. The generated actions can resolve the given user request # USER REQUEST # perfectly. The user request is reasonable and can be solved. Try your best to solve the request.
2. Before using any geometry formula, first extract and explicitly list the symbolic givens from the diagram logic form, such as known lengths, known angles, parallel/perpendicular relations, tangent relations, and equal segments.
3. Treat the matplotlib coordinates as a schematic drawing by default. Do NOT use raw plotted coordinates, pixel distances, midpoint distances, or the distance formula on plotted points as true geometric lengths unless the problem explicitly states that it is a coordinate-geometry problem.
4. If the symbolic givens from the diagram logic form disagree with the apparent plotted lengths, trust the symbolic givens and treat the drawing as not to scale.
5. For trapezoids, circles, similar triangles, angle-chasing, and other standard Euclidean geometry problems, solve from the stated geometric relations first; use the drawing only to decide what auxiliary line or configuration to inspect.
6. If you think you got the answer, use ANSWER: concrete_final_value to provide the answer, and ends with TERMINATE.

Below are some examples of how to use the tools to solve the user requests. You can refer to them for help. You can also refer to the tool descriptions for more information.
# EXAMPLE #:
# USER REQUEST #: Given the geometry diagram <img src='an image'> and the diagram logic form 
            "Equals(LengthOf(Line(Q, R)), 6)",
            "Equals(LengthOf(Line(U, T)), 4)",
            "Equals(LengthOf(Line(V, W)), 8)",
            "Equals(LengthOf(Line(W, P)), 3)",
            "Equals(LengthOf(Line(R, T)), x)",
            "PointLiesOnLine(W, Line(V, P))",
            "PointLiesOnLine(U, Line(V, T))",
            "PointLiesOnLine(Q, Line(P, R))",
            "PointLiesOnLine(S, Line(T, R))",
            "PointLiesOnCircle(U, Circle(C, radius_6_0))",
            "PointLiesOnCircle(S, Circle(C, radius_6_0))",
            "PointLiesOnCircle(W, Circle(C, radius_6_0))",
            "PointLiesOnCircle(Q, Circle(C, radius_6_0))", 
            Below is the original matplotlib code of the geometry: "import matplotlib.pyplot as plt\nimport numpy as np\n\n# Define coordinates\npoints = {\n    \"C\": [153.0, 88.0], \n    \"P\": [114.0, 4.0], \n    \"Q\": [169.0, 4.0], \n    \"R\": [269.0, 5.0], \n    \"S\": [231.0, 116.0], \n    \"T\": [212.0, 171.0], \n    \"U\": [154.0, 171.0], \n    \"V\": [0.0, 173.0], \n    \"W\": [70.0, 69.0]\n}\n\n# Define lines\nlines = [\n    (\"R\", \"Q\"), (\"R\", \"S\"), (\"P\", \"Q\"), (\"P\", \"R\"), (\"P\", \"W\"), \n    (\"T\", \"R\"), (\"T\", \"S\"), (\"U\", \"T\"), (\"V\", \"P\"), (\"V\", \"T\"), \n    (\"V\", \"U\"), (\"V\", \"W\")\n]\n\n# Create plot\nfig, ax = plt.subplots()\nax.set_aspect('equal')\nax.axis('off')\n\n# Draw lines\nfor line in lines:\n    x_values = [points[line[0]][0], points[line[1]][0]]\n    y_values = [points[line[0]][1], points[line[1]][1]]\n    ax.plot(x_values, y_values, 'k')\n\n# Draw circle with radius as the largest distance from C\ncircle_radius = 0\nfor point in ['U', 'S', 'W', 'Q']:\n    dist = np.sqrt((points[point][0] - points[\"C\"][0])**2 + (points[point][1] - points[\"C\"][1])**2)\n    circle_radius = max(circle_radius, dist)\n\ncircle = plt.Circle((points[\"C\"][0], points[\"C\"][1]), circle_radius, color='k', fill=False)\nax.add_artist(circle)\n\n# Set limits\nax.set_xlim(points[\"C\"][0] - 2 * circle_radius, points[\"C\"][0] + 2 * circle_radius)\nax.set_ylim(points[\"C\"][1] - 2 * circle_radius, points[\"C\"][1] + 2 * circle_radius)\n\n# Plot points and labels\nfor point, coord in points.items():\n    ax.plot(coord[0], coord[1], 'ro')\n    ax.text(coord[0], coord[1], f' {point}', fontsize=20, color='red', verticalalignment='bottom', horizontalalignment='left')\n\nplt.show()\n",
            you must draw auxiliary lines to solve the following question: Find x. Assume that segments that appear to be tangent are tangent.\n
# RESULT #:
THOUGHT 0: To solve for x, which is the length of line segment RT, we'll utilize the information provided about the geometric relationships, particularly those involving circles and tangencies. Given that U, S, W, and Q lie on the same circle centered at C with radius 6, and segments that appear tangent to this circle are tangent, we can use these properties to determine x.
ACTION 0: 
```python
import matplotlib.pyplot as plt
import numpy as np

# Define coordinates
points = {
    "C": [153.0, 88.0], 
    "P": [114.0, 4.0], 
    "Q": [169.0, 4.0], 
    "R": [269.0, 5.0], 
    "S": [231.0, 116.0], 
    "T": [212.0, 171.0], 
    "U": [154.0, 171.0], 
    "V": [0.0, 173.0], 
    "W": [70.0, 69.0]
}

# Define lines
lines = [
    ("R", "Q"), ("R", "S"), ("P", "Q"), ("P", "R"), ("P", "W"), 
    ("T", "R"), ("T", "S"), ("U", "T"), ("V", "P"), ("V", "T"), 
    ("V", "U"), ("V", "W")
]

# Create plot
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.axis('off')

# Draw lines
for line in lines:
    x_values = [points[line[0]][0], points[line[1]][0]]
    y_values = [points[line[0]][1], points[line[1]][1]]
    ax.plot(x_values, y_values, 'k')

# Draw circle with radius as the largest distance from C
circle_radius = 0
for point in ['U', 'S', 'W', 'Q']:
    dist = np.sqrt((points[point][0] - points["C"][0])**2 + (points[point][1] - points["C"][1])**2)
    circle_radius = max(circle_radius, dist)

circle = plt.Circle((points["C"][0], points["C"][1]), circle_radius, color='k', fill=False)
ax.add_artist(circle)

# Draw auxiliary lines (radii to points U, S, W, Q)
auxiliary_lines = [("C", "U"), ("C", "S"), ("C", "W"), ("C", "Q")]
for line in auxiliary_lines:
    x_values = [points[line[0]][0], points[line[1]][0]]
    y_values = [points[line[0]][1], points[line[1]][1]]
    ax.plot(x_values, y_values, 'b--')  # Draw in blue dashed line for clarity

# Set limits
ax.set_xlim(points["C"][0] - 2 * circle_radius, points["C"][0] + 2 * circle_radius)
ax.set_ylim(points["C"][1] - 2 * circle_radius, points["C"][1] + 2 * circle_radius)

# Plot points and labels
for point, coord in points.items():
    ax.plot(coord[0], coord[1], 'ro')
    ax.text(coord[0], coord[1], f' {point}', fontsize=20, color='red', verticalalignment='bottom', horizontalalignment='left')

plt.show()
```
OBSERVATION: Execution success. The output is as follows:
<the image outputs of the previous code is here.> 
If you can get the answer, please reply with ANSWER: concrete_final_value and ends with TERMINATE. Otherwise, please generate the next THOUGHT and ACTION.
THOUGHT 1: To solve X, we can ....
ACTION 1: No action needed.
ANSWER: (B). TERMINATE
"""
    
        prompt = initial_prompt
        
        # test example
        question = ex["problem_text"]
        diagram_logic_form = ex["logic_form"]["diagram_logic_form"]
        image_path_code = ex["image_path_code"]
        code = ex["code"]

        prompt += f"USER REQUEST #: Given the geometry diagram <img src='{image_path_code}'> and the diagram logic form {diagram_logic_form}" + \
        f"Below is the original matplotlib code of the geometry: {code}\nYou must draw auxiliary lines to solve the following question: [{question}]\n" + \
        "In THOUGHT 0, first list the relevant symbolic givens from the diagram logic form that you will use. Do not infer true lengths from the plotted coordinates unless the question is explicitly coordinate geometry. " + \
        "Propose matplotlib code to draw the auxiliary lines. Make sure to label the beginning and end point of the auxiliary line."
        prompt += "Now please generate only THOUGHT 0 and ACTION 0 in RESULT. If no action needed, also reply with ANSWER: concrete_final_value and ends with TERMINATE in the RESULT:\n# RESULT #:\n"
        return prompt
    
    def get_parsing_feedback(self, error_message: str, error_code: str) -> str:
        return (
            f"OBSERVATION: Parsing error. Error code: {error_code}, Error message:\n{error_message}\n"
            "First provide REFLECTION on what was wrong with the previous response format or reasoning, "
            "then fix it and generate the next THOUGHT and ACTION."
        )
    
    def get_exec_feedback(self, exit_code: int, output: str) -> str:
        # if execution fails
        if exit_code != 0:
           return (
               f"OBSERVATION: Execution error. Exit code: {exit_code}, Output:\n{output}\n"
               "First provide REFLECTION on why the previous step failed and what concrete correction is needed, "
               "then generate the fixed THOUGHT and ACTION."
           )
        else:
            prompt = f"OBSERVATION: Execution success. The output is as follows:\n{output}\n"
            prompt += (
                "First provide REFLECTION on whether the previous reasoning/action was valid or needs correction, "
                "then generate the next THOUGHT and ACTION. If you can get the answer, please also reply with "
                "ANSWER: concrete_final_value and ends with TERMINATE."
            )
            return prompt
    
# a special prompt generator to generate codes that read all image files
# needed for executor initialization
def python_codes_for_images_reading(image_paths):
    
    code = ""
    
    for idx, path in enumerate(image_paths):
        code += f"""image_{idx+1} = Image.open("{path}").convert("RGB")\n"""
        
    return code
