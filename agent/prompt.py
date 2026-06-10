
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
If you think you get the answer to the intial user request, you can reply with "ANSWER: <your answer>" and ends with "TERMINATE".
"""

HTML_VISUAL_ASSISTANT_MESSAGE = """You are a structured-vision reasoning planner for text-to-image prompts.
Your job is to reason through a prompt using HTML visual drafts instead of generated images or JSON.
You are coding in a Python jupyter notebook environment. Each ACTION must be one executable ```python``` block unless you are giving the final ANSWER.
Do not generate intermediate images. Do not output JSON as the visual draft format.

For each non-final turn:
1. Start with REFLECTION. On the first turn, say this is the initial planning step.
2. Then write THOUGHT with the current reasoning.
3. Then write ACTION with Python code that writes a self-contained HTML file in the current working directory.

Each HTML draft should function as a structured visual sketchpad:
- include a visible title and the original prompt;
- include a Thinking Text section explaining the reasoning step;
- include an Objects / Entities section;
- include an Attributes section when relevant;
- include a Relations / Constraints section for spatial, causal, temporal, physical, cultural, or textual-image constraints;
- include a Layout / Visual Draft section using semantic HTML, CSS boxes, tables, and/or inline SVG arrows when useful;
- include an Open Issues / Revision Targets section when uncertainty remains.

The HTML must be human-readable and self-contained. Use escaped text when embedding user content.
The code should print the written HTML path and a compact textual summary of what the draft contains.

Use local revision rather than regenerating the whole plan when the previous draft only needs a focused fix.
If you have enough confidence, reply with:
ANSWER: <final T2I generative prompt or final reasoning result> TERMINATE
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

def sketch_canvas(items, output_path, width=960, height=660, title=""):
    \"\"\"Create a spatial HTML/SVG annotation canvas that places multiple images side-by-side,
    connects them with arrows, and adds bounding-box overlays, labels, and callout notes.
    Use this when you want to annotate or compare images in a sketch-pad style rather than
    displaying them one by one.  The output is a self-contained HTML file with embedded images.

    Supported item types in `items` list:
      - image:  {\"type\":\"image\", \"id\":\"<id>\", \"path\":\"<path>\", \"x\":<px>, \"y\":<px>, \"width\":<px>,
                 \"panel\":True, \"panel_title\":\"<label>\", \"panel_color\":\"<hex>\"}
      - arrow:  {\"type\":\"arrow\", \"x1\":<px>, \"y1\":<px>, \"x2\":<px>, \"y2\":<px>,
                 \"style\":\"solid\"|\"dashed\"|\"dotted\", \"color\":\"<hex>\", \"label\":\"<text>\", \"curve\":0.3}
      - bbox:   canvas-absolute: {\"type\":\"bbox\", \"x\":<px>, \"y\":<px>, \"w\":<px>, \"h\":<px>,
                                  \"color\":\"<hex>\", \"label\":\"<text>\"}
                OR image-relative: {\"type\":\"bbox\", \"image_id\":\"<id>\", \"rel\":[rx,ry,rw,rh],
                                    \"color\":\"<hex>\", \"label\":\"<text>\"}
      - text:   {\"type\":\"text\", \"x\":<px>, \"y\":<px>, \"content\":\"<text>\",
                 \"size\":13, \"bg\":\"<hex>\"}
      - circle: {\"type\":\"circle\", \"cx\":<px>, \"cy\":<px>, \"r\":<px>,
                 \"color\":\"<hex>\", \"fill\":\"none\", \"label\":\"<text>\"}

    Args:
        items (List[dict]): list of annotation items (see above).
        output_path (str): path to write the HTML file (e.g. \"sketch_step1.html\").
        width (int): canvas width in pixels. Default 960.
        height (int): canvas height in pixels. Default 660.
        title (str): optional title shown above the canvas.

    Returns:
        str: absolute path to the written HTML file.

    Example:
        # Show original image and a zoom crop side-by-side with annotation
        path = sketch_canvas([
            {\"type\": \"image\", \"id\": \"orig\", \"path\": \"image_1.jpg\",
             \"x\": 20, \"y\": 80, \"width\": 420, \"panel\": True, \"panel_title\": \"Original\"},
            {\"type\": \"image\", \"id\": \"zoom\", \"path\": \"zoom_crop.png\",
             \"x\": 500, \"y\": 80, \"width\": 340, \"panel\": True, \"panel_title\": \"Zoom\"},
            {\"type\": \"arrow\", \"x1\": 440, \"y1\": 220, \"x2\": 500, \"y2\": 220,
             \"style\": \"dashed\", \"color\": \"#e11d48\"},
            {\"type\": \"bbox\", \"image_id\": \"zoom\", \"rel\": [0.05, 0.35, 0.55, 0.30],
             \"color\": \"#e11d48\", \"label\": \"target\"},
            {\"type\": \"text\", \"x\": 20, \"y\": 520,
             \"content\": \"The 22% value is in the left donut.\", \"bg\": \"#fef9c3\"},
        ], output_path=\"sketch_step1.html\")
        print(f\"SKETCH_PATH: {path}\")
    \"\"\"

# --- When to use sketch_canvas in your reasoning ---
# Use sketch_canvas when you have MULTIPLE images to compare or relate spatially:
#   - After cropping a region: place original + crop side-by-side, connect with a dashed arrow
#   - When comparing two images: annotate differences with text notes on each image
#   - When you have key findings: place the evidence image and add a text note explaining what you see
# Do NOT use bbox or circle items -- spatial coordinates cannot be reliably inferred.
# Only use: image, arrow, text.
# IMPORTANT: Print the returned path so it is captured as a generated file:
#   canvas_path = sketch_canvas([...], output_path=\"canvas_step1.html\")
#   print(f\"SKETCH_CANVAS: {canvas_path}\")
# The canvas and its annotations will be automatically incorporated into the
# next-step memory so you can build on it in subsequent reasoning steps.
```
# GOAL #: Based on the above tools, I want you to reason about how to solve the # USER REQUEST # and generate the actions step by step (each action is a python jupyter notebook code block) to solve the request.
You may need to use the tools above to process the images and make decisions based on the visual outputs of the previous code blocks.
Your visual ability is not perfect, so you should use these tools to assist you in reasoning about the images.

The jupyter notebook has already executed the following code to import the necessary packages:
```python
from PIL import Image
from IPython.display import display
from tools import *
from sketch_canvas import sketch_canvas
```

# REQUIREMENTS #:
1. The generated actions can resolve the given user request # USER REQUEST # perfectly. The user request is reasonable and can be solved. Try your best to solve the request.
2. The arguments of a tool must be the same number, modality, and format specified in # TOOL LIST #;
3. If you think you got the answer, use ANSWER: <your answer> to provide the answer, and ends with TERMINATE.
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
If you can get the answer, please reply with ANSWER: <your answer> and ends with TERMINATE. Otherwise, please generate the next THOUGHT and ACTION.
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
If you can get the answer, please reply with ANSWER: <your answer> and ends with TERMINATE. Otherwise, please generate the next THOUGHT and ACTION.
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
        prompt += "Now please generate only THOUGHT 0 and ACTION 0 in RESULT. If no action needed, also reply with ANSWER: <your answer> and ends with TERMINATE in the RESULT:\n# RESULT #:\n"
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
                "ANSWER: <your answer> and ends with TERMINATE."
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
    """Prompt generator for StruVis-style HTML visual drafts.

    This keeps the VisualSketchpad planner/executor loop but replaces
    intermediate image/JSON sketches with self-contained HTML artifacts.
    """

    def initial_prompt(self, query: str, n_images: int = 0) -> str:
        return f"""# USER REQUEST #
{query}

# GOAL #
Reason about this text-to-image generation request using StruVis-style thinking:
1. Decompose the prompt into explicit and hidden visual constraints.
2. Build an HTML structured-vision draft, not JSON and not an image.
3. Use the draft to check object coverage, attributes, relations, layout, physical/cultural/scientific consistency, and text-image design constraints.
4. Revise the HTML draft locally if needed.
5. Finish with a final generative prompt suitable for a text-to-image model.

# ACTION FORMAT #
For every non-final step, return exactly one Python code block. The code must write an HTML file such as `visual_draft_step_0.html` and print:
- HTML_DRAFT_PATH: <path>
- HTML_DRAFT_SUMMARY: <short summary>

# HTML DRAFT REQUIREMENTS #
Use semantic HTML, CSS, and optional inline SVG. Do not use JSON as the draft representation.
The HTML should make visual relationships inspectable through sections, diagrams, boxes, arrows, and labels.
Prefer a compact but complete draft over a long narrative. Use the following sections when applicable:
- Thinking Text
- Objects / Entities
- Attributes
- Relations / Constraints
- Layout / Visual Draft
- Open Issues / Revision Targets

# FINAL FORMAT #
When the draft is complete, respond with `ANSWER: ... TERMINATE`.
"""

    def get_parsing_feedback(self, error_message: str, error_code: str) -> str:
        return (
            "OBSERVATION: The previous response did not contain one executable Python code block.\n"
            f"Parser message: {error_message}\n"
            "Continue with REFLECTION, THOUGHT, and exactly one ACTION code block that writes the HTML draft. "
            "If the draft is complete, use ANSWER: ... TERMINATE."
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
            "If it is sufficient, provide ANSWER: <final generative prompt> TERMINATE."
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
3. ABSOLUTE RULE — matplotlib coordinates are for DRAWING ONLY, NEVER for computing lengths:
   The `points = {...}` dictionary contains DISPLAY-ONLY pixel coordinates for rendering the schematic. They are NOT the actual geometric values.
   FORBIDDEN (always produces wrong answers):
     np.linalg.norm(np.array(points["A"]) - np.array(points["B"]))  # pixel distance ≠ geometric length
     abs(points["A"][1] - points["D"][1])                           # pixel y-diff ≠ geometric length
   CORRECT: derive ALL lengths from the symbolic logic form using math/trigonometry only:
     import math; AC = BC * math.sin(math.radians(angle_B))
4. If the symbolic givens from the diagram logic form disagree with the apparent plotted lengths, trust the symbolic givens and treat the drawing as not to scale.
5. For trapezoids, circles, similar triangles, angle-chasing, and other standard Euclidean geometry problems, solve from the stated geometric relations first; use the drawing only to decide what auxiliary line or configuration to inspect.
6. MANDATORY SANITY CHECK: verify your computed value matches one of the given choices. If it does not match any choice, you used pixel coordinates instead of geometric formulas — discard and resolve using only symbolic givens.
7. If you think you got the answer, use ANSWER: <your answer> to provide the answer, and ends with TERMINATE.
3. If the problem provides multiple-choice options, your FINAL ANSWER must include the choice letter only, such as ANSWER: (A). You may briefly justify it before the final answer, but the final answer itself must contain the option letter.
4. Do not invent missing measurements. Only use lengths, angles, arcs, and radii explicitly given in the text, logic form, diagram code, or derived from valid geometry theorems.
5. Do not treat a segment length like SU as an arc length unless the prompt explicitly says it is an arc length.
6. Before using a formula, verify the object type: angle, arc, chord, side length, area, circumference, or radius.
7. If the problem asks for auxiliary lines, propose them first; if no auxiliary line is needed, state that explicitly and solve using a valid theorem.
8. For chess `winner_id` tasks, ACTION 0 MUST: (a) render the board as a PNG image using `chess.svg.board()` + `cairosvg.svg2png()` and `display()` it so it becomes visual evidence in memory, AND (b) print python-chess terminal-status signals (is_checkmate, is_stalemate, legal_moves, is_check), AND (c) write an HTML draft and print `HTML_DRAFT_PATH: <path>`.
9. For chess `winner_id` tasks, the board PNG must be rendered using `chess.svg` + `cairosvg` (NOT matplotlib Unicode) and shown via `display(PIL_image)` — this is required so the image appears in the memory canvas. The HTML draft must also include the SVG board, FEN, side-to-move, and terminal-status signals.
10. For chess `winner_id` tasks, after the initial HTML draft, every additional non-trivial action should, when feasible, write an updated HTML draft tied to the current reasoning step, such as check status, legal-move information, attacked-square overlays, material balance, king safety, or another board-centered visualization.
11. For chess `winner_id` tasks, do not guess from the picture alone. After writing the HTML board draft, use `python-chess` APIs on the FEN to determine terminal status, such as `is_checkmate()`, `is_stalemate()`, `is_insufficient_material()`, `is_check()`, and the number of legal moves, and combine that rules-based evidence with the HTML draft before answering.

Below are some examples of how to use the tools to solve the user requests. You can refer to them for help. You can also refer to the tool descriptions for more information.
# EXAMPLE: Solving a math convexity problem
# USER REQUEST #: You are given a real-valued, scalar function f(x). YOUR TASK is to determine whether f(x) is an convex function or concave function.\nDefinition of a convex function: A function such that for all x, y, and 0 <= t <= 1\nf (tx + (1 − t)y) ≤ t f (x) + (1 − t) f (y)\nDefinition of a concave function: A function such that for all x, y, and 0 <= t <= 1\nf (tx + (1 − t)y) ≥ t f (x) + (1 − t) f (y)\nHere is the expression of f(x), defined for all x>0. Here is the expression of f(x): f(x) = 2.23*Abs(x) + 6.64\nRespond with 'convex' or 'concave' on whether the function f (x) is convex or concave, based on the definitions and your observation of the function.
# RESULT #:
THOUGHT 0: I will plot the function f(x) = 2.23*Abs(x) + 6.64 to visualize the function.
ACTION 0:
```python
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Define the function f(x)
def f(x):
    return 2.23 * np.abs(x) + 6.64

# Generate x values from -10 to 10
x = np.linspace(-10, 10, 400)

# Compute y values using the function
y = f(x)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='f(x) = 2.23*|x| + 6.64')
plt.title('Plot of the function f(x) = 2.23*|x| + 6.64')

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
If you can get the answer, please reply with ANSWER: <your answer> and ends with TERMINATE. Otherwise, please generate the next THOUGHT and ACTION.
THOUGHT 1: For any two points on the graph shown in the image, the line segment connecting them lies above or on the graph. Therefore, the function is convex 
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
If you can get the answer, please reply with ANSWER: <your answer> and ends with TERMINATE. Otherwise, please generate the next THOUGHT and ACTION.
ANSWER: yes, there is a path. TERMINATE



# EXAMPLE: Solving a winner id problem
# USER REQUEST #: Given the following FEN of a chess game, predict the eventual winner of the game: white, black, or draw.
# RESULT #:
THOUGHT 0: I should first render the board as a PNG image (so it becomes visual evidence) and collect python-chess terminal-status signals to determine the winner.
ACTION 0:
```python
import chess, chess.svg, html, io
from cairosvg import svg2png
from pathlib import Path
from PIL import Image as PILImage
from IPython.display import display

fen = "1r1q1rk1/1b2b1Qp/4pp1B/pp1nP3/2pPN3/P1P5/1PB3PP/R4RK1 b - - 0 18"
board = chess.Board(fen)

# ── terminal-status signals ──────────────────────────────────────────────
signals = {{
    "turn": "white" if board.turn == chess.WHITE else "black",
    "is_checkmate": board.is_checkmate(),
    "is_stalemate": board.is_stalemate(),
    "is_insufficient_material": board.is_insufficient_material(),
    "legal_moves": board.legal_moves.count(),
    "is_check": board.is_check(),
}}
print("signals:", signals)

# ── render board as a PNG image so it appears as visual evidence ─────────
svg_data = chess.svg.board(board, size=400)
png_bytes = svg2png(bytestring=svg_data)
board_img = PILImage.open(io.BytesIO(png_bytes)).copy()
display(board_img)   # tracked by executor → shows in memory HTML

# ── write HTML draft for detailed inspection ─────────────────────────────
board_svg = chess.svg.board(board, size=420)
rows = "".join(f"<tr><th>{{html.escape(str(k))}}</th><td>{{html.escape(str(v))}}</td></tr>"
               for k, v in signals.items())
doc = f'''<!doctype html><html><body>
  <h1>Chess Winner ID</h1>
  <p><strong>FEN:</strong> <code>{{html.escape(fen)}}</code></p>
  <div style="display:grid;grid-template-columns:450px 1fr;gap:24px">
    <div>{{board_svg}}</div>
    <div><table>{{rows}}</table></div>
  </div></body></html>'''
path = Path("visual_draft_step_0.html")
path.write_text(doc, encoding="utf-8")
print(f"HTML_DRAFT_PATH: {{path}}")
print("HTML_DRAFT_SUMMARY: board PNG + SVG with terminal-status signals; black checkmated.")
```
OBSERVATION: Execution success. The output is as follows:
signals: {{'turn': 'black', 'is_checkmate': True, 'is_stalemate': False, 'is_insufficient_material': False, 'legal_moves': 0, 'is_check': True}}
<PIL.PngImagePlugin.PngImageFile image mode=RGB size=400x400>
HTML_DRAFT_PATH: visual_draft_step_0.html
HTML_DRAFT_SUMMARY: board PNG + SVG with terminal-status signals; black checkmated.
If you can get the answer, please reply with ANSWER: <your answer> and ends with TERMINATE. Otherwise, please generate the next REFLECTION, THOUGHT and ACTION.
REFLECTION 1: The signals show is_checkmate=True for the black side to move — the game is already over and White wins.
THOUGHT 1: Black is to move, is in check, and has zero legal moves → checkmate. White wins.
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
If you can get the answer, please reply with ANSWER: <your answer> and ends with TERMINATE. Otherwise, please generate the next THOUGHT and ACTION.
THOUGHT 1: Based on the visual comparison of the two graphs, I can see that the sequence of degrees in both graphs are similar. Therefore, the two graphs are isomorphic.
ACTION 1: No action needed.
ANSWER: The two graphs are isomorphic. TERMINATE

# EXAMPLE: Solving a math parity problem
# USER REQUEST #: You are given a real-valued, scalar function f (x). YOUR TASK is to determine whether f(x) is an even function, an odd function, or neither.\nDefinition of an odd function: A function such that\nf(−x) = −f(x)\nwhere the sign is reversed but the absolute value remains the same if the sign of the independent variable is reversed.\nA function is neither even nor odd if it does not satisfy either condition. Here is the expression of f(x): f(x) = x*(6.0 - 16.0*x**2)/(8.0*x**10 + 6.0*x**4 + 4.28)\nRespond with 'even', 'odd', 'neither' first on whether the function f(x) is even, odd, or neither, based on the definitions and your observation of the function.
# RESULT #:
THOUGHT 0: I will plot the function f(x) = x*(6.0 - 16.0*x**2)/(8.0*x**10 + 6.0*x**4 + 4.28) to visualize the function.
ACTION 0:
```python
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Define the function
def f(x):
    return x * (6.0 - 16.0 * x**2) / (8.0 * x**10 + 6.0 * x**4 + 4.28)

# Generate x values
x = np.linspace(-10, 10, 400)
# Calculate y values
y = f(x)

# Plot the function
plt.plot(x, y)
plt.title('Plot of f(x) = x*(6.0 - 16.0*x**2)/(8.0*x**10 + 6.0*x**4 + 4.28)')

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
If you can get the answer, please reply with ANSWER: <your answer> and ends with TERMINATE. Otherwise, please generate the next THOUGHT and ACTION.
THOUGHT 1: For the graph shown in the image, the function is symmetric with respect to the origin. Therefore, the function is odd.
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
        prompt += "Now please generate only THOUGHT 0 and ACTION 0 in RESULT. If no action needed, also reply with ANSWER: <your answer> and ends with TERMINATE in the RESULT:\n# RESULT #:\n"
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
                "ANSWER: <your answer> and ends with TERMINATE."
            )
            return prompt
    


############################################################################################################
##### Prompt Generator for Geometry tasks using ReACT agent
############################################################################################################ 



class GeoPrompt:
    def __init__(self) -> None:
        self._choices = []
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
3. ABSOLUTE RULE — matplotlib coordinates are for DRAWING ONLY, NEVER for computing lengths:
   The `points = {...}` dictionary contains DISPLAY-ONLY pixel coordinates for rendering the schematic. They are NOT actual geometric distances.
   FORBIDDEN (always produces wrong answers):
     np.linalg.norm(np.array(points["A"]) - np.array(points["B"]))  # pixel distance ≠ geometric length
     abs(points["A"][1] - points["D"][1])                           # pixel y-diff ≠ geometric length
   CORRECT: derive ALL lengths and angles from the symbolic logic form using math/trigonometry only:
     import math; AC = BC * math.sin(math.radians(angle_B))  # symbolic formula from given angle + side
4. If the symbolic givens from the diagram logic form disagree with the apparent plotted lengths, trust the symbolic givens and treat the drawing as not to scale.
5. For trapezoids, circles, similar triangles, angle-chasing, and other standard Euclidean geometry problems, solve from the stated geometric relations first; use the drawing only to decide what auxiliary line or configuration to inspect.
6. MANDATORY SANITY CHECK: before finalizing, verify your computed value matches one of the given choices. If it does not match any choice, you used pixel coordinates instead of geometric formulas — discard and resolve using only symbolic givens.
7. ALWAYS EXECUTE COMPUTATION: even if no auxiliary lines are needed, you MUST write a Python code block that numerically computes the answer using `import math` and the symbolic givens. NEVER skip directly to ANSWER without running a code block first. The execution result must confirm which choice matches your calculation.
8. If you think you got the answer, use ANSWER: X to provide the answer (where X is the choice letter A, B, C, or D), and end with TERMINATE. For multiple-choice problems output ONLY the letter — do NOT write a numeric value or expression as the answer.

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
If you can get the answer, please reply with ANSWER: <your answer> and ends with TERMINATE. Otherwise, please generate the next THOUGHT and ACTION.
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

        self._choices = ex.get("choices") or []
        choices_str = ""
        if self._choices:
            labeled = "\n".join(f"  ({chr(65+i)}) {c}" for i, c in enumerate(self._choices))
            choices_str = (
                f"\nMultiple-choice options (you MUST select one):\n{labeled}\n"
                "ANSWER FORMAT: write ANSWER: X where X is exactly one letter A, B, C, or D. "
                "Do NOT write a numeric value or expression — the letter only."
            )
        prompt += f"USER REQUEST #: Given the geometry diagram <img src='{image_path_code}'> and the diagram logic form {diagram_logic_form}" + \
        f"Below is the original matplotlib code of the geometry: {code}\nYou must draw auxiliary lines to solve the following question: [{question}]{choices_str}\n" + \
        "In THOUGHT 0, first list the relevant symbolic givens from the diagram logic form. " + \
        "REMINDER: the `points` dict in the code above contains pixel display coordinates — do NOT compute distances from them. " + \
        "Solve using only the symbolic givens (lengths, angles, theorems). " + \
        "Propose matplotlib code to draw the auxiliary lines. Make sure to label the beginning and end point of the auxiliary line."
        answer_fmt = " Your ANSWER must be a single letter (A/B/C/D)." if self._choices else ""
        prompt += (
            f"Now please generate only THOUGHT 0 and ACTION 0 in RESULT. "
            "ACTION 0 MUST be a Python code block that numerically computes your answer using `import math` — "
            "do NOT skip to ANSWER without executing a code block first."
            f"{answer_fmt}\n# RESULT #:\n"
        )
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
                "ANSWER: <your answer> and ends with TERMINATE."
            )
            if self._choices:
                letters = "/".join(chr(65+i) for i in range(len(self._choices)))
                prompt += (
                    f" This is a multiple-choice question — your ANSWER must be exactly one letter ({letters}). "
                    "Do NOT write a numeric value or expression."
                )
            return prompt

# a special prompt generator to generate codes that read all image files
# needed for executor initialization
def python_codes_for_images_reading(image_paths):
    
    code = ""
    
    for idx, path in enumerate(image_paths):
        code += f"""image_{idx+1} = Image.open("{path}").convert("RGB")\n"""
        
    return code
