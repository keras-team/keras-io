"""
Title: Multimodal and Agentic Workflows with Gemma 4 in KerasHub
Author: [Sachin Prasad](https://github.com/sachinprasadhs)
Date created: 2026/04/14
Last modified: 2026/04/20
Description: A comprehensive guide to multimodal and agentic workflows with Gemma 4 in KerasHub.
Accelerator: GPU
"""

"""
## Overview

Gemma 4 is a family of multimodal open models built for long-context reasoning,
coding, and agentic workflows. In KerasHub, Gemma 4 presets are exposed through
the standard `from_preset()` API, making it straightforward to load the model
family for text, image, video, and audio-aware workflows from a single
interface.

Gemma 4 introduces key capability and architectural advancements:

- Reasoning: All models in the family are designed as highly capable reasoners,
  with configurable thinking modes.
- Extended multimodalities: All Gemma 4 models process text and images with
  variable aspect ratio and resolution support, and the full family supports
  video-style frame understanding. Audio is available on the E2B and E4B models.
- Diverse and efficient architectures: Offers dense and Mixture-of-Experts
  variants across multiple sizes for scalable deployment.
- Optimized for on-device use: Smaller models are designed for efficient local
  execution on laptops and mobile devices.
- Increased context window: The small models feature a 128K context window,
  while the medium models support 256K.
- Enhanced coding and agentic capabilities: Strong coding performance is paired
  with native function-calling support for autonomous workflows.
- Native system prompt support: Gemma 4 supports the system role directly,
  enabling more structured and controllable conversations.

This guide focuses on practical usage examples in KerasHub. It covers text
generation, image captioning, object detection-style JSON parsing with box
overlays, audio transcription, function calling, coding, thinking and reasoning
mode, multi-image comparison, video-oriented prompting, OCR, and speech
translation.
"""

"""
## Prompt Structure and Special Tokens

Gemma 4 introduces a new approach to prompt formatting, relying on native
control tokens baked into its vocabulary rather than instruction-based text
delimiters. This ensures more reliable structured outputs and conversation
management.

Gemma 4 introduces several special tokens for dialogue and modalities:

- Dialogue delimiters: Uses `<|turn>system
`, `<|turn>user
`, and `<|turn>model
` to mark turns, and `<turn|>` to end a turn (also acting as the EOS token).
- Multimodal tokens: Supports `<|image|>` for images, `<|audio|>` for audio, and
  `<|video|>` for video content placement.
- Thinking mode: The `<|think|>` token enables the model to output its internal
  reasoning before the final answer.
- Tool calling: Uses `<|tool>`, `<tool|>`, `<|tool_call>`, `<tool_call|>`,
  `<|tool_response>`, and `<tool_response|>` for native function calling.
- Delimiter token: Uses `<|"|>` within structured data to treat special
  characters as literal text, preventing syntax errors.

Gemma 4 introduces dedicated tokens for tool management (like `<|tool>`,
`<|tool_call>`, and `<|tool_response>`). This simplifies the prompt
significantly because the model natively understands these boundaries without us
having to explain them in the prompt instructions.

This guide demonstrates how to use these tokens effectively across different
modalities and workflows.
"""

"""
## Setup

Install the latest Keras and KerasHub packages first.

To load Gemma 4 from Kaggle, make sure your environment exposes
`KAGGLE_USERNAME` and `KAGGLE_KEY`, and that your account has accepted the Gemma
4 model license.

Keras lets you choose the backend that fits your environment. This notebook
defaults to JAX, but you can switch to TensorFlow or PyTorch by setting
`KERAS_BACKEND` before importing Keras.

**Note**: Since this guide exercises the full range of Gemma 4's multimodal
capabilities (including high-resolution images and video frames), it requires
significant GPU memory. For a smooth experience and to avoid Out-Of-Memory (OOM)
errors, running on a high-end GPU/TPU with large memory (such as an NVIDIA H100
or similar) is strongly recommended.

"""

"""shell
pip install -q -U keras keras-hub
pip install -q -U soundfile scipy requests pillow matplotlib av
# Optional: Upgrade CUDA and JAX if you encounter ptxas errors
# pip install --upgrade nvidia-cuda-nvcc-cu12
# pip install -q --upgrade "jax[cuda12]"
"""

import os

os.environ["KERAS_BACKEND"] = "jax"  # or "tensorflow" or "torch"

import json
import re
from io import BytesIO

import keras
import keras_hub
import matplotlib.pyplot as plt
import numpy as np
import requests
import soundfile as sf
from PIL import Image
import av

from IPython.display import HTML, Markdown, display

AUDIO_URL = (
    "https://raw.githubusercontent.com/keras-team/keras-hub/master/"
    "keras_hub/src/tests/test_data/audio_transcription_tests/"
    "female_short_voice_clip_17sec.wav"
)
IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"

"""
## Helper utilities

The next cell keeps the rest of the notebook readable. In addition to loading
images and audio, it includes a parser for Gemma 4 detection-style responses.

Gemma 4 may emit a short natural-language preamble before the JSON payload. The
helper extracts the JSON block, rescales the 0-1000 coordinates back to pixel
space, and renders boxes with `keras.visualization.draw_bounding_boxes()`.
"""


def load_image(*sources):
    """Load one or more images from URLs or local file paths."""

    def _load_one(source):
        if source.startswith(("http://", "https://")):
            response = requests.get(source, timeout=30)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        return Image.open(source).convert("RGB")

    if not sources:
        return _load_one(IMAGE_URL)
    images = [_load_one(source) for source in sources]
    return images[0] if len(images) == 1 else images


def display_images(images, titles=None, figsize=(8, 6)):
    if not isinstance(images, (list, tuple)):
        images = [images]
    titles = titles or [None] * len(images)
    plt.figure(figsize=(figsize[0] * len(images), figsize[1]))
    for index, (image, title) in enumerate(zip(images, titles), start=1):
        plt.subplot(1, len(images), index)
        plt.imshow(image)
        plt.axis("off")
        if title:
            plt.title(title)
    plt.show()


def load_audio(path):
    raw, sample_rate = sf.read(path)
    if raw.ndim > 1:
        raw = raw.mean(axis=1)
    if sample_rate != 16000:
        from scipy import signal

        raw = signal.resample(raw, int(len(raw) * 16000 / sample_rate))
    return raw.astype(np.float32)


def strip_prompt(output, prompt):
    if isinstance(output, list):
        output = output[0]
    if output.startswith(prompt):
        return output[len(prompt) :]
    for marker in ("<start_of_turn>model\n", "<|turn>model\n"):
        index = output.rfind(marker)
        if index != -1:
            return output[index + len(marker) :]
    return output


def extract_json_block(text):
    fenced_match = re.search(r"```json\s*(\[.*?\])\s*```", text, flags=re.DOTALL)
    if fenced_match:
        return json.loads(fenced_match.group(1))

    bracket_match = re.search(r"(\[\s*\{.*\}\s*\])", text, flags=re.DOTALL)
    if bracket_match:
        return json.loads(bracket_match.group(1))

    raise ValueError("Could not find a JSON detection block in the model output.")


def gemma_boxes_to_keras_prediction(image, detections):
    width, height = image.size
    label_to_id = {}
    boxes = []
    labels = []

    for detection in detections:
        label = detection.get("label", "object")
        if label not in label_to_id:
            label_to_id[label] = len(label_to_id)

        y_min, x_min, y_max, x_max = detection["box_2d"]
        boxes.append(
            [
                y_min / 1000.0 * height,
                x_min / 1000.0 * width,
                y_max / 1000.0 * height,
                x_max / 1000.0 * width,
            ]
        )
        labels.append(label_to_id[label])

    prediction = {
        "boxes": np.array([boxes], dtype="float32"),
        "labels": np.array([labels], dtype="int32"),
    }
    class_mapping = {value: key for key, value in label_to_id.items()}
    return prediction, class_mapping


def render_detection_result(image, raw_output):
    detections = extract_json_block(raw_output)
    prediction, class_mapping = gemma_boxes_to_keras_prediction(image, detections)
    image_batch = np.expand_dims(np.array(image), axis=0)

    rendered = keras.visualization.draw_bounding_boxes(
        image_batch,
        prediction,
        bounding_box_format="yxyx",
        class_mapping=class_mapping,
        color=(255, 235, 59),
        line_thickness=3,
        text_thickness=2,
        font_scale=0.8,
    )

    plt.figure(figsize=(8, 8))
    plt.imshow(rendered[0].astype("uint8"))
    plt.axis("off")
    plt.show()
    return detections


"""
## Load the Gemma 4 preset

This guide uses `gemma4_instruct_2b` (2.3B effective, 5.1B with embeddings)
from Kaggle. It is a practical starting point because it supports text, image,
video-style prompting, and audio-enabled workflows in a relatively small
checkpoint.
"""

print("Loading preset: gemma4_instruct_2b")
model = keras_hub.models.Gemma4CausalLM.from_preset(
    "gemma4_instruct_2b",
    dtype="bfloat16",
)


"""
## 1. Text generation

Start with a simple usage example. This confirms that the preset, tokenizer, and
prompt formatting are all working before moving into multimodal prompts.
"""

PROMPT_TEXT = (
    "<|turn>user\n"
    "Write a short, creative pitch for a movie about a time-traveling\n"
    "historian who accidentally changes the recipe for pizza.<turn|>\n"
    "<|turn>model\n"
)

text_output = model.generate({"prompts": [PROMPT_TEXT]}, max_length=512)
print("\n" + "=" * 50 + "\nModel Output:\n" + "=" * 50)
display(Markdown(strip_prompt(text_output, PROMPT_TEXT)))

"""
## 2. Image captioning

Gemma 4 supports image understanding directly through
`Gemma4CausalLM.generate()`. A simple captioning prompt is a good way to verify
multimodal input handling before moving on to more structured visual tasks.
"""

PROMPT_IMAGE = (
    "<|turn>user\n"
    "<|image|>\n"
    "Describe what you see in this image<turn|>\n"
    "<|turn>model\n"
)
image = load_image()
display_images(image, titles=["Input image"])

image_output = model.generate(
    {"prompts": [PROMPT_IMAGE], "images": [image]},
    max_length=2048,
)
print("\n" + "=" * 50 + "\nModel Output:\n" + "=" * 50)
display(Markdown(strip_prompt(image_output, PROMPT_IMAGE)))

"""
## 3. Object detection-style localization

Gemma 4 can localize objects by returning structured JSON alongside natural-
language output. In practice, the model often produces a short explanation
before the JSON payload, so it is useful to parse the fenced block explicitly.

In this section, the model identifies cats in the image, the JSON is extracted,
the normalized 0-1000 coordinates are rescaled to image pixels, and the boxes
are rendered with Keras visualization utilities.
"""

PROMPT_DETECTION = (
    "<|turn>user\n"
    "<|image|>\n"
    "Identify every cat in this image. First give one short sentence, then\n"
    "return a JSON array inside a ```json``` block. Each item must contain\n"
    "<|turn>model\n"
)
detection_output = model.generate(
    {"prompts": [PROMPT_DETECTION], "images": [image]},
    max_length=2048,
)
raw_detection_text = strip_prompt(detection_output, PROMPT_DETECTION)
print("\n" + "=" * 50 + "\nModel Output:\n" + "=" * 50)
display(Markdown(raw_detection_text))

parsed_detections = render_detection_result(image, raw_detection_text)
parsed_detections

"""
## 4. Audio transcription

Audio input is available on the E2B and E4B Gemma 4 checkpoints. Video-style
understanding is supported across the model family, but audio is limited to
these smaller variants. The prompt below makes the task explicit and specifies
the exact output formatting.

**Note**: Gemma 4 expects audio input to be sampled at 16kHz. In this guide, we
use a helper function `load_audio` (defined in the utilities section) to read
the audio file, downmix it to mono if necessary, and resample it to the desired
16kHz rate.

"""

PROMPT_AUDIO = (
    "<|turn>user\n"
    "<|audio|>"
    "Transcribe the following speech segment in its original language. "
    "Follow these specific instructions for formatting the answer:\n"
    "* Only output the transcription, with no newlines.\n"
    "* Only output the transcription, with no newlines.\n"
    "* When transcribing numbers, write the digits, i.e. write 1.7\n"
    " and not one point seven, and write 3 instead of three.<turn|>\n"
    "<|turn>model\n"
)

audio_path = "female_short_voice_clip_17sec.wav"
if not os.path.exists(audio_path):
    print("Downloading audio file...")
    response = requests.get(AUDIO_URL)
    with open(audio_path, "wb") as f:
        f.write(response.content)

audio = load_audio(audio_path)
print("Audio shape:", audio.shape, "dtype:", audio.dtype)
audio_output = model.generate(
    {"prompts": [PROMPT_AUDIO], "audio": [audio]},
    max_length=2048,
)
print("\n" + "=" * 50 + "\nModel Output:\n" + "=" * 50)
display(Markdown(strip_prompt(audio_output, PROMPT_AUDIO)))

"""
## 5. Function calling

Gemma 4 supports native function calling for tool-augmented workflows. The basic
pattern is consistent across applications: declare the tool, let the model emit
a tool call, execute the tool externally, append the tool response, and then
continue generation.
"""

PROMPT_FUNC_CALL = (
    "<|turn>system\n"
    "You are a helpful assistant."
    "<|tool>declaration:get_current_weather{"
    'description:<|"|>Returns the current weather for a given city.,'
    "parameters:{"
    'location:{type:<|"|>string<|"|>,description:<|"|>The city name<|"|>}'
    "}}"
    "<tool|><turn|>\n"
    "<|turn>user\n"
    "What is the weather like in Paris right now?<turn|>\n"
    "<|turn>model\n"
)

tool_call_output = model.generate({"prompts": [PROMPT_FUNC_CALL]}, max_length=256)
tool_call_text = (
    tool_call_text[0] if isinstance(tool_call_text, list) else tool_call_text
)
tool_call_text = tool_call_text.split("<|turn>model\n")[-1]
print(tool_call_text)

PROMPT_WITH_RESPONSE = (
    PROMPT_FUNC_CALL
    + '<|tool_call>call:get_current_weather{location:<|\\"|>Paris<|\\"|>}'
    + "<tool_call|><|tool_response>response:get_current_weather{"
    + 'temperature:18,weather:<|\\"|>partly cloudy<|\\"|>}<tool_response|>\n'
    + "<|turn>model\n"
)

final_weather_output = model.generate(
    {"prompts": [PROMPT_WITH_RESPONSE]}, max_length=256
)
final_weather_text = final_weather_text.split("<|turn>model\n")[-1]
print(final_weather_text)

"""
## 6. Coding

Gemma 4 is also a strong coding model. You can use the same causal language
model interface for code generation, refactoring, explanation, and small utility
synthesis tasks.
"""

PROMPT_CODE = (
    "<|turn>user\n"
    "Write a Python function named `is_palindrome` that ignores\n"
    "punctuation, whitespace, and letter casing. Also include a few\n"
    "short test calls.<turn|>\n"
    "<|turn>model\n"
)

code_output = model.generate({"prompts": [PROMPT_CODE]}, max_length=512)
print("\n" + "=" * 50 + "\nModel Output:\n" + "=" * 50)
display(Markdown(strip_prompt(code_output, PROMPT_CODE)))

"""
## 7. HTML generation and rendering

Gemma 4 can generate structured code like HTML. In a notebook environment like
Colab, we can use Python's `IPython.display.HTML` to render the generated HTML
directly, providing a visual verification of the model's output.

"""

PROMPT_HTML = (
    "<|turn>user\n"
    "Create a stunning, interactive Glassmorphic product card in HTML and\n"
    "inline CSS. It should have a dark background for the page, a\n"
    "frosted-glass effect on the card (backdrop-filter: blur), a glowing\n"
    "and a glowing 'Buy Now' button. "
    "Make it look like a premium UI component. "
    "Return ONLY the HTML code inside a ```html block.<turn|>\n"
    "<|turn>model\n"
)

html_output = model.generate({"prompts": [PROMPT_HTML]}, max_length=2048)
html_text = strip_prompt(html_output, PROMPT_HTML)

# Extract HTML from the code block
match = re.search(r"```html\s*(.*?)\s*```", html_text, flags=re.DOTALL)
if match:
    extracted_html = match.group(1)
    print("\nRendering generated HTML:")
    display(HTML(extracted_html))
    print("\n" + "=" * 50 + "\nModel Output:\n" + "=" * 50)
    display(Markdown(html_text))
else:
    print("\nCould not find HTML block to render.")

"""
## 8. Thinking mode and Agentic Workflows

Thinking mode is controlled from the system prompt. Gemma 4 can emit a thought
channel before a tool call or before the final answer.

In this section, we demonstrate a small automated agentic workflow for a **Smart
Home Assistant**. We define local Python functions as tools, and write a simple
loop to automate the process: the model decides to call a tool, we execute it,
and feed the result back to the model until it completes the task.

"""


# Mock tools
def control_device(room, device, action):
    return f"Success: {device} in {room} is now {action}."


def read_sensor(room, sensor_type):
    if sensor_type == "temperature":
        return "22°C"
    elif sensor_type == "humidity":
        return "45%"
    return "Unknown sensor"


tools = {"control_device": control_device, "read_sensor": read_sensor}

PROMPT_AGENT = (
    "<|turn>system\n"
    "<|think|>You are a helpful smart home assistant."
    "<|tool>declaration:control_device{"
    'description:<|"|>Control a smart home device.<|"|>,'
    "parameters:{"
    'room:{type:<|"|>string<|"|>,description:<|"|>The room name<|"|>},'
    'device:{type:<|"|>string<|"|>,description:<|"|>The device name\n'
    '(e.g., lights, fan)<|"|>},\n'
    "}}"
    "<|tool>declaration:read_sensor{"
    'description:<|"|>Read a sensor value.<|"|>,'
    "parameters:{"
    'room:{type:<|"|>string<|"|>,description:<|"|>The room name<|"|>},'
    'room:{type:<|"|>string<|"|>,description:<|"|>The room name<|"|>},\n'
    "}}"
    "<tool|><turn|>\n"
    "<|turn>user\n"
    "Turn off the kitchen lights and check the\n"
    "temperature in the bedroom.<turn|>\n"
    "<|turn>model\n"
)


def run_agent_loop(prompt):
    conversation = prompt
    max_turns = 5

    for turn in range(max_turns):
        print(f"\n--- Agent Turn {turn + 1} ---")
        output = model.generate({"prompts": [conversation]}, max_length=1024)
        generated_text = strip_prompt(output, conversation)

        print("\n" + "=" * 50 + "\nModel Output:\n" + "=" * 50)
        display(Markdown(generated_text))

        # Check for tool call
        match = re.search(
            r"<|tool_call>(.*?)<tool_call|>", generated_text, flags=re.DOTALL
        )
        if match:
            call_str = match.group(1)
            call_match = re.search(r"call:(\w+)\{(.*?)\}", call_str)
            if call_match:
                tool_name = call_match.group(1)
                args_str = call_match.group(2)
                args = {}
                for pair in args_str.split(","):
                    join_pair = pair.split(":")
                    if len(join_pair) == 2:
                        val = join_pair[1].strip()
                        val = val.replace('<|\\"|>', "").replace("<|\\|>", "")
                        args[join_pair[0].strip()] = val

                print(f"\nExecuting tool: {tool_name} with args {args}")
                if tool_name in tools:
                    result = tools[tool_name](**args)
                    print(f"Tool Result: {result}")

                    conversation += generated_text
                    conversation += (
                        f"<|tool_response>response:{tool_name}"
                        f"{{{result}}}<tool_response|>\n"
                    )
                    conversation += "<|turn>model\n"
                else:
                    print(f"Error: Tool {tool_name} not found.")
                    break
            else:
                print("Error: Malformed tool call.")
                break
        else:
            print("\nAgent finished or no tool call.")
            break


run_agent_loop(PROMPT_AGENT)

"""
### Reasoning-only thinking example

Thinking mode is not limited to tool use. It can also be used for complex
reasoning tasks, such as solving math problems step-by-step.

"""

PROMPT_THINKING = (
    "<|turn>system\n"
    "<|think|>You are a helpful math tutor.<turn|>\n"
    "<|turn>user\n"
    "A train travels 120 km in 1.5 hours. How long\n"
    "will it take to travel 200 km at the same speed?<turn|>\n"
    "<|turn>model\n"
)

thinking_output = model.generate({"prompts": [PROMPT_THINKING]}, max_length=512)
thinking_text = (
    thinking_output[0] if isinstance(thinking_output, list) else thinking_output
)
thinking_text = thinking_text.split("<|turn>model\n")[-1]
print("\n" + "=" * 50 + "\nModel Output:\n" + "=" * 50)
display(Markdown(thinking_text))

"""
## 9. Multi-image comparison

Gemma 4 can compare multiple images within a single turn. This is useful for
comparing similar objects in different scenes or finding differences.

In this section, we use two images of zebras in different environments (lush
jungle vs. dry enclosure) and ask the model to describe the scenes and
interactions.
"""

ZEBRA_URL_1 = "http://images.cocodataset.org/val2017/000000113354.jpg"
ZEBRA_URL_2 = "http://images.cocodataset.org/val2017/000000104455.jpg"

image_1 = load_image(ZEBRA_URL_1)
image_2 = load_image(ZEBRA_URL_2)
display_images([image_1, image_2], titles=["Zebra Scene 1", "Zebra Scene 2"])

PROMPT_COMPARE = (
    "<|turn>user\n"
    "<|image|>\n"
    "<|image|>\n"
    "These images show the same type of animal in different environments. "
    "These images show the same type of animal in different environments.\n"
    "Describe the differences in the scenes and how the animals appear\n"
    "to be interacting with their surroundings.<turn|>\n"
    "<|turn>model\n"
)

comparison_output = model.generate(
    {"prompts": [PROMPT_COMPARE], "images": [[np.array(image_1), np.array(image_2)]]},
    max_length=2048,
)
print("\n" + "=" * 50 + "\nModel Output:\n" + "=" * 50)
display(Markdown(strip_prompt(comparison_output, PROMPT_COMPARE)))


"""
## 10. Video prompting

Gemma 4 supports native video understanding. In KerasHub, you can pass a video
as a sequence of frames directly to the model using the `videos` argument in the
input dictionary, and use the `<|video|>` token in the prompt to indicate where
the video content belongs.

**Note**: Video prompts involve processing many frames, which can result in very
long sequences. Make sure to increase `max_length` appropriately when generating
responses for video inputs.

"""


VIDEO_URL = (
    "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/"
    "Big_Buck_Bunny_360_10s_1MB.mp4"
)

# Download and decode video
print("Downloading test video...")
response = requests.get(VIDEO_URL, timeout=60)
response.raise_for_status()
container = av.open(BytesIO(response.content))
frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
video_frames = np.stack(frames)  # Shape: (T, H, W, C)

print(f"Decoded {video_frames.shape[0]} frames.")

T = video_frames.shape[0]
indices = np.arange(0, T, T / num_frames).astype(int)[:num_frames]
video_frames_sub = video_frames[indices]

print(f"Subsampled to {video_frames_sub.shape[0]} frames.")

PROMPT_VIDEO = "<|turn>user\n" "<|video|>Describe this video.<turn|>\n" "<|turn>model\n"

# Gemma4CausalLM expects batched inputs, so we add a batch dimension to video
video_output = model.generate(
    {"prompts": [PROMPT_VIDEO], "videos": [video_frames_sub]},
    max_length=4096,
)

print("\n" + "=" * 50 + "\nModel Output:\n" + "=" * 50)
display(Markdown(strip_prompt(video_output, PROMPT_VIDEO)))

"""
## 11. OCR, translation & entity extraction

Gemma 4 is strong on OCR and document understanding. For document-heavy
tasks, the prompting pattern stays simple: pass the image, ask for
exact extraction, and make the expected output format explicit.

In this section, we use an image of German street signs. We ask the model to
transcribe the German text (OCR), identify the location names (entities), and
provide their English translation.
"""

OCR_IMAGE_URL = "http://images.cocodataset.org/val2017/000000011615.jpg"
ocr_image = load_image(OCR_IMAGE_URL)
display_images(ocr_image, titles=["OCR input"])

PROMPT_OCR = (
    "<|turn>user\n"
    "<|image|>\n"
    "Extract the text from this image. First, transcribe all text in its\n"
    "original language (German). Then, identify the entities (place names\n"
    "like 'Severins-brücke') and provide their English translation.\n"
    "Format the output clearly.<turn|>\n"
    "<|turn>model\n"
)

ocr_output = model.generate(
    {"prompts": [PROMPT_OCR], "images": [ocr_image]},
    max_length=2048,
)
print("\n" + "=" * 50 + "\nModel Output:\n" + "=" * 50)
display(Markdown(strip_prompt(ocr_output, PROMPT_OCR)))

"""
## 12. Travel planning from location

Gemma 4's multimodal capabilities allow it to recognize famous landmarks and
provide contextual information, such as creating travel plans based on the
location shown in an image.
"""

LOCATION_URL = "http://images.cocodataset.org/val2017/000000036678.jpg"
location_image = load_image(LOCATION_URL)
display_images(location_image, titles=["Target Location"])

PROMPT_TRAVEL = (
    "<|turn>user\n"
    "<|image|>\n"
    "Identify the location in this image and create a 3-day travel plan\n"
    "for it. Include top attractions and a few restaurant suggestions.\n"
    "Format the output clearly with headings.<turn|>\n"
    "<|turn>model\n"
)

travel_output = model.generate(
    {"prompts": [PROMPT_TRAVEL], "images": [location_image]},
    max_length=2048,
)
print("\n" + "=" * 50 + "\nModel Output:\n" + "=" * 50)
display(Markdown(strip_prompt(travel_output, PROMPT_TRAVEL)))

"""
## Closing notes

This guide stays close to the low-level `Gemma4CausalLM.generate()` API on
purpose. It makes the prompt format explicit, shows how to pass images and
audio, and demonstrates how to post-process structured output for downstream
visualization.

The above examples demonstrate the high-level usage and capability of this
model. You can explore and perform tasks for many different use cases and
experiment with complex reasoning, agentic workflows, and multimodal
understanding.

**Note**: This model is trained with 140+ languages and it works well with Text
translation, speech translation, Image Translation and other such tasks.


For production usage, the main things to harden are authentication, response
validation, retry handling, and guardrails around tool execution and OCR or
detection post-processing.

### Further Reading

- For LoRA and QLoRA fine-tuning of Gemma, refer to the [Parameter-efficient
  fine-tuning of Gemma with LoRA and QLoRA](https://keras.io/examples/keras_reci
  pes/parameter_efficient_finetuning_of_gemma_with_lora_and_qlora/) example.
- For quantization, refer to the [INT8 quantization in
  Keras](https://keras.io/guides/int8_quantization_in_keras/) guide.

"""
