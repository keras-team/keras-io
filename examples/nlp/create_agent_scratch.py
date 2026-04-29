"""
Title: Building a Basic Agent from Scratch
Author: [Muhammad Anas Raza](https://linkedin.com/in/memanasraza)
Date created: 2026/04/28
Last modified: 2026/04/28
Description: A minimalist implementation of a ReAct-style agent using Keras Hub.
"""

"""
# Introduction
This example demonstrates how to construct a basic autonomous agent from scratch using a
pre-trained Large Language Model (LLM). The agent is designed to bridge the gap between
static text generation and dynamic task execution.

It performs fundamental operations such as mathematical calculations, unit conversions,
and retrieving real-time information. By utilizing a structured prompt format, the agent
identifies when to call external tools, processes their outputs, and formulates a final
response for the user.
"""

"""
## Setup and LLM
For this demonstration, we use the lightweight GPT-2 model. While small, it serves as
an excellent baseline for understanding the mechanics of tool-calling and prompt
engineering.
"""

"""shell
!pip install -q keras>=3.0 keras-hub
"""
import os
os.environ["KERAS_BACKEND"] = "jax"

import keras_hub
import datetime
import inspect
import json
import math
import re

MODEL = "hf://keras/gpt2_base_en"
llm   = keras_hub.models.CausalLM.from_preset(MODEL)
llm.compile(sampler="greedy")


"""
# Tools
Tools define the external functionalities available to our model. We have implemented
a calculator, a time utility, and a unit converter. To ensure robustness, the tools
include
aliases for common units, helping the LLM map informal user queries to specific functions.
"""

TOOLS = {}


def tool(desc):
    def decorator(fn):
        params = {
            n: p.annotation.__name__
            for n, p in inspect.signature(fn).parameters.items()
            if p.annotation != inspect.Parameter.empty
        }
        TOOLS[fn.__name__] = {"fn": fn, "desc": desc, "params": params}
        return fn

    return decorator


@tool("Evaluate a math expression. E.g. '2**10', 'math.sqrt(9)'")
def calculator(expression: str) -> str:
    ns = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    try:
        return str(round(eval(expression, {"__builtins__": {}}, ns), 8))
    except Exception as e:
        return f"Error: {e}"


@tool("Return the current UTC date and time.")
def get_time() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


@tool("Convert units: km/miles, kg/lbs, m/feet, celsius/fahrenheit.")
def convert(value: float, from_unit: str, to_unit: str) -> str:
    aliases = {
        "lb": "lbs",
        "pound": "lbs",
        "pounds": "lbs",
        "kilometer": "km",
        "kilometers": "km",
        "kilometre": "km",
        "mile": "miles",
        "mi": "miles",
        "foot": "feet",
        "ft": "feet",
        "kilogram": "kg",
        "kilograms": "kg",
        "c": "celsius",
        "f": "fahrenheit",
    }
    f = aliases.get(from_unit.lower(), from_unit.lower())
    t = aliases.get(to_unit.lower(), to_unit.lower())

    table = {
        ("km", "miles"): 0.621371,
        ("miles", "km"): 1.60934,
        ("kg", "lbs"): 2.20462,
        ("lbs", "kg"): 0.453592,
        ("m", "feet"): 3.28084,
        ("feet", "m"): 0.3048,
        ("g", "oz"): 0.035274,
        ("oz", "g"): 28.3495,
    }
    if (f, t) in table:
        return f"{value} {f} = {round(value * table[(f,t)], 4)} {t}"
    if f == "celsius" and t == "fahrenheit":
        return f"{value}C = {round(value*9/5+32, 2)}F"
    if f == "fahrenheit" and t == "celsius":
        return f"{value}F = {round((value-32)*5/9, 2)}C"
    return f"Unsupported: {from_unit} -> {to_unit}"


@tool("List all available tools.")
def list_tools() -> str:
    return "\n".join(f"{n}: {v['desc']}" for n, v in TOOLS.items())


print("Tools:", list(TOOLS))
print("Model:", MODEL)

"""
# Execution Logic
The following functions handle the interaction between the LLM's generated
instructions and the Python execution environment.
"""


def run_tool(name, args_str):
    if name not in TOOLS:
        return f"Unknown tool '{name}'"
    try:
        args = json.loads(args_str) if args_str.strip() else {}
        return str(TOOLS[name]["fn"](**args))
    except Exception as e:
        return f"Error: {e}"


"""
# Prompt Engineering
We utilize a 'few-shot' prompting strategy. By providing the LLM with examples of
Query -> Action -> Observation sequences, we teach it to follow a logical
thought process before arriving at an answer.
"""

ACTION_RE = re.compile(r"ACTION:\s*(\w+)\s*\(\s*(\{.*?\}|)\s*\)", re.DOTALL)


def make_prompt(query, ctx=""):
    tools_block = "\n".join(
        f"  {n}({', '.join(f'{p}:{t}' for p,t in v['params'].items())}) - {v['desc']}"
        for n, v in TOOLS.items()
    )

    few_shot = """Q: What time is it?
ACTION: get_time({})
OBSERVATION: 2024-01-15 10:30:00 UTC
ANSWER: The current UTC time is 2024-01-15 10:30:00.

Q: Convert 10 km to miles.
ACTION: convert({"value": 10, "from_unit": "km", "to_unit": "miles"})
OBSERVATION: 10 km = 6.2137 miles
ANSWER: 10 kilometers is 6.2137 miles.

Q: What is 5 multiplied by 12?
ACTION: calculator({"expression": "5*12"})
OBSERVATION: 60
ANSWER: 5 multiplied by 12 is 60.

Q: What is 2 to the power of 10?
ACTION: calculator({"expression": "2**10"})
OBSERVATION: 1024
ANSWER: 2 to the power of 10 is 1024.

"""
    return f"Tools:\n{tools_block}\n\n{few_shot}Q: {query}\n{ctx}"


"""
# Agent Loop
The agent loop iterates until a final answer is generated or the maximum step limit
is reached. It parses the LLM output for specific action tags and executes the
corresponding Python functions.
"""


def agent(query, max_steps=3):
    print(f"\n{'─'*50}\n👤 {query}\n{'─'*50}")
    ctx = ""
    seen_actions = set()

    for step in range(max_steps):
        prompt = make_prompt(query, ctx)
        token_estimate = min(1024, max(256, len(prompt) // 3 + 80))
        full = llm.generate(prompt, max_length=token_estimate)

        after = full[len(prompt):]

        if ctx.endswith("ANSWER:"):
            answer_part = after.split("ANSWER:")[-1].split("\n")[0].strip()
            if answer_part:
                print(f"\n✅ {answer_part}")
                return answer_part

        gen = after.split("\n")[0].strip()
        print(f"[{step+1}] {gen}")

        if gen.startswith("ANSWER:"):
            ans = gen[len("ANSWER:") :].strip()
            print(f"\n✅ {ans}")
            return ans

        m = ACTION_RE.search(gen)
        if m:
            action_key = m.group(1)
            if action_key in seen_actions:
                answer_part = after.split("ANSWER:")[-1].split("\n")[0].strip()
                print(f"\n✅ {answer_part}")
                return answer_part
            seen_actions.add(action_key)
            obs = run_tool(m.group(1), m.group(2))
            print(f"   🔧 {m.group(1)} -> {obs}")
            ctx += f"ACTION: {m.group(1)}({m.group(2)})\nOBSERVATION: {obs}\nANSWER:"
        else:
            print(f"💬 {gen}")
            return gen


"""
# Testing the Agent
Below, we test the agent's ability to reason and use tools for arithmetic
and unit conversion tasks.
"""

agent("What is 2+2?")

agent("Convert 100 miles to kms.")

agent("Convert 67.4 kgs to lbs.")

"""
This is a foundational example that may occasionally fail. In practice, performance and
reasoning can be significantly improved by using more advanced LLMs. You can explore more
[available models here](https://keras.io/keras_hub/presets/), though some may require
more advanced setup.
"""
