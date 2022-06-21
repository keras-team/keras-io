"""Lightweight fork of Keras-Autodocs.
"""
import warnings
from sphinx.util.inspect import Signature
import black
import re
import os
import inspect
import importlib
import shutil
import pathlib
from typing import Dict, Union
import itertools


class TFKerasDocumentationGenerator:
    def __init__(self, project_url=None):
        self.project_url = project_url

    def process_docstring(self, docstring):
        docstring = docstring.replace("Args:", "# Arguments")
        docstring = docstring.replace("Arguments:", "# Arguments")
        docstring = docstring.replace("Attributes:", "# Attributes")
        docstring = docstring.replace("Returns:", "# Returns")
        docstring = docstring.replace("Raises:", "# Raises")
        docstring = docstring.replace("Input shape:", "# Input shape")
        docstring = docstring.replace("Output shape:", "# Output shape")
        docstring = docstring.replace("Call arguments:", "# Call arguments")
        docstring = docstring.replace("Returns:", "# Returns")
        docstring = docstring.replace("Example:", "# Example\n")
        docstring = docstring.replace("Examples:", "# Examples\n")

        docstring = re.sub(r"\nReference:\n\s*", "\n**Reference**\n\n", docstring)
        docstring = re.sub(r"\nReferences:\n\s*", "\n**References**\n\n", docstring)

        # Fix typo
        docstring = docstring.replace("\n >>> ", "\n>>> ")

        lines = docstring.split("\n")
        doctest_lines = []
        usable_lines = []

        def flush_docstest(usable_lines, doctest_lines):
            usable_lines.append("```python")
            usable_lines += doctest_lines
            usable_lines.append("```")
            usable_lines.append("")

        for line in lines:
            if doctest_lines:
                if not line or set(line) == {" "}:
                    flush_docstest(usable_lines, doctest_lines)
                    doctest_lines = []
                else:
                    doctest_lines.append(line)
            else:
                if line.startswith(">>>"):
                    doctest_lines.append(line)
                else:
                    usable_lines.append(line)
        if doctest_lines:
            flush_docstest(usable_lines, doctest_lines)
        docstring = "\n".join(usable_lines)
        return process_docstring(docstring)

    def process_signature(self, signature):
        signature = signature.replace("tensorflow.keras", "tf.keras")
        signature = signature.replace("*args, **kwargs", "")
        return signature

    def render(self, element):
        if isinstance(element, str):
            object_ = import_object(element)
            if ismethod(object_):
                # we remove the modules when displaying the methods
                signature_override = ".".join(element.split(".")[-2:])
            else:
                signature_override = element
        else:
            signature_override = None
            object_ = element
        return self.render_from_object(object_, signature_override)

    def render_from_object(self, object_, signature_override: str):
        subblocks = []
        source_link = make_source_link(object_, self.project_url)
        if source_link is not None:
            subblocks.append(source_link)
        signature = get_signature(object_, signature_override)
        signature = self.process_signature(signature)
        subblocks.append(f"### `{get_name(object_)}` {get_type(object_)}\n")
        subblocks.append(code_snippet(signature))

        docstring = inspect.getdoc(object_)
        if docstring:
            docstring = self.process_docstring(docstring)
            subblocks.append(docstring)
        return "\n\n".join(subblocks) + "\n\n----\n\n"


def ismethod(function):
    return get_class_from_method(function) is not None


def import_object(string: str):
    """Import an object from a string.

    The object can be a function, class or method.
    For example: `'keras.layers.Dense.get_weights'` is valid.
    """
    last_object_got = None
    seen_names = []
    for name in string.split("."):
        seen_names.append(name)
        try:
            last_object_got = importlib.import_module(".".join(seen_names))
        except ModuleNotFoundError:
            assert last_object_got is not None, f"Failed to import path {string}"
            last_object_got = getattr(last_object_got, name)
    return last_object_got


def make_source_link(cls, project_url):
    if not hasattr(cls, "__module__"):
        return None
    if not project_url:
        return None

    base_module = cls.__module__.split(".")[0]
    project_url = project_url[base_module]
    assert project_url.endswith("/"), f"{base_module} not found"
    project_url_version = project_url.split("/")[-2].replace("v", "")
    module_version = importlib.import_module(base_module).__version__
    if module_version != project_url_version:
        raise RuntimeError(
            f"For project {base_module}, URL {project_url} "
            f"has version number {project_url_version} which does not match the "
            f"current imported package version {module_version}"
        )
    path = cls.__module__.replace(".", "/")
    line = inspect.getsourcelines(cls)[-1]
    return (
        f'<span style="float:right;">'
        f"[[source]]({project_url}/{path}.py#L{line})"
        f"</span>"
    )


def code_snippet(snippet):
    return f"```python\n{snippet}\n```\n"


def get_type(object_) -> str:
    if inspect.isclass(object_):
        return "class"
    elif ismethod(object_):
        return "method"
    elif inspect.isfunction(object_):
        return "function"
    elif hasattr(object_, "fget"):
        return "property"
    else:
        raise TypeError(
            f"{object_} is detected as not a class, a method, "
            f"a property, nor a function."
        )


def get_name(object_) -> str:
    if hasattr(object_, "fget"):
        return object_.fget.__name__
    return object_.__name__


def get_signature_start(function):
    """For the Dense layer, it should return the string 'keras.layers.Dense'"""
    if ismethod(function):
        prefix = f"{get_class_from_method(function).__name__}."
    else:
        try:
            prefix = f"{function.__module__}."
        except AttributeError:
            warnings.warn(
                f"function {function} has no module. "
                f"It will not be included in the signature."
            )
            prefix = ""
    return f"{prefix}{function.__name__}"


def get_signature_end(function):
    signature_end = Signature(function).format_args()
    if ismethod(function):
        signature_end = signature_end.replace("(self, ", "(")
        signature_end = signature_end.replace("(self)", "()")
        # work around case-specific bug
        signature_end = signature_end.replace(
            "synchronization=<VariableSynchronization.AUTO: 0>, aggregation=<VariableAggregationV2.NONE: 0>",
            "synchronization=tf.VariableSynchronization.AUTO, aggregation=tf.VariableSynchronization.NONE",
        )
    return signature_end


def get_function_signature(function, override=None):
    if override is None:
        signature_start = get_signature_start(function)
    else:
        signature_start = override
    signature_end = get_signature_end(function)
    return format_signature(signature_start, signature_end)


def get_class_signature(cls, override=None):
    if override is None:
        signature_start = f"{cls.__module__}.{cls.__name__}"
    else:
        signature_start = override
    signature_end = get_signature_end(cls.__init__)
    return format_signature(signature_start, signature_end)


def get_signature(object_, override):
    if inspect.isclass(object_):
        return get_class_signature(object_, override)
    elif inspect.isfunction(object_) or inspect.ismethod(object_):
        return get_function_signature(object_, override)
    elif hasattr(object_, "fget"):
        # properties
        if override:
            return override
        return get_function_signature(object_.fget)
    raise ValueError(f"Not able to retrieve signature for object {object_}")


def format_signature(signature_start: str, signature_end: str):
    """pretty formatting to avoid long signatures on one single line"""
    # first, we make it look like a real function declaration.
    fake_signature_start = "x" * len(signature_start)
    fake_signature = fake_signature_start + signature_end
    fake_python_code = f"def {fake_signature}:\n    pass\n"
    # we format with black
    mode = black.FileMode(line_length=90)
    formatted_fake_python_code = black.format_str(fake_python_code, mode=mode)
    # we make the final, multiline signature
    new_signature_end = extract_signature_end(formatted_fake_python_code)
    return signature_start + new_signature_end


def extract_signature_end(function_definition):
    start = function_definition.find("(")
    stop = function_definition.rfind(")")
    return function_definition[start : stop + 1]


def get_code_blocks(docstring):
    code_blocks = {}
    tmp = docstring[:]
    while "```" in tmp:
        tmp = tmp[tmp.find("```") :]
        index = tmp[3:].find("```") + 6
        snippet = tmp[:index]
        # Place marker in docstring for later reinjection.
        token = f"$KERAS_AUTODOC_CODE_BLOCK_{len(code_blocks)}"
        docstring = docstring.replace(snippet, token)
        code_blocks[token] = snippet
        tmp = tmp[index:]
    return code_blocks, docstring


def get_section_end(docstring, section_start):
    regex_indented_sections_end = re.compile(r"\S\n+(\S|$)")
    end = re.search(regex_indented_sections_end, docstring[section_start:])
    section_end = section_start + end.end()
    if section_end == len(docstring):
        return section_end
    else:
        return section_end - 2


def get_google_style_sections_without_code(docstring):
    regex_indented_sections_start = re.compile(r"\n# .+?\n")
    google_style_sections = {}
    for i in itertools.count():
        match = re.search(regex_indented_sections_start, docstring)
        if match is None:
            break
        section_start = match.start() + 1
        section_end = get_section_end(docstring, section_start)
        google_style_section = docstring[section_start:section_end]
        token = f"KERAS_AUTODOC_GOOGLE_STYLE_SECTION_{i}"
        google_style_sections[token] = google_style_section
        docstring = insert_in_string(docstring, token, section_start, section_end)
    return google_style_sections, docstring


def get_google_style_sections(docstring):
    # First, extract code blocks and process them.
    # The parsing is easier if the #, : and other symbols aren't there.
    code_blocks, docstring = get_code_blocks(docstring)
    google_style_sections, docstring = get_google_style_sections_without_code(docstring)
    docstring = reinject_strings(docstring, code_blocks)
    for section_token, section in google_style_sections.items():
        google_style_sections[section_token] = reinject_strings(section, code_blocks)
    return google_style_sections, docstring


def to_markdown(google_style_section: str) -> str:
    end_first_line = google_style_section.find("\n")
    section_title = google_style_section[2:end_first_line]
    section_body = google_style_section[end_first_line:]
    section_body = remove_indentation(section_body)
    if section_title in (
        "Arguments",
        "Attributes",
        "Raises",
        "Call arguments",
        "Returns",
    ):
        section_body = format_as_markdown_list(section_body)
    if section_body:
        return f"__{section_title}__\n\n{section_body}\n"
    else:
        return f"__{section_title}__\n"


def format_as_markdown_list(section_body):
    section_body = re.sub(r"\n([^ ].*?):", r"\n- __\1__:", section_body)
    section_body = re.sub(r"^([^ ].*?):", r"- __\1__:", section_body)
    return section_body


def reinject_strings(target, strings_to_inject):
    for token, string_to_inject in strings_to_inject.items():
        target = target.replace(token, string_to_inject)
    return target


def process_docstring(docstring):
    if docstring[-1] != "\n":
        docstring += "\n"
    google_style_sections, docstring = get_google_style_sections(docstring)
    for token, google_style_section in google_style_sections.items():
        markdown_section = to_markdown(google_style_section)
        docstring = docstring.replace(token, markdown_section)
    return docstring


def get_class_from_method(meth):
    if inspect.ismethod(meth):
        for cls in inspect.getmro(meth.__self__.__class__):
            if cls.__dict__.get(meth.__name__) is meth:
                return cls
        meth = meth.__func__  # fallback to __qualname__ parsing
    if inspect.isfunction(meth):
        cls = getattr(
            inspect.getmodule(meth),
            meth.__qualname__.split(".<locals>", 1)[0].rsplit(".", 1)[0],
        )
        if isinstance(cls, type):
            return cls
    return getattr(meth, "__objclass__", None)  # handle special descriptor objects


def insert_in_string(target, string_to_insert, start, end):
    target_start_cut = target[:start]
    target_end_cut = target[end:]
    return target_start_cut + string_to_insert + target_end_cut


def remove_indentation(string):
    lines = string.split("\n")
    leading_spaces = [count_leading_spaces(l) for l in lines if l]
    if leading_spaces:
        min_leading_spaces = min(leading_spaces)
        string = "\n".join(l[min_leading_spaces:] for l in lines)
    return string.strip()  # Drop leading/closing empty lines


def count_leading_spaces(s):
    ws = re.search(r"\S", s)
    if ws:
        return ws.start()
    return 0
