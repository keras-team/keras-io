import re
import string
import markdown
import copy
import pathlib
import os


def save_file(path, content):
    parent = pathlib.Path(path).parent
    if not os.path.exists(parent):
        os.makedirs(parent)
    f = open(path, "w", encoding="utf8")
    f.write(content)
    f.close()


def process_outline_title(title):
    title = re.sub(r"`(.*?)`", r"<code>\1</code>", title)
    title = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", title)
    return title


def turn_title_into_id(title):
    title = title.lower()
    title = title.replace("&amp", "amp")
    title = title.replace("&", "amp")
    title = title.replace("<code>", "")
    title = title.replace("</code>", "")
    title = title.translate(str.maketrans("", "", string.punctuation))
    title = title.replace(" ", "-")
    return title


def make_outline(md_source):
    lines = md_source.split("\n")
    outline = []
    in_code_block = False
    for line in lines:
        if line.startswith("```"):
            in_code_block = not in_code_block
        if in_code_block:
            continue
        if line.startswith("# "):
            title = line[2:]
            title = process_outline_title(title)
            outline.append(
                {
                    "title": title,
                    "url": "#" + turn_title_into_id(title),
                    "depth": 1,
                }
            )
        if line.startswith("## "):
            title = line[3:]
            title = process_outline_title(title)
            outline.append(
                {
                    "title": title,
                    "url": "#" + turn_title_into_id(title),
                    "depth": 2,
                }
            )
        if line.startswith("### "):
            title = line[4:]
            title = process_outline_title(title)
            outline.append(
                {
                    "title": title,
                    "url": "#" + turn_title_into_id(title),
                    "depth": 3,
                }
            )
    return outline


def render_markdown_to_html(md_content):
    return markdown.markdown(
        md_content,
        extensions=[
            "fenced_code",
            "tables",
            "codehilite",
            "mdx_truly_sane_lists",
            "smarty",
        ],
        extension_configs={
            "codehilite": {
                "guess_lang": False,
            },
            "smarty": {
                "smart_dashes": True,
                "smart_quotes": False,
                "smart_angled_quotes": False,
                "smart_ellipses": False,
            },
        },
    )


def set_active_flag_in_nav_entry(entry, relative_url):
    entry = copy.copy(entry)
    if relative_url.startswith(entry["relative_url"]):
        entry["active"] = True
    else:
        entry["active"] = False
    children = [
        set_active_flag_in_nav_entry(child, relative_url)
        for child in entry.get("children", [])
    ]
    entry["children"] = children
    return entry
