import json
import os
import re
from pathlib import Path

from autogen_utils import make_outline, save_file
from examples_master import EXAMPLES_INFO


def parse_examples():
    # Helper method to parse EXAMPLE_INFO into the format
    # `{category: {sub_category: example_data}}.``
    def add_key_val(key, val, dictionary):
        if key in dictionary:
            dictionary[key].append(val)
        else:
            dictionary[key] = [val]

    category_to_examples = {}
    for example_name, val in EXAMPLES_INFO.items():
        categories = val["category"]
        # Category has at most 2 levels.
        if len(categories) == 1:
            add_key_val(categories[0], example_name, category_to_examples)
        elif len(categories) == 2:
            if categories[0] in category_to_examples:
                dictionary = category_to_examples[categories[0]]
                add_key_val(categories[1], example_name, dictionary)
            else:
                category_to_examples[categories[0]] = {
                    categories[1]: [example_name],
                }
    return category_to_examples


def make_examples_nav_index(base_url):
    # Makes the navbar index for example pages.
    category_to_examples = parse_examples()
    examples_nav = []
    for parent_category, entry in category_to_examples.items():
        if isinstance(entry, list):
            child_categories = []
        else:
            child_categories = list(entry.keys())
        examples_nav.append((parent_category, child_categories))

    examples_index = []
    landing_url = base_url + "examples/"
    relative_landing_url = "/examples/"
    for nav_item in examples_nav:
        parent_category = nav_item[0]
        parent_path = parent_category.lower().replace(" ", "_")
        parent_url = landing_url + parent_path + "/"
        relative_parent_url = relative_landing_url + parent_path + "/"
        children_index = []

        for child_category in nav_item[1]:
            child_path = child_category.lower().replace(" ", "_")
            url = parent_url + child_path + "/"
            relative_url = relative_parent_url + child_path + "/"
            children_index.append(
                {
                    "title": child_category,
                    "relative_url": relative_url,
                    "url": url,
                    "children": [],
                }
            )

        examples_index.append(
            {
                "title": parent_category,
                "relative_url": parent_url,
                "url": relative_parent_url,
                "children": children_index,
            }
        )

    return {
        "title": "Code examples",
        "relative_url": landing_url,
        "url": relative_landing_url,
        "children": examples_index,
    }


def generate_example_page_url(path_stack):
    path_stack = [s.lower().replace(" ", "_") for s in path_stack]
    return "/" + str(Path(*path_stack)) + "/"


def save_example_md_files(
    base_md_source_dir,
    base_url,
    md_content,
    toc_content,
    page_title,
    path_stack,
):
    md_content = md_content.replace("{{toc}}", toc_content)
    # Lower case and replace space with underscore for `path_stack`.
    url_path_stack = [s.lower().replace(" ", "_") for s in path_stack]
    save_dir = Path(base_md_source_dir) / Path(*url_path_stack)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    location_history = []
    for i in range(len(path_stack)):
        location_history.append(
            {
                "url": generate_example_page_url(path_stack[: i + 1]),
                "title": path_stack[i],
            }
        )

    metadata = json.dumps(
        {
            "location_history": location_history,
            "outline": make_outline(md_content),
            "location": "/" + "/".join([s.replace("/", "") for s in path_stack]),
            "url": base_url + str(Path(*path_stack)) + "/",
            "title": page_title,
        }
    )

    md_file_path = save_dir / "index.md"
    metadata_file_path = save_dir / "index_metadata.json"

    save_file(md_file_path, md_content)
    save_file(metadata_file_path, metadata)


def generate_example_page(
    base_md_source_dir,
    base_url,
    page_title,
    category_to_examples,
    path_stack,
    md_content,
):
    # Generate each page, and recursively generate children pages.
    url = generate_example_page_url(path_stack)
    toc_content = f"# [{page_title}]({url})\n"

    if isinstance(category_to_examples, list):
        toc_content += generate_examples_list_md(
            category_to_examples, path_stack, is_landing_page=False
        )
        save_example_md_files(
            base_md_source_dir,
            base_url,
            md_content,
            toc_content,
            page_title,
            path_stack,
        )
        toc_content_to_return = f"# [{page_title}]({url})\n"
        toc_content_to_return += generate_examples_list_md(
            category_to_examples, path_stack, is_landing_page=True
        )
        return toc_content_to_return

    for category, entry in category_to_examples.items():
        child_toc_content = generate_example_page(
            base_md_source_dir,
            base_url,
            category,
            entry,
            path_stack + [category],
            "{{toc}}",
        )
        # Correct the title level.
        child_toc_content = re.sub(r"(#+)", r"\1#", child_toc_content)
        toc_content += child_toc_content
    save_example_md_files(
        base_md_source_dir, base_url, md_content, toc_content, page_title, path_stack
    )
    return toc_content


def generate_examples_list_md(examples_list, path_stack, is_landing_page):
    # Helper method to generate a markdown list from example list.
    md_content = ""
    for example_name in examples_list:
        example_properties = EXAMPLES_INFO[example_name]
        # On landing page, we only show selected samples.
        if is_landing_page and not example_properties["on_landing_page"]:
            continue
        example_path = example_properties["path"]
        example_url = f"{example_path}"
        example_item = f"- [{example_name}]({example_url})\n"
        md_content += example_item
    if is_landing_page:
        path_stack = [path.lower().replace(" ", "_") for path in path_stack]
        url = generate_example_page_url(path_stack)
        md_content += f"- [More examples...]({url})\n"
    return md_content


def generate_all_examples_page(base_md_source_dir, base_url):
    # Generate all example pages, from the landing page "/examples/".
    category_to_examples = parse_examples()

    # Generate the landing page and children pages.
    landing_page_file = open("../templates/examples/index.md", encoding="utf8")
    landing_page_md = landing_page_file.read()
    landing_page_file.close()
    generate_example_page(
        base_md_source_dir,
        base_url,
        "",
        category_to_examples,
        ["examples"],
        landing_page_md,
    )
