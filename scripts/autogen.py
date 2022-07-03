"""Documentation generator for Keras.io

USAGE:

python autogen.py make
python autogen.py serve

DEPENDENCIES:

pygments
jinja2
markdown
requests
mdx_truly_sane_lists
"""

import shutil
import string
import copy
import json
import re
import os
import sys
from pathlib import Path
import http.server
import socketserver
import signal
import docstrings
import jinja2
import markdown
import requests
import multiprocessing

from master import MASTER
import tutobooks
import generate_tf_guides


EXAMPLES_GH_LOCATION = Path("keras-team") / "keras-io" / "blob" / "master" / "examples"
GUIDES_GH_LOCATION = Path("keras-team") / "keras-io" / "blob" / "master" / "guides"
PROJECT_URL = {
    "keras": "https://github.com/keras-team/keras/tree/v2.9.0/",
    "keras_tuner": "https://github.com/keras-team/keras-tuner/tree/1.1.2/",
    "keras_cv": "https://github.com/keras-team/keras-cv/tree/v0.2.8/",
    "keras_nlp": "https://github.com/keras-team/keras-nlp/tree/v0.3.0/",
}


class KerasIO:
    def __init__(
        self,
        master,
        url,
        templates_dir,
        md_sources_dir,
        site_dir,
        theme_dir,
        guides_dir,
        examples_dir,
        refresh_guides=False,
        refresh_examples=False,
    ):
        self.master = master
        self.url = url
        self.templates_dir = templates_dir
        self.md_sources_dir = md_sources_dir
        self.site_dir = site_dir
        self.theme_dir = theme_dir
        self.guides_dir = guides_dir
        self.examples_dir = examples_dir
        self.refresh_guides = refresh_guides
        self.refresh_examples = refresh_examples

        self.docstring_printer = docstrings.TFKerasDocumentationGenerator(PROJECT_URL)
        self.make_examples_master()

    def make_examples_master(self):
        for entry in self.master["children"]:
            if entry["path"] == "examples/":
                examples_entry = entry
                break
        for entry in examples_entry["children"]:  # e.g. {"path": "nlp", ...}
            children = entry.get("children", [])
            preexisting = [e["path"] for e in children]
            subdir = entry["path"]  # e.g. nlp
            path = Path(self.examples_dir) / subdir  # e.g. examples/nlp
            for fname in sorted(os.listdir(path)):
                if fname.endswith(".py"):  # e.g. examples/nlp/test.py
                    name = fname[:-3]
                    example_path = name.split("/")[-1]
                    if example_path not in preexisting:
                        f = open(path / fname, encoding="utf-8")
                        f.readline()
                        title_line = f.readline()
                        f.close()
                        assert title_line.startswith("Title: ")
                        title = title_line[len("Title: ") :]
                        children.append({"path": example_path, "title": title.strip()})
            entry["children"] = children

    def make_md_sources(self):
        print("Generating md sources")
        if os.path.exists(self.md_sources_dir):
            print("Clearing", self.md_sources_dir)
            shutil.rmtree(self.md_sources_dir)
        os.makedirs(self.md_sources_dir)

        self.make_tutobook_sources(
            guides=self.refresh_guides, examples=self.refresh_examples
        )
        self.sync_tutobook_templates()

        self.make_md_source_for_entry(self.master, path_stack=[], title_stack=[])
        self.sync_external_readmes_to_sources()  # Overwrite e.g. sources/governance.md

    def sync_external_readmes_to_sources(self):
        # TODO keras-team/keras/CONTRIBUTING -> contributing.md

        # keras-team/governance/README -> governance.md
        r = requests.get(
            "https://raw.githubusercontent.com/keras-team/"
            "governance/master/README.md"
        )
        content = r.text
        content = content[content.find("---\n") + 4 :]
        content = content.replace("\n#", "\n##")
        fpath = Path(self.md_sources_dir) / "governance.md"
        md = open(fpath).read()
        assert "{{sig_readme}}" in md
        md = md.replace("{{sig_readme}}", content)
        save_file(fpath, md)

    def preprocess_tutobook_md_source(
        self, md_content, fname, github_repo_dir, img_dir, site_img_dir
    ):
        # Insert colab button and github button.
        name = fname[:-3]
        md_content_lines = md_content.split("\n")
        button_lines = [
            "\n",
            '<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> '
            "[**View in Colab**](https://colab.research.google.com/github/"
            + github_repo_dir
            + "/ipynb/"
            + name
            + ".ipynb"
            + ")  "
            '<span class="k-dot">•</span>'
            '<img class="k-inline-icon" src="https://github.com/favicon.ico"/> '
            "[**GitHub source**](https://github.com/"
            + github_repo_dir
            + "/"
            + fname
            + ")",
            "\n",
        ]
        md_content_lines = md_content_lines[:6] + button_lines + md_content_lines[6:]
        md_content = "\n".join(md_content_lines)
        # Normalize img urls
        md_content = md_content.replace(
            str(img_dir) + "/" + name, self.url + site_img_dir
        )
        # Insert --- before H2 titles
        md_content = md_content.replace("\n## ", "\n---\n## ")
        return md_content

    def make_tutobook_sources_for_directory(
        self, src_dir, target_dir, img_dir, site_img_dir, github_repo_dir
    ):
        # e.g.
        # make_tutobook_sources_for_directory(
        #    "examples/nlp", "examples/nlp/md", "examples/nlp/img", "img/examples/nlp")
        print("Making tutobook sources for", src_dir)

        working_ipynb_dir = Path(src_dir) / "ipynb"
        if not os.path.exists(working_ipynb_dir):
            os.makedirs(working_ipynb_dir)

        for fname in os.listdir(src_dir):
            if fname.endswith(".py"):
                print("...Processing", fname)
                name = fname[:-3]
                py_path = Path(src_dir) / fname
                nb_path = working_ipynb_dir / (name + ".ipynb")
                md_path = Path(target_dir) / (name + ".md")
                print("...Write to", md_path)
                tutobooks.py_to_md(py_path, nb_path, md_path, img_dir)
                md_content = open(md_path).read()
                md_content = self.preprocess_tutobook_md_source(
                    md_content, fname, github_repo_dir, img_dir, site_img_dir
                )
                open(md_path, "w").write(md_content)
        shutil.rmtree(working_ipynb_dir)

    def make_tutobook_ipynbs(self):
        def process_one_dir(src_dir, target_dir):
            if os.path.exists(target_dir):
                print("Clearing", target_dir)
                shutil.rmtree(target_dir)
            os.makedirs(target_dir)
            for fname in os.listdir(src_dir):
                if fname.endswith(".py"):
                    print("...Processing", fname)
                    name = fname[:-3]
                    py_path = Path(src_dir) / fname
                    nb_path = target_dir / (name + ".ipynb")
                    py_to_nb(py_path, nb_path, fill_outputs=False)

        # Guides
        guides_dir = Path(self.guides_dir)
        ipynb_dir = guides_dir / "ipynb"
        process_one_dir(guides_dir, ipynb_dir)

        # Examples
        for name in os.listdir(self.examples_dir):
            path = Path(self.examples_dir) / name
            if os.path.isdir(path):
                ipynb_dir = path / "ipynb"
                process_one_dir(path, ipynb_dir)

    def add_example(self, path, working_dir=None):
        """e.g. add_example('vision/cats_and_dogs')"""

        # Prune out the ../ path
        if path.startswith("../examples/"):
            path = path.replace("../examples/", "")

        folder, name = path.split(os.path.sep)
        assert path.count(os.path.sep) == 1
        if name.endswith(".py"):
            name = name[:-3]

        ipynb_dir = Path(self.examples_dir) / folder / "ipynb"
        if not os.path.exists(ipynb_dir):
            os.makedirs(ipynb_dir)

        md_dir = Path(self.examples_dir) / folder / "md"
        if not os.path.exists(md_dir):
            os.makedirs(md_dir)

        img_dir = Path(self.examples_dir) / folder / "img"
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        py_path = Path(self.examples_dir) / folder / (name + ".py")
        md_path = md_dir / (name + ".md")
        nb_path = ipynb_dir / (name + ".ipynb")

        self.disable_warnings()
        tutobooks.py_to_nb(py_path, nb_path, fill_outputs=False)
        tutobooks.py_to_md(py_path, nb_path, md_path, img_dir, working_dir=working_dir)

        md_content = open(md_path).read()
        github_repo_dir = str(EXAMPLES_GH_LOCATION / folder)
        site_img_dir = os.path.join("img", "examples", folder, name)
        md_content = self.preprocess_tutobook_md_source(
            md_content, name + ".py", github_repo_dir, img_dir, site_img_dir
        )
        open(md_path, "w").write(md_content)

    def add_guide(self, name, working_dir=None):
        """e.g. add_guide('functional_api')"""

        # Prune out the ../ path
        if name.startswith("../guides/"):
            name = name.replace("../guides/", "")

        if name.endswith(".py"):
            name = name[:-3]
        ipynb_dir = Path(self.guides_dir) / "ipynb"
        if not os.path.exists(ipynb_dir):
            os.makedirs(ipynb_dir)

        md_dir = Path(self.guides_dir) / "md"
        if not os.path.exists(md_dir):
            os.makedirs(md_dir)

        img_dir = Path(self.guides_dir) / "img"
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        py_path = Path(self.guides_dir) / (name + ".py")
        md_path = md_dir / (name + ".md")
        nb_path = ipynb_dir / (name + ".ipynb")

        self.disable_warnings()
        tutobooks.py_to_nb(py_path, nb_path, fill_outputs=False)
        tutobooks.py_to_md(py_path, nb_path, md_path, img_dir, working_dir=working_dir)

        md_content = open(md_path).read()
        github_repo_dir = str(GUIDES_GH_LOCATION)
        site_img_dir = "img/guides/" + name
        md_content = self.preprocess_tutobook_md_source(
            md_content, name + ".py", github_repo_dir, img_dir, site_img_dir
        )
        open(md_path, "w").write(md_content)

    @staticmethod
    def disable_warnings():
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['AUTOGRAPH_VERBOSITY'] = '0'

    def make_tutobook_sources(self, guides=True, examples=True):
        """Populate `examples/nlp/md`, `examples/nlp/img/`, etc.

        - guides/md/ & /png/
        - examples/nlp/md/ & /png/
        - examples/computer_vision/md/ & /png/
        - examples/structured_data/md/ & /png/
        - examples/timeseries/md/ & /png/
        - examples/generative_dl/md/ & /png/
        - examples/keras_recipes/md/ & /png/
        """
        # Guides
        if guides:
            target_dir = Path(self.guides_dir) / "md"
            img_dir = Path(self.guides_dir) / "img"
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            if os.path.exists(img_dir):
                shutil.rmtree(img_dir)
            os.makedirs(target_dir)
            os.makedirs(img_dir)
            self.make_tutobook_sources_for_directory(
                src_dir=Path(self.guides_dir),
                target_dir=target_dir,
                img_dir=img_dir,
                site_img_dir="img/guides/",
                github_repo_dir=str(GUIDES_GH_LOCATION),
            )

        # Examples
        if examples:
            for name in os.listdir(self.examples_dir):
                path = Path(self.examples_dir) / name
                if os.path.isdir(path):
                    target_dir = path / "md"
                    img_dir = path / "img"
                    if os.path.exists(target_dir):
                        shutil.rmtree(target_dir)
                    if os.path.exists(img_dir):
                        shutil.rmtree(img_dir)
                    os.makedirs(target_dir)
                    os.makedirs(img_dir)
                    self.make_tutobook_sources_for_directory(
                        src_dir=path,  # e.g. examples/nlp
                        target_dir=target_dir,  # e.g. examples/nlp/md
                        img_dir=img_dir,  # e.g. examples/nlp/img
                        site_img_dir="img/examples/" + name,  # e.g. img/examples/nlp
                        github_repo_dir=str(EXAMPLES_GH_LOCATION / name),
                    )

    def sync_tutobook_templates(self):
        """Copy generated `.md`s to source_dir.

        Note: intro guides are copied to getting_started.

        guides/md/ -> sources/guides/
        guides/md/intro_* -> sources/getting_started/
        examples/*/md/ -> sources/examples/*/
        """
        # Guides
        copy_inner_contents(
            Path(self.guides_dir) / "md", Path(self.templates_dir) / "guides", ext=".md"
        )
        # Special cases
        shutil.copyfile(
            Path(self.templates_dir) / "guides" / "intro_to_keras_for_engineers.md",
            Path(self.templates_dir)
            / "getting_started"
            / "intro_to_keras_for_engineers.md",
        )
        shutil.copyfile(
            Path(self.templates_dir) / "guides" / "intro_to_keras_for_researchers.md",
            Path(self.templates_dir)
            / "getting_started"
            / "intro_to_keras_for_researchers.md",
        )

        # Examples
        for dir_name in os.listdir(Path(self.examples_dir)):
            dir_path = Path(self.examples_dir) / dir_name  # e.g. examples/nlp
            if os.path.isdir(dir_path):
                dst_dir = Path(self.templates_dir) / "examples" / dir_name
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                copy_inner_contents(dir_path / "md", dst_dir, ext=".md")

    def sync_tutobook_media(self):
        """Copy generated `.png`s to site_dir.

        Note: intro guides are copied to getting_started.

        guides/img/ -> site/img/guides/
        examples/*/img/ -> site/img/examples/*/
        """
        # Copy images for guide notebooks
        for name in os.listdir(Path(self.guides_dir) / "img"):
            path = Path(self.guides_dir) / "img" / name
            if os.path.isdir(path):
                shutil.copytree(path, Path(self.site_dir) / "img" / "guides" / name)
        # Copy images for examples notebooks
        for dir_name in os.listdir(Path(self.examples_dir)):
            dir_path = Path(self.examples_dir) / dir_name
            if os.path.isdir(dir_path):
                if not os.path.exists(dir_path / "img"):
                    continue  # No media was generated for this tutobook.

                dst_dir = Path(self.site_dir) / "img" / "examples" / dir_name
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)

                for name in os.listdir(dir_path / "img"):
                    path = dir_path / "img" / name
                    if os.path.isdir(path):
                        shutil.copytree(
                            path,
                            Path(self.site_dir) / "img" / "examples" / dir_name / name,
                        )

    def make_nav_index(self):
        max_depth = 3
        path_stack = []

        def make_nav_index_for_entry(entry, path_stack, max_depth):
            path = entry["path"]
            if path != "/":
                path_stack.append(path)
            url = self.url + str(Path(*path_stack)) + "/"
            relative_url = "/" + str(Path(*path_stack)) + "/"
            if len(path_stack) < max_depth:
                children = [
                    make_nav_index_for_entry(child, path_stack[:], max_depth)
                    for child in entry.get("children", [])
                ]
            else:
                children = []
            return {
                "title": entry["title"],
                "relative_url": relative_url,
                "url": url,
                "children": children,
            }

        return [
            make_nav_index_for_entry(entry, path_stack[:], max_depth)
            for entry in self.master["children"]
        ]

    def make_md_source_for_entry(self, entry, path_stack, title_stack):
        path = entry["path"]
        if path != "/":
            path_stack.append(path)
            title_stack.append(entry["title"])
        print("...Processing", Path(*path_stack))
        parent_url = self.url + str(Path(*path_stack)) + "/"
        if path.endswith("/"):
            dir_path = Path(self.md_sources_dir) / Path(*path_stack)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        template_path = Path(self.templates_dir) / Path(*path_stack)
        if path.endswith("/"):
            template_path /= "index.md"
        else:
            template_path = template_path.with_suffix(".md")

        if os.path.exists(template_path):
            template_file = open(template_path, encoding="utf8")
            print(template_path)
            template = template_file.read()
            template_file.close()
        else:
            template = ""
            if entry.get("toc"):
                template += "{{toc}}\n\n"
            if entry.get("generate"):
                template += "{{autogenerated}}\n"
        if not template.startswith("# "):
            template = "# " + entry["title"] + "\n\n" + template
        generate = entry.get("generate")
        children = entry.get("children")
        if generate:
            generated_md = ""
            for element in generate:
                generated_md += self.docstring_printer.render(element)
            if "{{autogenerated}}" not in template:
                raise RuntimeError(
                    "Template found for %s but missing "
                    "{{autogenerated}} tag." % (template_path,)
                )
            template = template.replace("{{autogenerated}}", generated_md)
        if entry.get("toc"):
            if not children:
                raise ValueError(
                    "For template %s, "
                    "a table of contents was requested but "
                    "the entry had no "
                    "children" % (template_path,)
                )
            toc = generate_md_toc(children, parent_url)
            if "{{toc}}" not in template:
                raise RuntimeError(
                    "Table of contents requested for %s but "
                    "missing {{toc}} tag." % (template_path,)
                )
            template = template.replace("{{toc}}", toc)
        source_path = Path(self.md_sources_dir) / Path(*path_stack)
        if path.endswith("/"):
            md_source_path = source_path / "index.md"
            metadata_path = source_path / "index_metadata.json"
        else:
            md_source_path = source_path.with_suffix(".md")
            metadata_path = str(source_path) + "_metadata.json"

        # Save md source file
        save_file(md_source_path, template)

        # Save metadata file
        location_history = []
        for i in range(len(path_stack)):
            stripped_path_stack = [s.replace("/", "") for s in path_stack[: i + 1]]
            url = self.url + "/".join(stripped_path_stack)
            location_history.append(
                {
                    "url": url,
                    "title": title_stack[i],
                }
            )
        metadata = json.dumps(
            {
                "location_history": location_history[:-1],
                "outline": make_outline(template) if entry.get("outline", True) else [],
                "location": "/" + "/".join([s.replace("/", "") for s in path_stack]),
                "url": parent_url,
                "title": entry["title"],
            }
        )
        save_file(metadata_path, metadata)

        if children:
            for entry in children:
                self.make_md_source_for_entry(entry, path_stack[:], title_stack[:])

    def make_map_of_symbol_names_to_api_urls(self):
        def recursive_make_map(entry, current_url):
            current_url /= entry["path"]
            entry_map = {}
            if "generate" in entry:
                for symbol in entry["generate"]:
                    object_ = docstrings.import_object(symbol)
                    object_type = docstrings.get_type(object_)
                    object_name = symbol.split(".")[-1]

                    if symbol.startswith("tensorflow.keras."):
                        symbol = symbol.replace("tensorflow.keras.", "keras.")
                    object_name = object_name.lower().replace("_", "")
                    entry_map[symbol] = (
                        str(current_url) + "#" + object_name + "-" + object_type
                    )

            if "children" in entry:
                for child in entry["children"]:
                    entry_map.update(recursive_make_map(child, current_url))
            return entry_map

        self._map_of_symbol_names_to_api_urls = recursive_make_map(
            self.master, Path("")
        )

    def render_md_sources_to_html(self):
        self.make_map_of_symbol_names_to_api_urls()
        print("Rendering md sources to HTML")
        base_template = jinja2.Template(open(Path(self.theme_dir) / "base.html").read())
        docs_template = jinja2.Template(open(Path(self.theme_dir) / "docs.html").read())

        all_urls_list = []

        if os.path.exists(self.site_dir):
            print("Clearing", self.site_dir)
            shutil.rmtree(self.site_dir)

        nav = self.make_nav_index()

        for src_location, _, fnames in os.walk(self.md_sources_dir):

            pool = multiprocessing.Pool(processes=8)
            workers = [
                pool.apply_async(
                    self.render_single_file, args=(src_location, fname, nav)
                )
                for fname in fnames
            ]

            for worker in workers:
                url = worker.get()
                if url is not None:
                    all_urls_list.append(url)
            pool.close()
            pool.join()

        # Images & css
        shutil.copytree(Path(self.theme_dir) / "css", Path(self.site_dir) / "css")
        shutil.copytree(Path(self.theme_dir) / "img", Path(self.site_dir) / "img")

        # Landing page
        landing_template = jinja2.Template(
            open(Path(self.theme_dir) / "landing.html").read()
        )
        landing_page = landing_template.render({"base_url": self.url})
        save_file(Path(self.site_dir) / "index.html", landing_page)

        # Search page
        search_main = open(Path(self.theme_dir) / "search.html").read()
        search_page = base_template.render(
            {
                "title": "Search Keras documentation",
                "nav": nav,
                "base_url": self.url,
                "main": search_main,
            }
        )
        save_file(Path(self.site_dir) / "search.html", search_page)

        # 404 page
        page404 = base_template.render(
            {
                "title": "Page not found",
                "nav": nav,
                "base_url": self.url,
                "main": docs_template.render(
                    {
                        "title": "404",
                        "content": "<h1>404: Page not found</h1>",
                        "base_url": self.url,
                    }
                ),
            }
        )
        save_file(Path(self.site_dir) / "404.html", page404)

        # Tutobooks
        self.sync_tutobook_media()
        sitemap = "\n".join(all_urls_list) + "\n"
        save_file(Path(self.site_dir) / "sitemap.txt", sitemap)

    def render_single_file(self, src_location, fname, nav):
        if not fname.endswith(".md"):
            return

        base_template = jinja2.Template(open(Path(self.theme_dir) / "base.html").read())
        docs_template = jinja2.Template(open(Path(self.theme_dir) / "docs.html").read())

        src_dir = Path(src_location)
        target_dir = src_location.replace(self.md_sources_dir, self.site_dir)
        if not os.path.exists(target_dir):
            try:
                os.makedirs(target_dir)
            except FileExistsError:
                # Might be created by a concurrent process.
                pass

        # Load metadata for page
        metadata_file = open(str(Path(src_location) / fname[:-3]) + "_metadata.json")
        metadata = json.loads(metadata_file.read())
        metadata_file.close()
        if fname == "index.md":
            # Render as index.html
            target_path = Path(target_dir) / "index.html"
            relative_url = (str(target_dir) + "/").replace(self.site_dir, "/")
            relative_url = relative_url.replace("//", "/")
        else:
            # Render as fname_no_ext/index.tml
            fname_no_ext = ".".join(fname.split(".")[:-1])
            full_target_dir = Path(target_dir) / fname_no_ext
            os.makedirs(full_target_dir)
            target_path = full_target_dir / "index.html"
            relative_url = (str(full_target_dir) + "/").replace(self.site_dir, "/")
            relative_url = relative_url.replace("//", "/")

        md_file = open(src_dir / fname, encoding="utf-8")
        md_content = md_file.read()
        md_file.close()
        md_content = replace_links(md_content)

        # Convert Keras symbols to links to the Keras docs
        for symbol, symbol_url in self._map_of_symbol_names_to_api_urls.items():
            md_content = re.sub(
                r"`((tf\.|)" + symbol + ")`", r"[`\1`](" + symbol_url + ")", md_content
            )

        # Convert TF symbols to links to tensorflow.org
        tmp_content = copy.copy(md_content)
        replacements = {}
        while "`tf." in tmp_content:
            index = tmp_content.find("`tf.")
            if tmp_content[index - 1] == "[":
                tmp_content = tmp_content[tmp_content.find("`tf.") + 1 :]
                tmp_content = tmp_content[tmp_content.find("`") + 1 :]
            else:
                tmp_content = tmp_content[tmp_content.find("`tf.") + 1 :]
                symbol = tmp_content[: tmp_content.find("`")]
                tmp_content = tmp_content[tmp_content.find("`") + 1 :]
                if "/" not in symbol and "(" not in symbol:
                    path = symbol.replace(".", "/")
                    path = path.replace("(", "")
                    path = path.replace(")", "")
                    replacements["`" + symbol + "`"] = (
                        "[`"
                        + symbol
                        + "`](https://www.tensorflow.org/api_docs/python/"
                        + path
                        + ")"
                    )
        for key, value in replacements.items():
            md_content = md_content.replace(key, value)

        html_content = markdown.markdown(
            md_content,
            extensions=[
                "fenced_code",
                "tables",
                "codehilite",
                "mdx_truly_sane_lists",
            ],
            extension_configs={
                "codehilite": {
                    "guess_lang": False,
                },
            },
        )
        html_content = insert_title_ids_in_html(html_content)
        local_nav = [set_active_flag_in_nav_entry(entry, relative_url) for entry in nav]

        title = md_content[2 : md_content.find("\n")]
        html_docs = docs_template.render(
            {
                "title": title,
                "content": html_content,
                "location_history": metadata["location_history"],
                "base_url": self.url,
                "outline": metadata["outline"],
            }
        )
        html_page = base_template.render(
            {
                "title": title,
                "nav": local_nav,
                "base_url": self.url,
                "main": html_docs,
            }
        )
        save_file(target_path, html_page)
        return relative_url

    def make(self):
        self.make_md_sources()
        self.render_md_sources_to_html()
        self.make_tutobook_ipynbs()

    def serve(self):
        os.chdir(self.site_dir)
        socketserver.ThreadingTCPServer.allow_reuse_address = True
        server = socketserver.ThreadingTCPServer(
            ("", 8000), http.server.SimpleHTTPRequestHandler
        )
        server.daemon_threads = True

        def signal_handler(signal, frame):
            try:
                if server:
                    server.server_close()
            finally:
                sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        try:
            print("Serving on 0.0.0.0:8000")
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


def save_file(path, content):
    f = open(path, "w", encoding="utf8")
    f.write(content)
    f.close()


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


def replace_links(content):
    # Make sure all Keras guides point to keras.io.
    for entry in generate_tf_guides.CONFIG:
        keras_name = entry["source_name"]
        tf_name = entry["target_name"]
        content = content.replace(
            "https://www.tensorflow.org/guide/keras/" + tf_name,
            "https://keras.io/guides/" + keras_name,
        )
    return content


def process_outline_title(title):
    title = re.sub(r"`(.*?)`", r"<code>\1</code>", title)
    title = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", title)
    return title


def strip_markdown_tags(md):
    # Strip links
    md = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", md)
    return md


def copy_inner_contents(src, dst, ext=".md"):
    for fname in os.listdir(src):
        fpath = Path(src) / fname
        fdst = Path(dst) / fname
        if fname.endswith(ext):
            shutil.copyfile(fpath, fdst)
        if os.path.isdir(fpath):
            copy_inner_contents(fpath, fdst, ext)


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


def insert_title_ids_in_html(html):
    marker = "replace_me_with_id_for:"
    marker_end = ":end_of_title"
    for i in range(1, 5):
        match = "<h" + str(i) + ">(.*?)</h" + str(i) + ">"
        replace = (
            "<h"
            + str(i)
            + r' id="'
            + marker
            + r"\1"
            + marker_end
            + r'">\1</h'
            + str(i)
            + ">"
        )
        html = re.sub(match, replace, html)

    while 1:
        start = html.find(marker)
        if start == -1:
            break
        title = html[start + len(marker) :]
        title = title[: title.find(marker_end)]
        normalized_title = title
        normalized_title = normalized_title.replace("<code>", "")
        normalized_title = normalized_title.replace("</code>", "")
        if ">" in normalized_title:
            normalized_title = normalized_title[normalized_title.find(">") + 1 :]
            normalized_title = normalized_title[: normalized_title.find("</")]
        normalized_title = turn_title_into_id(normalized_title)
        html = html.replace(marker + title + marker_end, normalized_title)
    return html


def turn_title_into_id(title):
    title = title.lower()
    title = title.replace("&amp", "amp")
    title = title.replace("&", "amp")
    title = title.replace("<code>", "")
    title = title.replace("</code>", "")
    title = title.translate(str.maketrans("", "", string.punctuation))
    title = title.replace(" ", "-")
    return title


def generate_md_toc(entries, url, depth=2):
    assert url.endswith("/")
    entries = [e for e in entries if not e.get("skip_from_toc")]
    generated = ""
    if set(len(x.get("generate", [])) for x in entries) == {1}:
        print_generate = False
    else:
        print_generate = True
    for entry in entries:
        title = entry["title"]
        path = entry["path"]
        full_url = url + path
        children = entry.get("children")
        generate = entry.get("generate")
        if children or (print_generate and generate):
            title_prefix = "### "
        else:
            title_prefix = "- "
        generated += title_prefix + "[{title}]({full_url})\n".format(
            title=title, full_url=full_url
        )
        if children:
            assert path.endswith("/"), f"{path} should end with /"
            for child in children:
                if child.get("skip_from_toc", False):
                    continue
                child_title = child["title"]
                child_path = child["path"]
                child_url = full_url + child_path
                generated += "- [{child_title}]({child_url})\n".format(
                    child_title=child_title, child_url=child_url
                )
            generated += "\n"
        elif generate and print_generate:
            for gen in generate:
                obj = docstrings.import_object(gen)
                obj_name = docstrings.get_name(obj)
                obj_type = docstrings.get_type(obj)
                link = "{full_url}/#{obj_name}-{obj_type}".format(
                    full_url=full_url, obj_name=obj_name, obj_type=obj_type
                ).lower()
                name = gen.split(".")[-1]
                generated += "- [{name} {obj_type}]({link})\n".format(
                    name=name, obj_type=obj_type, link=link
                )
            generated += "\n"
    return generated


def get_working_dir(arg):
    if not arg.startswith("--working_dir="):
        return None
    return arg[len("--working_dir=") :]


if __name__ == "__main__":
    keras_io = KerasIO(
        master=MASTER,
        url=os.path.sep,
        templates_dir=os.path.join("..", "templates"),
        md_sources_dir=os.path.join("..", "sources"),
        site_dir=os.path.join("..", "site"),
        theme_dir=os.path.join("..", "theme"),
        guides_dir=os.path.join("..", "guides"),
        examples_dir=os.path.join("..", "examples"),
        refresh_guides=False,
        refresh_examples=False,
    )

    cmd = sys.argv[1]
    if cmd not in {"make", "serve", "add_example", "add_guide", "generate_tf_guides"}:
        raise ValueError(
            "Must specify command `make`, `serve`, `add_example`, `add_guide` or `generate_tf_guides`."
        )
    if cmd in {"add_example", "add_guide"}:
        if not len(sys.argv) in (3, 4):
            raise ValueError(
                "Must specify example/guide to add, e.g. "
                "`autogen.py add_example vision/cats_and_dogs`"
            )
    if cmd == "make":
        keras_io.make_md_sources()
        keras_io.render_md_sources_to_html()
    elif cmd == "serve":
        keras_io.serve()
    elif cmd == "add_example":
        keras_io.add_example(
            sys.argv[2],
            working_dir=get_working_dir(sys.argv[3]) if len(sys.argv) == 4 else None,
        )
    elif cmd == "add_guide":
        tutobooks.MAX_LOC = 500
        keras_io.add_guide(
            sys.argv[2],
            working_dir=get_working_dir(sys.argv[3]) if len(sys.argv) == 4 else None,
        )
    elif cmd == "generate_tf_guides":
        generate_tf_guides.generate_tf_guides()
