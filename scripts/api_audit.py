from master import MASTER
import keras
import tensorflow

MODULE_CLS = type(keras)


def audit_api_docs():
    all_documented_symbols = set()

    def process_doc_module(module):
        if "generate" in module:
            for symbol in module["generate"]:
                try:
                    symbol = eval(symbol)
                except NameError:
                    continue
                all_documented_symbols.add(symbol)
        if "children" in module:
            for child in module["children"]:
                process_doc_module(child)

    for child in MASTER["children"]:
        process_doc_module(child)

    missing = []

    def walk_module(module_path):
        module = eval(module_path)
        elements = dir(module)
        for name in elements:
            if name.startswith("_"):
                continue
            element_path = module_path + "." + name
            element = eval(element_path)

            if isinstance(element, MODULE_CLS):
                walk_module(element_path)
            else:
                if element not in all_documented_symbols:
                    missing.append(element_path)

    walk_module("keras")
    for path in missing:
        print("Undocumented:", path)


def compare_two_modules(ref_module_path, target_module_path):
    missing = []
    different = []
    ref_module = eval(ref_module_path)
    target_module = eval(target_module_path)
    ref_elements = dir(ref_module)
    target_elements = dir(target_module)
    for name in ref_elements:
        if name.startswith("_"):
            continue
        ref_element_path = ref_module_path + "." + name
        if name not in target_elements:
            print("Missing:", ref_element_path)
            missing.append(ref_element_path)
        else:
            ref_element = eval(ref_element_path)
            target_element_path = target_module_path + "." + name
            if isinstance(ref_element, MODULE_CLS):
                missing_2, different_2 = compare_two_modules(
                    ref_element_path, target_element_path
                )
                missing += missing_2
                different += different_2
            else:
                target_element = eval(target_element_path)
                if target_element is not ref_element:
                    different.append(ref_element_path)
                    print("Different:", ref_element_path)
    return missing, different


def audit_keras_vs_tf_keras():
    compare_two_modules("tensorflow.keras", "keras")


def list_all_keras_ops():
    all_ops = [e for e in dir(keras.ops) if not e.startswith("_")]
    for o in all_ops:
        print(o)
    print("total", len(all_ops))

    np_ops = [e for e in dir(keras.ops.numpy) if not e.startswith("_")]
    print("numpy", len(np_ops))


if __name__ == "__main__":
    audit_api_docs()
