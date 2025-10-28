import argparse
from webbrowser import get
import xml.etree.ElementTree as ET
import pdb
import unicodedata
from pathlib import Path
from collections import Counter
import re

from failure_msg_patterns import failure_msg_patterns
from print_order import print_order

class IndentPrinter:
    def __init__(self, indent_str="\t"):
        self.level = 0
        self.indent_str = indent_str

    def write(self, text):
        # print one logical line at current indent
        print(f"{self.indent_str * self.level}{text}")

    def indent(self):
        # use this in a `with` block
        class _IndentCtx:
            def __init__(self, outer):
                self.outer = outer
            def __enter__(self):
                self.outer.level += 1
            def __exit__(self, exc_type, exc, tb):
                self.outer.level -= 1
        return _IndentCtx(self)

p = IndentPrinter(indent_str="    ")

def get_model_name(test_name: str) -> str:
    return test_name.split("[")[1][:-1]

def safe_filename_slug(s: str, maxlen: int = 128) -> str:
    s = (s or "unnamed").strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^\w.\-]+", "_", s) # keep letters, digits, _, ., -
    s = s.strip("._")
    return s[:maxlen] or "unnamed"

pattern_13 = re.compile(r"ValueError: Error code: 13")
pattern_ird = re.compile(r"ValueError: IRD_LF_CACHE .*")
pattern_f_legal = re.compile(r"error: failed to legalize operation '([^']+)'")

def group_failure_message(message: str) -> bool:
    # for pattern in failure_msg_patterns:
    #     if message.startswith(pattern):
    #         return pattern
    return message

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    return root

def traverse_tests(root):
    for testcase in root.iter("testcase"):
        yield testcase

def handle_folder(folder: Path):
    p.write(f"Processing folder: {folder}")
    for file in folder.glob("*.xml"):
        with p.indent():
            handle_file(file)

def handle_file(file: Path):
    root = parse_xml(file)
    p.write(f"Processing file: {file}")
    for testcase in traverse_tests(root):
        with p.indent():
            handle_testcase(testcase)

def handle_testcase(test_case):
    test_name = test_case.get("name")
    assert test_name is not None, "Test name must be present"
    p.write(get_model_name(test_name))
    with p.indent():
        system_out = test_case.find("system-out")
        assert system_out is not None, "System-out must be present"
        system_err = test_case.find("system-err")
        assert system_err is not None, "System-err must be present"
        failure = test_case.find("failure")
        if failure is None:
            handle_wo_failure(test_case)
        else:
            handle_with_failure(test_case)
        # pdb.set_trace()

def handle_wo_failure(test_case):
    p.write("No failure")

def handle_with_failure(test_case):
    failure = test_case.find("failure")
    test_name = get_model_name(test_case.get("name"))
    assert failure is not None, "Failure must be present"
    message = failure.get("message")
    assert message is not None, "Message must be present"
    grouped_message = group_failure_message(message)
    tests[test_name] = {"grouped_message": grouped_message, "failure_message": message}
    stderr = test_case.find("system-err")
    # check for failed to legalize operation
    re_legalize = re.compile(r"failed to legalize operation '([^']+)'")
    match = re_legalize.search(stderr.text)
    if match:
        legalize_operation = match.group(1)
        tests[test_name]["legalize_operations"] = legalize_operation

tests = {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_file", type=str, required=False)
    parser.add_argument("--folder", type=str, required=False)
    args = parser.parse_args()

    if args.xml_file is not None:
        handle_file(Path(args.xml_file))
    elif args.folder is not None:
        handle_folder(Path(args.folder))
    else:
        print("Error: Either --xml_file or --folder must be provided")

    new_line = "\n"
    failure_message_counter = Counter([msg.split(new_line)[0] for msg in (tests[test]['grouped_message'] for test in tests)])
    print("Failure messages:")
    for ids, (message, count) in enumerate(sorted(failure_message_counter.items(), key=lambda x: x[1], reverse=True)):
        print(f"{ids+1}: {count}: {message.split(new_line)[0]}")
    print("Legalize operations:")
    legalize_operation_counter = Counter([tests[test]['legalize_operations'] for test in tests if 'legalize_operations' in tests[test]])
    total = sum(legalize_operation_counter.values())
    for ids, (operation, count) in enumerate(sorted(legalize_operation_counter.items(), key=lambda x: x[1], reverse=True)):
        print(f"{ids+1}: {count}: {operation}")
    
    print("-"*100)
    print(f"Total: {total}")

    for test in print_order:
        if test not in tests:
            print(f"{test}: Not found")
        else:
            print(f"{test}: {tests[test]['grouped_message']}{(' Legalize operations: ' + tests[test]['legalize_operations']) if 'legalize_operations' in tests[test] else ' Not found'}")


if __name__ == "__main__":
    main()