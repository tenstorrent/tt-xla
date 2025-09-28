import argparse
import xml.etree.ElementTree as ET
import pdb
import unicodedata
import re

def safe_filename_slug(s: str, maxlen: int = 128) -> str:
    s = (s or "unnamed").strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^\w.\-]+", "_", s) # keep letters, digits, _, ., -
    s = s.strip("._")
    return s[:maxlen] or "unnamed"

class ExceptionParser:
    def __init__(self, regex=None):
        self.pattern = re.compile(r"E AttributeError: '([^']+)' object has no attribute 'shape'")

    def parse(self,testcase):
        text = testcase.find("failure").text
        m = self.pattern.search(text)
        if m:
            obj = m.groups()
            print(obj)

class ErrorParser:
    def __init__(self):
        self.exception_parsers = {
            "AttributeError" : ExceptionParser()
        }

    def parse(self, testcase, exception):
        if exception not in self.exception_parsers:
            return 
        return self.exception_parsers[exception].parse(testcase)

error_parser = ErrorParser()

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    return root

def traverse_tests(root):
    for testcase in root.iter("testcase"):
        yield testcase

def handle_testcase(test_case):
    test_name = test_case.get("name")
    assert test_name is not None, "Test name must be present"
    failure = test_case.find("failure")
    if failure is None:
        return
    failure_msg = failure.get("message")
    exception = failure_msg.split(":")[0]
    error_parser.parse(test_case, exception)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_file", type=str, required=False, default="./results.junit.xml")
    args = parser.parse_args()

    root = parse_xml(args.xml_file)
    for testcase in traverse_tests(root):
        handle_testcase(testcase)

if __name__ == "__main__":
    main()