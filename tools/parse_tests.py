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

DEBUG_VERB = True

class AttributeErrorParser:
    def __init__(self, regex=None):
        self.pattern = re.compile(r"AttributeError: '([^']+)' object has no attribute 'shape'")

    def parse(self,testcase):
        text = testcase.find("failure").text
        m = self.pattern.search(text)
        if m:
            obj = m.groups()
            if DEBUG_VERB:
                print(testcase.get("name"), obj)


class ValueErrorParser:
    def __init__(self):
        self.pattern_13 = re.compile(r"ValueError: Error code: 13")
        self.pattern_ird = re.compile(r"ValueError: IRD_LF_CACHE .*")
        self.pattern_f_legal = re.compile(r"error: failed to legalize operation '([^']+)'")

    def parse(self, testcase):
        test_name = testcase.get("name")
        fail_text = testcase.find("failure").text
        m = self.pattern_13.search(fail_text)
        if m:
            fm = self.pattern_f_legal.search(testcase.find("system-err").text)
            if fm:
                op_name = fm.groups()
                if DEBUG_VERB:
                    print(testcase.get("name"), "Failed to legalize:", op_name[0])
                return
        m = self.pattern_ird.search(fail_text)
        if m:
            if DEBUG_VERB:
                print(testcase.get("name"), m.group())
            return
        with open(f"dumps/{safe_filename_slug(test_name)}", "w") as f:
            f.write(fail_text)

class OSErrorParser:
    def parse(self, testcase):
        test_name = testcase.get("name")
        fail_text = testcase.find("failure").text
        if DEBUG_VERB:
            # print(test_name)
            with open(f"dumps/{safe_filename_slug(test_name)}", "w") as f:
                f.write(fail_text)

class RuntimeErrorParser:
    def parse(self, testcase):
        test_name = testcase.get("name")
        fail_text = testcase.find("failure").text
        # print(test_name)
        with open(f"dumps/{safe_filename_slug(test_name)}", "w") as f:
            f.write(fail_text)



class ErrorParser:
    def __init__(self):
        self.exception_parsers = {
            "AttributeError" : AttributeErrorParser(),
            "ValueError": ValueErrorParser(),
            "OSError": OSErrorParser(),
            "RuntimeError": RuntimeErrorParser(),
        }

    def parse(self, testcase, exception):
        if exception not in self.exception_parsers:
            print(testcase.get("name"), exception)
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
    parser.add_argument("--xml_file", type=str, required=False, default="./results1.junit.xml")
    args = parser.parse_args()

    root = parse_xml(args.xml_file)
    for testcase in traverse_tests(root):
        handle_testcase(testcase)

if __name__ == "__main__":
    main()