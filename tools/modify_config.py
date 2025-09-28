import libcst as cst
import libcst.matchers as m
from libcst.display import dump_graphviz
from typing import Optional, Union
import pdb


EXPECTED = cst.Attribute(value=cst.Name("ModelTestStatus"),
                         attr=cst.Name("EXPECTED_PASSING"))

MODEL_TARGET = '"mnist/pytorch-full-training"'
TARGET_CONFIG = "test_config"
TARGET_FIELD = '"status"'

class DictModifier(m.MatcherDecoratableTransformer):
    def visit_SimpleStatementLine(self, _node):
        return m.matches(_node, m.SimpleStatementLine(
            body=[m.Assign(targets=[m.AssignTarget(target=m.Name(TARGET_CONFIG))])]
        ))

    def visit_DictElement(self, _node):
        return m.matches(_node, m.DictElement(
            key=m.SimpleString(MODEL_TARGET),
            value=m.Dict(),
        ))

    def leave_DictElement(self, orig: cst.DictElement, _upd: cst.DictElement):
        if not m.matches(orig, m.DictElement(key=m.SimpleString(TARGET_FIELD))):
            return _upd
        return _upd.with_changes(value=EXPECTED)


def main(path):
    with open(path, "r") as f:
        src_text = f.read()
        module = cst.parse_module(src_text)
        with open("dot.dot", "w") as f:
            f.write(dump_graphviz(module))
        pdb.set_trace()
    updater = DictModifier()
    modified = module.visit(updater)
    with open(path, "w") as f:
        print(modified.code)
        f.write(modified.code)

    pass





if __name__ == "__main__":
    main("./test_config.py")