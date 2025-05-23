import time
from contextlib import contextmanager
from typing import Generator
from jax.tree_util import KeyPath, SequenceKey, DictKey, GetAttrKey, FlattenedIndexKey


@contextmanager
def timer(desc: str) -> Generator[None, None, None] :
    """
    A context manager to time the execution of a code block.

    Args:
        description (str): A description for the code block being timed.
                           This will be included in the output message.
    """
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"{elapsed_time:4.4f}s elapsed for {desc}")


def keystr_simple(keypath: KeyPath, separator: str = "") -> str:
    """Backported equivalent to jax.treeutil.keystr(keypath,simple=True, delimiter=...). 
    """

    def simple(k):
        if isinstance(k, SequenceKey):
            return str(k.idx)
        elif isinstance(k, DictKey):
            return str(k.key)
        elif isinstance(k, GetAttrKey):
            return str(k.name)
        elif isinstance(k, FlattenedIndexKey):
            return str(k.index)
        else:
            return str(k)

    return separator.join(
        simple(k) for k in keypath
    )