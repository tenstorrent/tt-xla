# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Utilities for failing reasons

import io
import os
import re
import sys
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, ClassVar, List, Optional

from loguru import logger
from pytest import ExceptionInfo


# Thread-local storage for captured C++ stderr during compilation
_thread_local = threading.local()


def get_captured_cpp_stderr() -> Optional[str]:
    """Get the captured C++ stderr from the current thread."""
    return getattr(_thread_local, "cpp_stderr", None)


def set_captured_cpp_stderr(value: Optional[str]):
    """Set the captured C++ stderr for the current thread."""
    _thread_local.cpp_stderr = value


def _tty_debug(msg):
    """Write debug message to /dev/tty to bypass all capture."""
    try:
        tty = os.open('/dev/tty', os.O_WRONLY)
        os.write(tty, f"{msg}\n".encode())
        os.close(tty)
    except:
        pass


@contextmanager
def capture_cpp_stderr():
    """
    Context manager to capture stderr AND stdout at the file descriptor level.
    This captures output from C++ code (like MLIR compilation errors)
    that bypasses Python's sys.stderr/sys.stdout.

    The captured output is stored in thread-local storage and can be
    retrieved using get_captured_cpp_stderr().

    Usage:
        with capture_cpp_stderr():
            # C++ code that prints to stderr/stdout
            model.to(device)

        captured = get_captured_cpp_stderr()
        if captured and "error:" in captured:
            print(f"MLIR error: {captured}")
    """
    _tty_debug("[capture_cpp_stderr] Entering context manager")

    # Save original stderr and stdout file descriptors
    original_stderr_fd = sys.stderr.fileno()
    original_stdout_fd = sys.stdout.fileno()
    saved_stderr_fd = os.dup(original_stderr_fd)
    saved_stdout_fd = os.dup(original_stdout_fd)
    _tty_debug(f"[capture_cpp_stderr] Saved stderr fd: {saved_stderr_fd}, stdout fd: {saved_stdout_fd}")

    # Create temporary files to capture stderr and stdout
    with tempfile.NamedTemporaryFile(mode="w+", delete=True, suffix=".stderr") as tmp_err, \
         tempfile.NamedTemporaryFile(mode="w+", delete=True, suffix=".stdout") as tmp_out:
        try:
            _tty_debug(f"[capture_cpp_stderr] Created temp files: err={tmp_err.name}, out={tmp_out.name}")
            # Redirect stderr and stdout to temp files
            os.dup2(tmp_err.fileno(), original_stderr_fd)
            os.dup2(tmp_out.fileno(), original_stdout_fd)
            sys.stderr = os.fdopen(original_stderr_fd, "w", closefd=False)
            sys.stdout = os.fdopen(original_stdout_fd, "w", closefd=False)
            _tty_debug("[capture_cpp_stderr] Stderr and stdout redirected, yielding...")

            yield

        finally:
            _tty_debug("[capture_cpp_stderr] In finally block")
            # Flush stderr and stdout
            sys.stderr.flush()
            sys.stdout.flush()

            # Restore original stderr and stdout
            os.dup2(saved_stderr_fd, original_stderr_fd)
            os.dup2(saved_stdout_fd, original_stdout_fd)
            os.close(saved_stderr_fd)
            os.close(saved_stdout_fd)
            sys.stderr = sys.__stderr__
            sys.stdout = sys.__stdout__

            # Read captured content from both streams
            tmp_err.seek(0)
            tmp_out.seek(0)
            captured_stderr = tmp_err.read()
            captured_stdout = tmp_out.read()

            _tty_debug(f"[capture_cpp_stderr] Captured stderr: {len(captured_stderr)} chars")
            _tty_debug(f"[capture_cpp_stderr] Captured stdout: {len(captured_stdout)} chars")

            # Combine both for MLIR error detection
            combined = ""
            if captured_stderr:
                combined += captured_stderr
                _tty_debug(f"[capture_cpp_stderr] stderr preview: {captured_stderr[:500]}")
            if captured_stdout:
                combined += "\n" + captured_stdout if combined else captured_stdout
                _tty_debug(f"[capture_cpp_stderr] stdout preview: {captured_stdout[:500]}")

            # Check for MLIR errors in either stream
            if combined and "loc(" in combined and "error:" in combined:
                _tty_debug("[capture_cpp_stderr] MLIR ERROR FOUND!")
                # Find and print the error line
                for line in combined.split('\n'):
                    if 'error:' in line:
                        _tty_debug(f"[capture_cpp_stderr] Error line: {line[:200]}")
                        break

            # Store in thread-local storage
            set_captured_cpp_stderr(combined if combined else None)

if TYPE_CHECKING:
    # ComponentChecker is only imported for type checking to avoid circular imports
    from .checks_xla import ComponentChecker


@dataclass
class ExceptionData:
    class_name: str
    message: str
    error_log: str
    stdout: Optional[str] = None
    stderr: Optional[str] = None


MessageCheckerType = Callable[[str], bool]


class MessageChecker:
    """
    Class with helper methods to create message checker functions.
    Each method returns a function that takes an exception message as input
    and returns a boolean indicating whether the message matches the criteria.
    """

    @staticmethod
    def contains(message: str) -> bool:
        """Check if the message contains the given substring."""
        return lambda ex_message: message in ex_message

    @staticmethod
    def starts_with(message: str) -> bool:
        """Check if the message starts with the given substring."""
        return lambda ex_message: ex_message.startswith(message)

    @staticmethod
    def equals(message: str) -> bool:
        """Check if the message is equal to the given string."""
        return lambda ex_message: ex_message == message

    @staticmethod
    def regex(pattern: str) -> bool:
        """Check if the message matches the given regex pattern."""
        return lambda ex_message: re.search(pattern, ex_message) is not None

    @staticmethod
    def any(*checkers: MessageCheckerType) -> bool:
        """Check if any of the checkers match the message (or)."""
        return lambda ex_message: any(checker(ex_message) for checker in checkers)

    @staticmethod
    def neg(checker: MessageCheckerType) -> bool:
        """Negate the checker function (not)."""
        return lambda ex_message: not checker(ex_message)

    @staticmethod
    def last_line(checker: MessageCheckerType) -> str:
        """Apply the checker to the last line of the message."""
        return lambda ex_message: checker(
            ex_message.splitlines()[-1] if ex_message else ex_message
        )


# Short alias for MessageChecker used in failing reasons definitions
M = MessageChecker


@dataclass
class ExceptionCheck:
    """
    Class representing a set of checks to identify a specific exception.
    """

    class_name: Optional[str] = None
    component: Optional["ComponentChecker"] = None
    message: List[MessageCheckerType] = field(default_factory=list)
    error_log: List[MessageCheckerType] = field(default_factory=list)
    stdout: List[MessageCheckerType] = field(default_factory=list)
    stderr: List[MessageCheckerType] = field(default_factory=list)

    def __contains__(self, ex: ExceptionData) -> bool:
        """
        Check if the exception data matches this exception check via 'in' operator.
        """
        return self.check(ex)

    def check(self, ex: ExceptionData) -> bool:
        """
        Check if the exception data matches this exception check.

        Args:
            ex (ExceptionData): The exception data to check.

        Returns:
            bool: True if the exception data matches, False otherwise.
        """
        if self.class_name:
            if ex.class_name != self.class_name:
                return False
        if self.component is not None:
            if ex not in self.component:
                return False
        for message_check in self.message:
            if not message_check(ex.message):
                return False
        for message_check in self.error_log:
            if not message_check(ex.error_log):
                return False
        for message_check in self.stdout:
            if ex.stdout is None or not message_check(ex.stdout):
                return False
        for message_check in self.stderr:
            if ex.stderr is None or not message_check(ex.stderr):
                return False
        return True


@dataclass
class FailingReason:
    """
    Class representing a failing reason for a specific exception.
    It contains a description and a list of exception checks.
    """

    # Static class variable to be populated later to avoid circular import
    component_checker_none: ClassVar[Optional["ComponentChecker"]] = None

    description: str
    checks: List[ExceptionCheck] = field(default_factory=list)

    def __post_init__(self):
        self.checks = [
            check
            for check in self.checks
            if check.component is None
            or check.component != self.__class__.component_checker_none
        ]
        if len(self.checks) == 0:
            logger.trace(
                f"FailingReason '{self.description}' has no checks defined, it will not be used."
            )
        elif len(self.checks) > 1:
            logger.trace(
                f"FailingReason '{self.description}' has multiple ({len(self.checks)}) checks defined."
            )

    @property
    def component_checker(self) -> Optional["ComponentChecker"]:
        for check in self.checks:
            component = check.component
            if component is None or component == self.__class__.component_checker_none:
                continue
            return component
        return None

    @property
    def component_checker_description(self) -> Optional[str]:
        component_checker = self.component_checker
        return component_checker.description if component_checker else None

    def __contains__(self, ex: ExceptionData) -> bool:
        return self.check(ex)

    def check(self, ex: ExceptionData) -> bool:
        for check in self.checks:
            if ex in check:
                return True
        return False

    def __repr__(self) -> str:
        return f"FailingReason(description={self.description!r})"


class PyTestUtils:

    @classmethod
    def get_long_repr(cls, exc: Exception) -> str:
        """Get long representation of exception similar to pytest's longrepr."""
        long_repr = None
        if hasattr(exc, "__traceback__"):
            exc_info = (type(exc), exc, exc.__traceback__)
            long_repr = ExceptionInfo(exc_info).getrepr(style="long")
        else:
            long_repr = ExceptionInfo.from_exc_info().getrepr(style="long")
        long_repr = str(long_repr)
        return long_repr

    @classmethod
    def remove_colors(cls, text: str) -> str:
        # Remove colors from text
        text = re.sub(r"#x1B\[\d+m", "", text)
        text = re.sub(r"#x1B\[\d+;\d+;\d+m", "", text)
        text = re.sub(r"#x1B\[\d+;\d+;\d+;\d+;\d+m", "", text)

        text = re.sub(r"\[\d+m", "", text)
        text = re.sub(r"\[\d+;\d+;\d+m", "", text)
        text = re.sub(r"\[\d+;\d+;\d+;\d+;\d+m", "", text)

        text = re.sub(r"\[1A", "", text)
        text = re.sub(r"\[1B", "", text)
        text = re.sub(r"\[2K", "", text)

        return text
