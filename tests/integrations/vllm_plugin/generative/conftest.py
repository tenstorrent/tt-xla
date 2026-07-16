# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared conftest for vLLM generative tests."""
import json
import math
import re
import signal
from pathlib import Path

import psutil
import pytest

TEST_TIMEOUT_FALLBACK_SECONDS = 60 * 60


def _load_test_durations() -> dict[str, float]:
    """Load per-test durations from the repository .test_durations file."""
    durations_file = Path(__file__).resolve().parents[4] / ".test_durations"
    if not durations_file.exists():
        return {}

    try:
        data = json.loads(durations_file.read_text())
    except json.JSONDecodeError:
        return {}

    if not isinstance(data, dict):
        return {}

    return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}


_TEST_DURATIONS = _load_test_durations()


def _get_timeout_seconds(nodeid: str) -> int:
    """Return timeout as 2x recorded duration for this test."""
    recorded_seconds = _TEST_DURATIONS.get(nodeid)
    if recorded_seconds is None:
        return TEST_TIMEOUT_FALLBACK_SECONDS
    return max(1, int(math.ceil(recorded_seconds * 2)))


# Common English function words used by `assert_output_coherent` to detect
# the 2D-mesh sampler garbage-output bug (issue #4440). Coherent natural-
# language continuations contain several of these per ~30-token output;
# token-soup garbage contains ~zero.
_STOPWORDS = frozenset(
    """
    the a an and or but i you he she it we they is are was were be been
    have has had do does did of to in on at for with by from as that this
    my your her his their can will would should like go get make me so
    not if when what how there here
    """.split()
)
_WORD_RE = re.compile(r"[A-Za-z']+")
_MIN_STOPWORD_RATIO = 0.10
_MIN_STOPWORD_COUNT = 2
_MIN_WORDS = 10


def assert_output_coherent(text: str) -> None:
    """Heuristic assertion: text is natural-language, not token soup.

    Uses English stopword count/ratio as the token-soup detector — coherent
    continuations contain several stopwords per ~30-token output, while
    token-soup garbage contains ~zero. Requires both a minimum word count
    (too few word-like fragments is itself suspicious, not a reason to skip
    the check) and a minimum absolute stopword count in addition to the
    ratio — with only a handful of words, a single coincidental stopword
    match (e.g. a 3-letter fragment that happens to spell "and") trivially
    clears a ratio-only threshold regardless of whether the surrounding text
    means anything.
    """
    s = text.strip()
    assert s, f"empty output: {text!r}"
    words = [w.lower() for w in _WORD_RE.findall(s)]
    assert (
        len(words) >= _MIN_WORDS
    ), f"too few word-like tokens ({len(words)} < {_MIN_WORDS}), likely garbage: {text!r}"
    stopword_count = sum(1 for w in words if w in _STOPWORDS)
    sr = stopword_count / len(words)
    assert stopword_count >= _MIN_STOPWORD_COUNT and sr >= _MIN_STOPWORD_RATIO, (
        f"stopword signal too weak (count={stopword_count} < {_MIN_STOPWORD_COUNT} or "
        f"ratio={sr:.3f} < {_MIN_STOPWORD_RATIO}): {text!r}"
    )


# Unambiguous greedy prompt->answer checks. Batched, they force >1 seq/device
# (where TP/DP+TP prefill corrupts a slot); answers are landslides so a corrupted
# slot fails but benign near-tie TP fp drift (#5520) does not.
GROUNDED_BATCH_CHECKS = [
    ("1 + 1 =", "2"),
    ("The opposite of up is", "down"),
    ("Roses are red, violets are", "blue"),
    ("To be or not to be, that is the", "question"),
]


def assert_batch_grounded(outputs, checks=GROUNDED_BATCH_CHECKS) -> None:
    """Greedy wide-batch correctness: each output contains its grounded answer
    and isn't a degenerate repeat-loop. Detects per-slot prefill corruption
    (garbage or 'answer-then-loop'); tolerant of the #5520 near-tie fp drift
    since the answers are unambiguous."""
    assert len(outputs) == len(
        checks
    ), f"expected {len(checks)} outputs, got {len(outputs)}"
    for (prompt, expected), out in zip(checks, outputs):
        text = out.outputs[0].text
        token_ids = out.outputs[0].token_ids
        # Repeat-loop guard: degenerate loops (e.g. "down, up, down, up") sit far
        # below a coherent continuation's unique-token ratio (~0.8).
        uniq_ratio = len(set(token_ids)) / max(len(token_ids), 1)
        assert uniq_ratio >= 0.5, (
            f"degenerate/repetitive output for {prompt!r} "
            f"(unique ratio {uniq_ratio:.2f}): {text!r}"
        )
        assert (
            expected.lower() in text.lower()
        ), f"expected {expected!r} for {prompt!r}, got {text!r}"


def check_host_memory(model_name: str) -> float:
    """Assert child process RSS is below the known threshold for a model.

    Inspired by https://github.com/tenstorrent/tt-xla/issues/3611 where
    a vllm upgrade caused a ~3x host memory regression during compilation.

    Measures the current RSS of child processes (e.g. vLLM EngineCore)
    while they are still running. Call this after generation completes
    but before the engine is torn down.

    Returns the max child process RSS in GB.
    """
    # Known-good baselines with ~50% headroom.
    # Update these when adding new models or if baselines shift.
    model_rss_limits_gb = {
        "Qwen/Qwen3-0.6B": 5,
        "Qwen/Qwen3-32B": 150,
        # 26B-A4B measured ~57 GB host RSS; ~50% headroom.
        "google/gemma-4-26B-A4B-it": 85,
    }

    # Measures max RSS across child processes; assumes one engine at a time.
    children = psutil.Process().children(recursive=True)
    rss_gb = max((c.memory_info().rss for c in children), default=0) / 1024**3
    threshold = model_rss_limits_gb.get(model_name)

    limit_str = f"{threshold} GB" if threshold is not None else "none set"
    print(f"[MEM] {model_name}: max child RSS = {rss_gb:.1f} GB (limit: {limit_str})")

    if threshold is not None:
        assert rss_gb < threshold, (
            f"Max child RSS {rss_gb:.1f} GB exceeds {threshold} GB "
            f"for {model_name} — possible host memory regression"
        )

    return rss_gb


@pytest.fixture(autouse=True)
def _test_timeout(request):
    """Kill any test that hangs longer than 2x its recorded duration."""

    timeout_seconds = _get_timeout_seconds(request.node.nodeid)

    def _handler(_signum, _frame):
        raise TimeoutError(f"Test {request.node.nodeid} exceeded {timeout_seconds}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
