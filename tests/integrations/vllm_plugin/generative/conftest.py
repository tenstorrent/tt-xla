# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared conftest for vLLM generative tests."""
import re

import psutil

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
_MIN_WORDS = 5


def assert_output_coherent(text: str) -> None:
    """Heuristic assertion: text is natural-language, not token soup.

    Uses English stopword ratio as the token-soup detector — coherent
    continuations contain several stopwords per ~30-token output, while
    token-soup garbage contains ~zero.
    """
    s = text.strip()
    assert s, f"empty output: {text!r}"
    words = [w.lower() for w in _WORD_RE.findall(s)]
    assert words, f"output has no word characters: {text!r}"
    if len(words) < _MIN_WORDS:
        return
    sr = sum(1 for w in words if w in _STOPWORDS) / len(words)
    assert (
        sr >= _MIN_STOPWORD_RATIO
    ), f"stopword ratio too low ({sr:.3f} < {_MIN_STOPWORD_RATIO}): {text!r}"


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
