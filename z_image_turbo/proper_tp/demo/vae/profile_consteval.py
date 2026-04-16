#!/usr/bin/env python3
"""Profile the consteval warmup — time each main_const_eval_N call individually."""

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import utils
import main as generated_main

print("Loading static inputs ...")
orig = os.getcwd()
os.chdir(_HERE)
try:
    inputs = generated_main.load_inputs_for__main()
finally:
    os.chdir(orig)
print(f"Loaded {len(inputs)} inputs")

# Patch: wrap every main_const_eval_N to time it
import types

timings = {}
for name in dir(generated_main):
    if not name.startswith("main_const_eval_"):
        continue
    fn = getattr(generated_main, name)
    if not callable(fn):
        continue
    def make_timed(n, f):
        def timed(*args, **kwargs):
            t0 = time.perf_counter()
            result = f(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            timings[n] = elapsed
            return result
        return timed
    setattr(generated_main, name, make_timed(name, fn))

print("\nRunning consteval (timing each function) ...")
t_total = time.perf_counter()
# Manually call consteval__main (clears + repopulates cache)
generated_main._cached__main = {}
generated_main.consteval__main(generated_main._cached__main, inputs)
total_s = time.perf_counter() - t_total
print(f"Total consteval time: {total_s:.1f} s\n")

# Sort by time
sorted_t = sorted(timings.items(), key=lambda x: -x[1])
print(f"{'Function':<35} {'Time (s)':>10}")
print("-" * 47)
for name, t in sorted_t[:30]:
    print(f"  {name:<33} {t:>10.3f}")

print(f"\n  ... ({len(timings)} functions total)")
cumulative = sum(v for _, v in sorted_t[:10])
print(f"\nTop-10 functions account for {cumulative:.1f} s of {total_s:.1f} s total")
