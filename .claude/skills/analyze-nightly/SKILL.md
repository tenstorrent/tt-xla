---
name: analyze-nightly
description: Analyze a GitHub Actions run and summarize failures
disable-model-invocation: true
allowed-tools: Read, Read(/tmp/**), Glob, Grep, Bash(git clone *), Bash(gh run view *), Bash(gh run list *), Bash(gh run download *), Bash(gh pr view *), Bash(tee *), Bash(gh api *), Bash(gh api * > /tmp/**), Bash(wc -l /tmp/**), Bash(jq *), Bash(mkdir -p /tmp/**), Bash(rm -rf /tmp/**), Bash(for *)
context: fork
argument-hint: run-id
model: opus
---

Create a summary of test failures or job failures, grouped by ownership area:
- Ownership area: PJRT unit tests.
    - PJRT tests are rooted under various directories in pjrt_implementation/.
- Ownership area: vLLM integration and multi-host execution.
    - vLLM integration tests are rooted in tests/integrations/vllm_plugin/.
    - vLLM integration is rooted in integrations/vllm_tt/.
    - Multi-host execution PyTorch tests are rooted in tests/torch/multi_host/.
- Ownership area: Performance benchmarks.
    - Performance benchmarks are rooted in tests/benchmark/.
- Ownership area: Examples.
    - Examples are rooted under examples/.
    - Test which runs examples is in tests/examples/test_examples.py.
- Ownership area: PyTorch and JAX single-chip and multi-chip tests.
    - This is a catch-all category for tests rooted under tests/.
Important note: "test" refers broadly to any test run, performance benchmark
run, tt-xla project demo run, or tt-xla project example run.

Analyze the run with run-id $0 and create a summary of test failures:
- Tests/jobs are run in the GitHub Actions workflow with run-id $0.
- GitHub repo associated with the workflow is github.com/tenstorrent/tt-xla.
- Read the workflow file .github/workflows/schedule-nightly.yml to build
  context around which jobs are executed in the workflow. Jobs mostly invoke
  other workflow *.yml files; read other worklow *.yml files that are
  referenced from .github/workflows/schedule-nightly.yml. Some .yml files
  may be in another repository, namely github.com/tenstorrent/tt-forge. To
  access files in other repositories, use git to clone them to /tmp/. Ensure
  that these cloned files are deleted after you complete executing all your
  other tasks.
- For the run with run-id $0 fetch all job-ids by using the `gh` CLI tool.
  Run `gh run view $0 --json jobs --jq '.jobs[].url'` to fetch all job URLs,
  which have the following format, from which you can extract {job-id}:
  https://github.com/tenstorrent/tt-xla/actions/runs/{run-id}/job/{job-id}
- Fetch details for every job by using the `gh api` subcommand. Discard any job
  that was successfully completed, canceled, skipped, or is still in progress.
  Focus only on jobs that failed!
- For each failed job, identify which step(s) failed. If multiple steps failed,
  focus on the first one, assuming the first one is the one that cause
  subsequent step failures. Fetch and read raw logs for the failed step by
  using the `gh` CLI tool. Analyze the logs in search for error messages,
  failure messages, timeout messages, or any other text indicating a root cause
  for the failure of that step of the job. Keywords to look for are error,
  assert, assertion, failed, throw, failure, fatal, timeout, timed out, HTTP
  error codes from 400 and 500 range, Linux exit codes corresponding to signals
  processes can receive, etc. Don't limit yourself to only these keywords, there
  may be others that indicate a root cause, these are just the most common ones.
- Strategies for how to improve log crawling: (a) Always use case insensitive
  pattern matching for specific words or phrases. (b) If the logs are truncated
  or too large, download them to /tmp/ in a temporary directory. Ensure that
  this temporary directory is deleted after you complete executing all your
  other tasks. (c) If a log exceeds 5000 lines, first try to use Grep to search
  for keywords rather than reading the full log. Only if you fail to identify
  a root cause with Grep, read the whole log. (d) if you identify a root cause,
  searching backward through the log will in most cases yield a line of text
  identifying the specific test that failed.
- Job steps (and therefore their logs) that are responsible for running tests
  are always running multiple tests, not just one. Tests are in most cases
  either a pytest command specifying a parametrized test, or model tests invoked
  by the tests/runner/test_models.py script with the model name as the parameter
  in brackets. The same test may run on different hardware architectures, but
  never in the same job, so it is possible for the same test to fail in multiple
  jobs with the same root cause.
- Gather information relevant to the failure: name of the test and/or model
  that failed (or job step name that failed if previous is not applicable),
  root cause, hardware architecture (if applicable). If the same test fails with
  the same root cause on multiple architectures, list it once with all affected
  architectures in the {arch-list}.
- Summarize which tests or job steps have failed. If a job step failed because
  a test failed, present it as a test failure. If a job step failed before any
  test is run, present it as a job failure unrelated to test execution. Group
  all failures by ownership area. If an ownership area has no failures, do not
  emit any text for that area in the output.

Output format that you need to follow (raw Markdown text):
```
# {ownership-area-name}

## {root-cause-1}
- {test-or-step-name} ({arch-list}) -> [job-link]({url})
- {test-or-step-name} ({arch-list}) -> [job-link]({url})

## {root-cause-2}
- {test-or-step-name} ({arch-list}) -> [job-link]({url})
- {test-or-step-name} ({arch-list}) -> [job-link]({url})

# {ownership-area-name}

## {root-cause-1}
- {test-or-step-name} ({arch-list}) -> [job-link]({url})
- {test-or-step-name} ({arch-list}) -> [job-link]({url})

## {root-cause-2}
- {test-or-step-name} ({arch-list}) -> [job-link]({url})
- {test-or-step-name} ({arch-list}) -> [job-link]({url})
```

Always respect these additional constraints:
- Never ask for, and never run any `gh {subcommand}` commands that may modify
  the state of the GitHub repository (for example issue creation/deletion,
  PR closing, branch manipulation etc.), especially when using the `gh api`
  subcommand. Always use only read-only calls!
- Never execute multiple commands separated by a semi-colon!
- Always use for loops in bash for executing commands for different job-ids!
  Never execute the same command separately for multiple job-ids!
