---
name: analyze-nightly
description: Analyze and summarize failures from a GitHub Actions workflow
disable-model-invocation: true
allowed-tools: Bash(gh run view *), Bash(gh run list *), Bash(gh pr view *), Read, Glob, Grep, Bash(tee *), Read(/tmp/**), Bash(gh api *), Bash(wc *), Bash(jq *)
context: fork
argument-hint: [run-id]
model: opus
---

Create a summary of test/job failures, loosely grouped by type of failure:

**Note**: "Test" refers broadly to any test run, (perf) benchmark run,
tt-xla project demo run, and tt-xla project example run.

1. Tests/jobs are run in the GitHub Actions workflow with run-id $0.
2. GitHub repo associated with the workflow is github.com/tenstorrent/tt-xla.
3. Analyze the workflow file .github/workflows/schedule-nightly.yml to build
   context around which jobs are run as part of the workflow. Jobs mostly use
   other workflow .yml files, so if you need details of each job, analyze those
   files. Some .yml files may be in another repository, namely
   github.com/tenstorrent/tt-forge.
4. For the run with run-id $0, fetch all job-ids by using the appropriate
   `gh {subcommand}` call. Since the command for fetching job-ids is non-obvious
   here is a command that return all job URLs:
   `gh run view $0 --json jobs --jq '.jobs[].url'`. Each URL has the following
   format:
   https://github.com/tenstorrent/tt-xla/actions/runs/{run-id}/job/{job-id}.
5. Fetch job details by using the appropriate `gh api` call. For the rest of
   these instructions, focus only on the jobs that failed. Disregard others,
   and whenever a job is mentioned in the following instructions, note that
   instructions only apply for those filtered failed jobs. Always disregard
   successful, canceled, and skipped jobs.
6. For each job, identify which steps of the jobs have failed. If multiple steps
   have failed, focus on the first one, assuming the first one is the one that
   cause subsequent steps to fail as well. Fetch logs for the failed step.
7. Analyze each fetched log, analyze the text in search of error messages,
   failure messages, timeout messages etc. Keywords to look for are error,
   assert, assertion, failed, failure, fatal, timeout, timed out, and similar
   (case insensitive). There may be others as well, do not rely only on
   these keywords.
8. Job steps (and therefore their logs) that are responsible for running tests
    are actually responsible for running multiple tests, so also
    analyze the text for test beginning markers so you can associate the error
    message with the name of the tests. Tests are either simple pytest-invoked
    scripts or AI model tests invoked by tests/runner/test_models.py script.
    Each test may run on different hardware architectures, so it is possible for
    the test to fail in multiple jobs with the same root cause.
9. Summarize which tests/steps failed, loosely grouped by root cause for failure.
    If no steps were run because one of the required setup steps failed before the
    step responsible for running tests had a chance to run, then include the
    step failure in the summary instead of test failure. The generated output
    must be a text bullet list, where a top level bullet is a verbatim failure
    message or similar (e.g. timeout), and the sub-bullets must be a list of
    tests and/or job steps that failed with the given message.

Always respect these additional constraints:

- Never ask for, and never run any `gh {subcommand}` calls that may modify the
  state of the GitHub repository (for example issue creation/deletion,
  PR closing, branch manipulation etc.), especially `gh api` calls.
  Always use only read-only calls.
