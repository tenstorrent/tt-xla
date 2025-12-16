# Breaking Into the Source With a Debugger

This page explains how to break into the native source code of the PJRT plugin.

## Prerequisites

- Clone and build the TT-XLA project.
    - The build has to be of the `Debug` type, e.g. `-DCMAKE_BUILD_TYPE=Debug`.
    - This is needed for native binaries to have debug symbols.
- Verify `gdb` is installed by running `gdb --version`.
    - Needed for debugging of native code.
- This guide is scoped to Visual Studio Code only.
- Install "C/C++" (Microsoft) and "Python" (Microsoft) VS Code extensions.
    - "Python" will auto-install the "Python Debugger" extension as well.
    - "Python Debugger" extension enables `debugpy` debugging.

## Step 1. Create an empty `launch.json` file

1. In the repository root, create a `.vscode/` directory.
    - Note: This directory is not tracked by `git`.
2. Create a new file `.vscode/launch.json` with the following JSON content:

```json
{
  "version": "0.2.0",
  "configurations": []
}
```

This file is used for configuring multiple debugging profiles.

## Step 2. Run a Python script / test in `debugpy`

Create a new debugging profile called `Python: Current File` in `launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    { // Python: Current File
        "name": "Python: Current File",
        "type": "debugpy",
        "request": "launch",
        "program": "${file}",
        "console": "integratedTerminal",
        "justMyCode": false
    }
  ]
}
```

Verify that this profile works:

1. Create a new Python script and set a breakpoint in VS Code.
2. Run a VS Code command `Debug: Select and Start Debugging` and select the
   `Python: Current File` profile while the Python script tab is open.
3. Validate that the breakpoint will be hit.

Now, replace the `Python: Current File` with a new profile for running tests,
`PyTest: Current File`:

```json
{
  "version": "0.2.0",
  "configurations": [
    { // PyTest: Current File
      "name": "PyTest: Current File",
      "type": "debugpy",
      "request": "launch",
      "module": "pytest",
      "args": [
        "-s",
        "${file}"
      ],
      "console": "integratedTerminal",
      "justMyCode": false
    }
  ]
}
```

Verify that this profile works:

1. Make sure `venv` is activated and `git` submodules are initialized.
2. Open a Python test from the `tests/` directory and set a breakpoint.
3. Run the new `PyTest: Current File` profile and validate that the breakpoint
   will be hit.

## Step 3. Attach `gdb` to a running PJRT client

Since running Python tests is the most common way to also test the PJRT plugin,
and because it is common to debug Python and native code side-by-side, this
guide will focus on that scenario. However, this step can be applied to
any running process, assuming you have the time to attach the debugger to your
process before it exits.

First, create a new debugging profile `Native: Attach to PJRT Client`
in `launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    { // PyTest: Current File (from Step 2)
    },
    { // Native: Attach to PJRT Client
      "name": "Native: Attach to PJRT Client",
      "type": "cppdbg",
      "request": "attach",
      "program": "${workspaceFolder}/venv/bin/python",
      "processId": "${command:pickProcess}",
      "MIMode": "gdb",
      // pjrt_plugin_tt.so is in this location
      "additionalSOLibSearchPath": "${workspaceFolder}/build/pjrt_implementation/src"
    },
  ]
}
```

Verify that this profile works:

1. Make sure `venv` is activated and `git` submodules are initialized.
2. Select a Python test to run from the `tests/` directory and set a breakpoint
   at the beginning of the test.
3. Run the `PyTest: Current File` debugging profile and wait for `debugpy` to
   break into the Python code.
    - At this point, the process running the test is stalled, and you have time
    to attach `gdb` to the process.
4. Open a C++ file that you wish to debug, and put a breakpoint where you
   wish to break. For exercise, almost all tests should pass through
   `ClientInstance::initialize`.
5. Run the `Native: Attach to PJRT Client` debugging profile without stopping
   the existing `PyTest: Current File` profile (that would kill the test driver
   process), which will prompt you to select which process you wish to attach.
   Select the `pytest` process that is running your test. Note that
   when you are in a remote SSH workspace session you will see
   multiple options, and you need to pick the right one (the server).
6. Resume execution of the `PyTest: Current File` profile to unblock the Python
   interpreter, and wait for the breakpoint in C++ code to be hit in the
   `Native: Attach to PJRT Client` debugger session.
7. Once the breakpoint is hit, you can debug the native PJRT code.
