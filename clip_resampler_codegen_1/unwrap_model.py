#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Unwrap model_ttnn.py by inlining all method calls into the forward method.
"""

import re
from collections import OrderedDict


def parse_methods(content):
    """Parse all method definitions from the file, handling multi-line signatures."""
    methods = OrderedDict()

    # Find all method start positions
    method_starts = list(re.finditer(r"^    def (\w+)\(", content, re.MULTILINE))

    for i, match in enumerate(method_starts):
        method_name = match.group(1)
        start_pos = match.start()

        # Find the closing paren and colon of the signature
        sig_start = match.end() - 1  # Position of opening paren
        paren_depth = 1
        pos = sig_start + 1

        while pos < len(content) and paren_depth > 0:
            if content[pos] == "(":
                paren_depth += 1
            elif content[pos] == ")":
                paren_depth -= 1
            pos += 1

        # pos is now after the closing paren, find the colon
        while pos < len(content) and content[pos] != ":":
            pos += 1
        colon_pos = pos

        # Extract the parameters string (between opening paren and closing paren)
        params_str = content[sig_start + 1 : pos - 1]  # -1 to exclude closing paren

        # Remove 'self' and parse remaining params
        params = []
        # Split by comma, handling newlines
        for p in params_str.split(","):
            p = p.strip()
            if p and p != "self":
                params.append(p)

        # Find the end of this method (start of next method or end of class/file)
        if i + 1 < len(method_starts):
            end_pos = method_starts[i + 1].start()
        else:
            # Find end of class or file
            class_match = re.search(r"\nclass ", content[colon_pos:])
            if class_match:
                end_pos = colon_pos + class_match.start()
            else:
                end_pos = len(content)

        # Extract body (everything after the colon until end)
        body = content[colon_pos + 1 : end_pos]

        methods[method_name] = {
            "params": params,
            "body": body,
        }

    return methods


def parse_args(args_str):
    """Parse comma-separated arguments, handling nested parentheses."""
    args = []
    if not args_str.strip():
        return args

    depth = 0
    current_arg = []
    for char in args_str:
        if char in "([{":
            depth += 1
            current_arg.append(char)
        elif char in ")]}":
            depth -= 1
            current_arg.append(char)
        elif char == "," and depth == 0:
            arg = "".join(current_arg).strip()
            if arg:
                args.append(arg)
            current_arg = []
        else:
            current_arg.append(char)

    arg = "".join(current_arg).strip()
    if arg:
        args.append(arg)

    return args


def substitute_params(body, params, args):
    """Substitute parameter names with actual arguments."""
    result = body
    for i, param in enumerate(params):
        if i < len(args):
            arg = args[i]
            result = re.sub(rf"\b{re.escape(param)}\b", arg, result)
    return result


def inline_method_body(
    method_body, params, args, assignment_vars, var_prefix, base_indent
):
    """Inline a method body with parameter substitution and variable renaming."""
    # Substitute parameters
    inlined = substitute_params(method_body, params, args)

    # Find local variable assignments (at start of line, space before =)
    # Distinguish from keyword args (no space before =)
    local_vars = set()
    for match in re.finditer(r"^\s+(\w+)\s+=", inlined, re.MULTILINE):
        var_name = match.group(1)
        if var_name not in params:
            local_vars.add(var_name)

    # Rename local variables, but only when they appear as standalone words
    # not as keyword argument names (keyword args have no space before '=')
    for var in sorted(local_vars, key=len, reverse=True):
        new_name = f"{var_prefix}_{var}"
        # Replace the variable name, but NOT when it's a keyword arg (directly followed by '=', no space)
        # Keyword args: name=value (no space)
        # Assignments: name = value (space before =)
        # Use negative lookahead (?!=) to avoid renaming keyword arguments
        inlined = re.sub(rf"(?<![.\w])\b{var}\b(?!=)", new_name, inlined)

    # Process lines
    output_lines = []
    lines = inlined.split("\n")

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Check for return statement
        return_match = re.match(r"return\s*(.*)$", stripped)
        if return_match:
            return_expr = return_match.group(1).strip()
            if return_expr and assignment_vars:
                # Return with value - assign to the target vars
                output_lines.append(f"{base_indent}{assignment_vars} = {return_expr}")
            elif return_expr:
                # Return with value but no assignment - keep as return
                output_lines.append(f"{base_indent}return {return_expr}")
            # else: bare 'return' - skip it (don't emit anything)
        else:
            output_lines.append(f"{base_indent}{stripped}")

    return output_lines


def find_matching_paren(text, start=0):
    """Find the position of the matching closing parenthesis."""
    depth = 0
    for i, char in enumerate(text[start:], start):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def unwrap_forward(content, methods):
    """Unwrap all method calls in forward into inline ttnn ops."""

    forward_info = methods["forward"]
    forward_body = forward_info["body"]

    output_lines = []
    output_lines.append("    def forward(self, pixel_values):")

    call_counter = 0

    lines_buffer = forward_body.split("\n")
    idx = 0

    # Track pending assignments across lines (for wrapped assignments like "x = (\n  self.method(...)")
    pending_assignment = None
    pending_indent = None

    while idx < len(lines_buffer):
        line = lines_buffer[idx]
        stripped = line.strip()

        if not stripped:
            idx += 1
            continue

        indent_match = re.match(r"^(\s*)", line)
        base_indent = indent_match.group(1) if indent_match else "        "

        # Check for assignment start without method call (e.g., "x = (" on its own line)
        assign_only_match = re.match(r"^(\s*)([\w,\s]+)\s*=\s*\(\s*$", line)
        if assign_only_match:
            pending_assignment = assign_only_match.group(2).strip()
            pending_indent = assign_only_match.group(1)
            idx += 1
            continue

        # Check for method call
        # Pattern: optional "vars = " then "self.method(" with optional leading "("
        call_match = re.match(r"^(\s*)((?:[\w,\s]+)\s*=\s*)?\(?self\.(\w+)\((.*)", line)

        if call_match:
            assignment = call_match.group(2)
            method_name = call_match.group(3)
            args_part = call_match.group(4)

            # Use pending assignment if we have one and this line has no assignment
            had_pending = False
            if pending_assignment and not assignment:
                assignment = pending_assignment + " = "
                base_indent = pending_indent if pending_indent else base_indent
                had_pending = True
                pending_assignment = None
                pending_indent = None
            else:
                pending_assignment = None
                pending_indent = None

            # Count parens to see if call spans multiple lines
            paren_depth = 1
            for c in args_part:
                if c == "(":
                    paren_depth += 1
                elif c == ")":
                    paren_depth -= 1

            full_args = args_part

            # Read more lines if needed
            while paren_depth > 0 and idx + 1 < len(lines_buffer):
                idx += 1
                next_line = lines_buffer[idx]
                full_args += "\n" + next_line
                for c in next_line:
                    if c == "(":
                        paren_depth += 1
                    elif c == ")":
                        paren_depth -= 1

            # If we had a pending assignment (wrapped like "x = (\n  self.method(...)\n)"),
            # skip the next line if it's just a closing paren
            if had_pending and idx + 1 < len(lines_buffer):
                next_stripped = lines_buffer[idx + 1].strip()
                if next_stripped == ")":
                    idx += 1  # Skip the wrapper's closing paren

            # Strip trailing content after the call closes
            # Count parens to find where the call ends, then strip
            full_args = full_args.strip()
            # Remove trailing parens that close the call (and any wrapper)
            while full_args.endswith(")"):
                full_args = full_args[:-1]
            full_args = full_args.strip()

            # Parse arguments
            args = parse_args(full_args)

            # Get assignment vars
            assignment_vars = None
            if assignment:
                assignment_vars = assignment.strip().rstrip("=").strip()

            # Check if we should inline this method
            if method_name in methods and method_name not in ("forward", "__init__"):
                method_info = methods[method_name]

                var_prefix = f"v{call_counter}"
                call_counter += 1

                # Add comment
                output_lines.append(f"{base_indent}# --- {method_name} ---")

                # Inline
                inlined = inline_method_body(
                    method_info["body"],
                    method_info["params"],
                    args,
                    assignment_vars,
                    var_prefix,
                    base_indent,
                )
                output_lines.extend(inlined)
            else:
                # Keep original
                output_lines.append(line)
        else:
            # Not a method call - check if it's a closing paren for a wrapped assignment
            if stripped == ")" and pending_assignment:
                # Skip the lone closing paren - it was for the wrapper
                pending_assignment = None
                pending_indent = None
            else:
                output_lines.append(line)
                pending_assignment = None
                pending_indent = None

        idx += 1

    return "\n".join(output_lines)


def generate_unwrapped_model():
    """Generate the unwrapped model file."""
    with open("model_ttnn.py", "r") as f:
        content = f.read()

    methods = parse_methods(content)

    print(f"Found {len(methods)} methods")

    output = []
    output.append("# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC")
    output.append("#")
    output.append("# SPDX-License-Identifier: Apache-2.0")
    output.append('"""TTNN model with all ops inlined into forward()."""')
    output.append("")
    output.append("import ttnn")
    output.append("import utils")
    output.append("from consteval import run_const_evals")
    output.append("from models.common.lightweightmodule import LightweightModule")
    output.append("")
    output.append("")
    output.append("class CLIPVisionEncoderAndResamplerTTNN(LightweightModule):")
    output.append("    def __init__(self, weights, cache, device):")
    output.append("        self.device = device")
    output.append("        self.weights = weights")
    output.append("        self.cer = run_const_evals(weights, cache)")
    output.append("")

    unwrapped = unwrap_forward(content, methods)
    output.append(unwrapped)

    return "\n".join(output)


if __name__ == "__main__":
    result = generate_unwrapped_model()

    with open("model_ttnn_unwrapped.py", "w") as f:
        f.write(result)

    print(f"\nGenerated model_ttnn_unwrapped.py ({len(result)} chars)")
    print(f"Lines: {result.count(chr(10)) + 1}")

    import py_compile

    try:
        py_compile.compile("model_ttnn_unwrapped.py", doraise=True)
        print("Syntax OK!")
    except py_compile.PyCompileError as e:
        print(f"Syntax error: {e}")
