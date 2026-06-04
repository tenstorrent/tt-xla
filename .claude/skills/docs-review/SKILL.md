---
name: docs-review
description: Review and fix language correctness and formatting in documentation files.
disable-model-invocation: false
allowed-tools: Read, Grep, Write, Edit, Bash
context: fork
model: opus
---

## Parameters

- `--base <ref>` (optional): Scope review to documentation files on the current branch when compared to <ref>.
- `--all` (flag): Review the entire docs/ directory tree.
- `--no-edits` (flag): Don't fix anything; provide a list of editing suggestions as output.

## When to use

Use this skill whenever the focus is on the **written English** of documentation text,
not its technical accuracy / content. It targets prose found in Markdown (`.md`),
reStructuredText (`.rst`), and plain text (`.txt`) files found exclusively under
the docs/ directory tree.

Trigger this skill when the user:

- Asks to **review, proofread, audit, or check** documentation.
- Mentions **grammar, spelling, punctuation, tense, voice, tone, or style** in the context of docs.
- Wants documentation **cleaned up, polished, tightened, or rewritten** for clarity and consistency.
- Asks whether docs follow a **style guide** or are **consistent** in terminology, capitalization, or formatting.
- Wants to **review documentation changes before committing**, before opening a PR, or across a branch.

Do **not** use this skill for:

- Reviewing code or code comments.
- Verifying that documentation is **technically correct** (e.g., that a described API actually behaves as documented).
- Generating new documentation from scratch.

## Workflow

### 1. Determine review scope

Resolve the set of files to review based on the flags.
Only files under `docs/` with a `.md`, `.rst`, or `.txt` extension are eligible.
Anything outside that tree or with a different extension must be filtered out, even if it appears in a diff.

Apply the flags in this order of precedence:

1. `--all` -- list every eligible file under `docs/`:
   ```bash
   git ls-files 'docs/**/*.md' 'docs/**/*.rst' 'docs/**/*.txt'
   ```
   If `--all` is set, ignore `--base`.

2. `--base <ref>` -- list eligible files that differ between `<ref>` and `HEAD`:
   ```bash
   git diff --name-only <ref>...HEAD -- 'docs/**/*.md' 'docs/**/*.rst' 'docs/**/*.txt'
   ```
   Verify `<ref>` exists first (`git rev-parse --verify <ref>`); if it does not, stop and report the error.

3. Default (no flags) -- list eligible files with uncommitted changes (staged or unstaged) relative to `HEAD`:
   ```bash
   git diff --name-only HEAD -- 'docs/**/*.md' 'docs/**/*.rst' 'docs/**/*.txt'
   git ls-files --others --exclude-standard -- 'docs/**/*.md' 'docs/**/*.rst' 'docs/**/*.txt'
   ```
   Union both lists so newly added untracked files are included.

After resolution:

- If the resulting set is empty, stop and tell the user there is nothing to review (mention which scope was checked).
- Confirm the resolved file list to the user before reading content, so they can correct the scope if it is wrong.

### 2. Gather content to review

Load the content of every file in the resolved set. Use the `Read` tool,
not `cat` or shell pipes, so line numbers are preserved for later citation.

Two pieces of content matter for each file:

- **Full file text** -- always read, regardless of scope. The reviewer needs surrounding context
(the sentence before, the section heading, the surrounding list) to judge any single line.
Reviewing isolated diff lines produces shallow feedback and false positives (e.g., flagging an `ambiguous-pronoun` whose antecedent is in the previous sentence).
- **Changed regions** -- only in `--base` and default modes. Collect the diff so the review can prioritize lines the user actually touched:
  ```bash
  # --base mode
  git diff <ref>...HEAD -- <file>

  # Default mode (uncommitted, staged + unstaged)
  git diff HEAD -- <file>
  ```
  For untracked new files in default mode, treat the entire file as "changed."

When reading, identify and **exclude** the following regions from prose review -- they are not English text and must not be edited:

- Fenced code blocks (```` ``` ```` in Markdown, `.. code-block::` in RST, indented code blocks).
- Inline code spans (`` `like this` ``).
- URLs and link targets (the `https://...` part, not the link text).
- HTML comments (`<!-- ... -->`) unless they contain user-facing prose.
- RST directives and roles (`.. note::`, `` :ref:`...` ``), except for the prose inside them.
- YAML/TOML front matter at the top of Markdown files (treat as metadata, not prose).
- Tables -- review cell text, but do not reformat the table structure.

Keep a per-file map of `{line_number -> text}` for the reviewable lines only.
Findings in later steps must cite these line numbers so the user can locate them.

In `--all` mode, the review covers the full reviewable text of every file.
In `--base` and default modes, the review prioritizes lines inside changed hunks,
but may flag issues on adjacent unchanged lines when they are part of
the same sentence or paragraph as a changed line.

### 3. Apply review rules

For every reviewable line in the per-file map, evaluate it against the rules in the [Language rules](#language-rules) section,
and against the [Markdown rules](#markdown-rules) section when the file is a `.md` file.
For `.rst` and `.txt` files, only the language rules apply.

Each violation becomes a **finding**. A finding has the following fields:

- `file` -- path relative to the repository root.
- `line` -- 1-based line number from the per-file map. Use a range (e.g., `42-45`) when the violation spans multiple lines.
- `rule` -- the rule name from the rules section (e.g., `active-voice`, `serial-comma`, `dangling-modifier`).
   Use a stable, kebab-case identifier so findings can be grouped.
- `severity` -- one of:
  - `error` -- objective language violation: grammar, subject-verb agreement, punctuation errors, misspellings, broken Markdown syntax.
  - `warning` -- clarity or consistency issue: passive voice in a how-to step, inconsistent terminology, an undefined acronym.
     The text is understandable but worse than it could be.
  - `info` -- stylistic preference: a shorter synonym is available, a sentence could be tightened, a list could be parallel.
     Surface only when high confidence.
- `quote` -- the exact text being flagged (verbatim, including surrounding whitespace if relevant).
- `suggestion` -- the proposed replacement text. Required for `error` and `warning`. For `info`, include when concrete; omit when the fix is open-ended.
- `rationale` -- one short sentence explaining which rule was triggered and why. No more than one sentence per finding.

Rules for applying the rules:

- **Do not flag content inside excluded regions** (code blocks, inline code, URLs, RST directives, front matter).
  If a violation appears inside such a region, drop it.
- **One finding per rule violation per location.** If the same sentence violates two rules, emit two findings.
  If the same rule fires on ten consecutive lines for the same reason (e.g., trailing whitespace), collapse them into one finding with a line range.
- **Prefer the strictest rule that applies.** If both a grammar rule (`error`) and a style rule (`info`) apply to the same span,
  keep the grammar finding and drop the style one -- it will be moot once the rewrite happens.
- **Be conservative with `info`.** False positives erode trust faster than false negatives.
  If the rule's application is debatable in context, drop it.
- **Do not invent rules.** Every finding must point to a documented rule in the rules sections.
  If the prose feels wrong but no rule applies, do not emit a finding.
- **Quote accurately.** The `quote` field must match the file byte-for-byte (modulo trailing newlines),
  so the user can `grep` for it and so the fix step can find and replace it deterministically.

At the end of this step, the working state is a list of findings per file, ready to be reported and (unless `--no-edits`) applied.

### 4. Report findings

Present the findings to the user before any file is modified. The report is always emitted, regardless of whether `--no-edits` is set.
When fixes will be applied, it serves as a preview; when they will not, it is the deliverable.

Follow the structure defined in the [Output format](#output-format) section. In short:

- **Header** -- one line stating the scope that was reviewed (`--all`, `--base <ref>`, or `uncommitted`), the number of files reviewed,
  and the total finding count broken down by severity (e.g., `3 errors, 7 warnings, 2 info`).
- **Per-file sections** -- group findings by file. Within each file, order findings by severity (`error` -> `warning` -> `info`), then by line number. Files with zero findings are listed
  once under a "Clean files" line at the end of the report; do not give them their own section.
- **Per-finding entry** -- render every field from step 3 (`line`, `rule`, `severity`, `quote`, `suggestion`, `rationale`) in the format
  defined in [Output format](#output-format).
- **Summary** -- close with a list of findings per rule, sorted by frequency, so the user can spot systemic issues at a glance.

Rules for reporting:

- **Do not paraphrase the `quote`.** Render it verbatim. If it is long, truncate with ellipses,
  but keep enough on either side of the violation that the user can locate it.
- **Do not collapse `error` findings.** Each grammar or punctuation violation gets its own entry,
  even if the same rule fires twice in one file.
- **Do collapse repetitive `warning`/`info` findings** that share a rule and a fix pattern
  (e.g., the same passive construction used in three list items): one entry, with all line numbers cited.
- **Order severities top-to-bottom within a file**: `error` first, then `warning`, then `info`. This keeps high-impact issues above the fold.
- **Do not editorialize.** The report contains findings only -- no commentary on overall doc quality, no praise,
  no recommendations beyond the per-finding `suggestion`.

After the report is rendered:

- If `--no-edits` is set, emit the closing line specified in [Output format](#output-format) and **stop here**. The skill's job is done.
- Otherwise, proceed to step 5 without waiting for confirmation.
  The user can interrupt if the report surfaces something they want to handle differently;
  explicit confirmation prompts would be noise on every invocation.

### 5. Apply fixes (unless `--no-edits`)

This step is skipped entirely when `--no-edits` is set. Otherwise, walk the findings list and apply every fix
whose `suggestion` field is populated. Findings without a `suggestion` (open-ended `info` items) are reported
but not edited.

For each file with at least one applicable finding:

- Apply edits using the `Edit` tool. Use `quote` as the `old_string` and `suggestion` as the `new_string`.
  This guarantees byte-accurate replacement and a clear diff.
- Apply findings **in descending line order** within a file. Editing from the bottom up keeps line numbers
  in earlier findings valid even if a fix changes line count.
- If a `quote` is no longer unique in the file (because an earlier fix introduced a duplicate), expand it
  with one line of context above and below before retrying. Do not switch to `replace_all` -- a same-string
  violation elsewhere in the file may need a different fix.
- If an `Edit` call fails because the `quote` no longer matches the file (a previous fix overlapped this span),
  **skip the finding** and add it to a `deferred` list. Do not retry with a guessed replacement.

Cross-cutting rules:

- **Never edit excluded regions.** If a finding's line falls inside a region that step 2 marked as excluded,
  drop it before editing -- even if it somehow survived step 3. Defense in depth.
- **Never edit files outside the resolved set.** A fix that wants to touch a file step 1 did not select is a bug; drop it.
- **Do not amend, stage, or commit.** This skill leaves changes in the working tree. Staging, committing, or pushing is the user's call.
- **Do not reformat untouched prose.** The only changes to the file are the ones a finding justifies.
  Do not normalize line lengths, re-wrap paragraphs, or fix unrelated quirks while passing through.

After all files are processed, emit a short closing report:

- The number of fixes applied, grouped by file.
- The number of fixes deferred (with file and line), if any, and the reason (`quote not found after prior edit`).
- The number of `info` findings that were reported but not auto-applied because they had no concrete `suggestion`.

Tell the user to `git diff` to review the changes before committing.

## Language rules

The rules below are grouped by category. Each rule is the authoritative source for a `rule` identifier used in findings.
The kebab-case slug at the start of every bullet is the canonical `rule` value.

### Grammar and syntax

- `present-tense` -- Use the present tense to describe behavior. "The function returns X" not "The function will return X."
- `active-voice` -- Use active voice over passive. "The parser reads the file" not "The file is read by the parser."
- `second-person` -- Address the reader as "you." Avoid "we" unless referring to a team decision.
- `subject-verb-agreement` -- Maintain subject-verb agreement, especially with collective nouns ("the data are" vs. "the dataset is").
- `dangling-modifier` -- Place modifiers next to what they modify. "After installing the package, run the tests" -- not "After installing the package, the tests run."
- `parallel-structure` -- Use parallel structure in lists and series. All items should share grammatical form (all verbs, all nouns, all imperatives).

### Word choice

- `concrete-terms` -- Prefer concrete over abstract terms. "Returns a list of users" beats "Handles user retrieval."
- `consistent-terminology` -- Use consistent terminology. Do not alternate between "parameter," "argument," and "option" for the same concept.
- `define-jargon` -- Avoid jargon unless the audience is known to understand it; define it on first use otherwise.
- `filler-words` -- Eliminate filler words: "very," "really," "quite," "basically," "actually," "simply," "just" and similar.
- `condescending-adverbs` -- Avoid "easily," "simply," and "obviously" -- what is obvious to the writer may frustrate the reader.
- `rfc2119-keywords` -- Use "must," "should," and "may" with RFC 2119 precision when specifying requirements.
- `shorter-words` -- Prefer shorter words when meaning is preserved: "use" over "utilize," "help" over "facilitate."
- `latin-abbreviations` -- Avoid Latin abbreviations (e.g., i.e., etc.) in prose; use "for example," "that is," "and so on."

### Sentence construction

- `sentence-length` -- Keep sentences under ~25 words. Break long sentences at logical clauses.
- `one-idea-per-sentence` -- One idea per sentence. Compound ideas obscure meaning.
- `front-loaded-information` -- Front-load important information. Put the subject and main verb early.
- `nominalization` -- Avoid nominalizations. "Decide" beats "make a decision"; "configure" beats "perform configuration."
- `hedging` -- Cut hedging language: "perhaps," "might possibly," "in some cases may potentially."

### Punctuation

- `serial-comma` -- Use the serial (Oxford) comma for clarity in lists of three or more.
- `semicolons` -- Use semicolons sparingly. Prefer two sentences.
- `dash-usage` -- Use em-dashes for parenthetical asides, en-dashes for ranges, hyphens for compound modifiers.
- `compound-modifier-hyphen` -- Hyphenate compound modifiers before nouns: "command-line tool," "open-source library."
- `list-item-punctuation` -- End list items consistently -- either all with periods or all without, based on whether they are full sentences.

### Capitalization and formatting

- `heading-case` -- Use sentence case for headings unless a project style guide mandates title case; be consistent throughout.
- `proper-noun-case` -- Capitalize proper nouns correctly: JavaScript, GitHub, macOS, npm, iOS.
- `emphasis-capitalization` -- Do not capitalize for emphasis. Use italics or bold instead.
- `code-formatting` -- Use code formatting (backticks) for identifiers, file paths, commands, and literal values.
- `bold-italics-usage` -- Use bold for UI elements the reader interacts with; italics for introducing new terms.

### Structure and flow

- `bluf` -- Lead with the conclusion (BLUF -- bottom line up front). State what the reader needs before the details.
- `descriptive-headings` -- Use descriptive headings that summarize content, not "Introduction" or "Overview."
- `paragraph-unity` -- One paragraph per idea; break when the topic shifts.
- `list-vs-prose` -- Use lists for parallel items, prose for connected reasoning.
- `context-before-code` -- Provide context before code. Explain what and why before showing how.

### Clarity and precision

- `acronym-on-first-use` -- Define acronyms on first use: "Just-In-Time (JIT) compilation." Exception: universally known ones (API, URL, HTTP).
- `ambiguous-pronoun` -- Avoid ambiguous pronouns. If "it," "this," or "they" could refer to two antecedents, repeat the noun.
- `units-and-types` -- Specify units and types. "Timeout in milliseconds" not just "timeout."
- `state-assumptions` -- State assumptions and prerequisites before instructions.

### Tone

- `neutral-tone` -- Be neutral and professional. Avoid humor.
- `no-apology` -- Do not apologize for the software or anticipate user errors patronizingly.
- `avoid-idioms` -- Avoid culture-specific idioms and metaphors ("hit it out of the park," "low-hanging fruit") -- they confuse non-native speakers.
- `inclusive-language` -- Write inclusively. Prefer "allowlist/blocklist," "primary/replica," and the singular "they" over gendered defaults.

### Consistency

- `spelling-convention` -- Pick one spelling convention (US or UK English) and apply it throughout.
- `date-number-format` -- Standardize date and number formats. ISO 8601 (2026-05-15) avoids regional ambiguity.
- `glossary` -- Maintain a glossary or style guide for project-specific terms.
- `identifier-case-match` -- Match code identifiers exactly when referenced in prose, including case.

### Examples and code

- `show-output` -- Show output alongside commands when the result matters.
- `realistic-placeholders` -- Use realistic placeholder values (`alice@example.com`, not `foo@bar`).
- `inline-annotations` -- Annotate non-obvious lines with brief inline explanations, not multi-paragraph commentary.

## Markdown rules

These rules apply only to `.md` files. Each rule's kebab-case slug is its canonical `rule` value.

- `single-h1` -- Use exactly one H1 (`#`) per file. Subsequent sections start at H2.
- `heading-level-skip` -- Do not skip heading levels (H2 -> H4). Each level descends by one.
- `blank-line-around-heading` -- Surround every heading with a blank line above and below.
- `blank-line-around-block` -- Surround lists, fenced code blocks, and tables with a blank line above and below.
- `fenced-code-language` -- Fenced code blocks must declare a language tag (```` ```bash ````, ```` ```python ````). Use `text` for plain output.
- `consistent-list-marker` -- Within a file, use one bullet marker (`-`, `*`, or `+`). Do not mix.
- `link-text` -- Link text must describe the destination. Do not use "click here," "here," or a bare URL as link text.
- `bare-url` -- Wrap bare URLs in angle brackets (`<https://...>`) or proper Markdown links. Do not paste raw URLs in prose.
- `image-alt-text` -- Every image must have non-empty alt text: `![description](path)`.
- `trailing-whitespace` -- Remove trailing whitespace from every line, except deliberate Markdown hard line breaks (two trailing spaces).
- `final-newline` -- End the file with a single trailing newline.
- `consistent-emphasis-marker` -- Use one emphasis style per file: `*italic*` or `_italic_`, `**bold**` or `__bold__`. Do not mix.
- `nested-list-indent` -- Indent nested list items with two spaces (or four, consistently within a file). Do not mix indentation.
- `no-raw-html` -- Avoid raw HTML unless Markdown cannot express the structure (e.g., complex tables, `<details>`).
- `empty-link-or-image` -- Do not commit links or images with empty targets (`[text]()`, `![alt]()`).

## Output format

The review report is plain Markdown, emitted to the conversation (not written to a file).
It has four parts in this order: a one-line header, per-file sections, a clean-files line, and a rule-frequency summary.

### Header

```
docs-review -- scope: <scope> | <N> files reviewed | <E> errors, <W> warnings, <I> info
```

- `<scope>` is one of `--all`, `--base <ref>`, or `uncommitted`.
- `<N>` is the count of files that were read (clean + flagged).
- `<E>`, `<W>`, `<I>` are total finding counts by severity. Omit a category with zero count (e.g., `3 errors, 2 info`).

### Per-file sections

One H3 per file, in alphabetical path order. Skip files with no findings -- they go on the clean-files line.

```
### <path/to/file.md>

- [<severity>] L<line> | `<rule>` -- <rationale>
  > <quote>
  -> <suggestion>
```

Field rendering:

- `<severity>` is one of `error`, `warning`, `info`. Render in lowercase.
- `L<line>` is the line number, or a range `L<start>-<end>` for multi-line findings.
- `` `<rule>` `` is the kebab-case rule slug in backticks.
- `> <quote>` is the verbatim offending text on its own line, prefixed with `>`.
    - Multi-line quotes use `>` on each line.
    - Truncate with `...` when longer than ~120 characters, keeping the violating span centered.
- `-> <suggestion>` is the proposed replacement on its own line.
    - Omit the entire `->` line when the finding has no `suggestion` (open-ended `info` items only).

Within a file, sort findings by severity (`error` -> `warning` -> `info`), then by line number ascending.

### Clean files

If any reviewed files had zero findings, append one line after the per-file sections:

```
Clean: <count> file(s) -- <comma-separated paths>
```

If the path list is longer than ~10 entries, show the first 10 and append `... (+<N> more)`.

### Rule-frequency summary

Close with an H3 summary, sorted by count descending, then alphabetically by rule for ties:

```
### Summary

- `<rule>` x <count>
- `<rule>` x <count>
```

Include every rule that fired at least once. This is the user's at-a-glance view of systemic issues.

### Closing line under `--no-edits`

When `--no-edits` is set, end with:

```
No edits applied (--no-edits). Findings above are the deliverable.
```

### Closing report when fixes were applied

When step 5 ran, emit the closing report described in that step *after* the rule-frequency summary,
separated by a blank line. Format:

```
### Applied

- <path>: <N> fix(es)
- ...

Deferred (<N>):
- <path>:L<line> -- <reason>
- ...

Reported only (no suggestion): <N>

Run `git diff` to review the changes before committing.
```

Omit the `Deferred` block when zero. Omit the `Reported only` line when zero.
