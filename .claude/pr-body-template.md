# tt-xla PR Body Template

Use this template when creating PRs. Fill in each section; remove sections that are not applicable.

---

```markdown
### Ticket
<!-- Link to GitHub Issue, Linear ticket, or "N/A" -->

### Problem description
<!-- What is broken or missing? Provide context for *why* this change is needed. -->

### What's changed
<!-- Approach used. Summary of changes and their impact.
     For multi-commit PRs, list what each logical group of commits does. -->

### Testing
<!-- How was this tested? Be specific about which test suites were run and on what hardware. -->

#### Test commands run
```bash
# e.g.
pytest -v tests/jax/single_chip
pytest -v tests/torch -m single_device
pre-commit run --all-files
```

#### Checklist
- [ ] New/existing tests provide coverage for changes
- [ ] `pre-commit run --all-files` passes (black, clang-format, isort, SPDX header check)
- [ ] SPDX copyright header added to all new source files (`// SPDX-License-Identifier: Apache-2.0`)
- [ ] CLAUDE.md updated if architecture, commands, or dev workflow changed
- [ ] Relevant CODEOWNERS areas notified (auto-assigned by GitHub)
- [ ] No debug prints, TODOs, or temporary hacks left in
```

---

## CODEOWNERS Quick Reference

| Path | Owners |
|---|---|
| `/.github/` | @vmilosevic @kmabeeTT @nsumrakTT @vvukomanTT |
| `/pjrt_implementation/` | @mrakitaTT @pilkicTT @nvukobratTT @acolicTT @ajakovljevicTT |
| `/tests/integrations/vllm_plugin/` | @AleksKnezevic @kmabeeTT @mmanzoorTT @ljovanovicTT |
| `/integrations/vllm_plugin/` | @AleksKnezevic @kmabeeTT @mmanzoorTT @ljovanovicTT |
| `/python_package/tt_torch/` | @nvukobratTT @dgolubovicTT @jameszianxuTT @AleksKnezevic |
| `/python_package/tt_jax/` | @mrakitaTT @ajakovljevicTT @sdjukicTT @sgligorijevicTT |
| `/tests/` | @mrakitaTT @ajakovljevicTT @AleksKnezevic @kmabeeTT |
| `/third_party/` | @mrakitaTT @pilkicTT @nvukobratTT @acolicTT |
| `/scripts/` | @rpavlovicTT @odjuricicTT @AleksKnezevic @mrakitaTT |
| `*` (global) | @mrakitaTT @nvukobratTT @AleksKnezevic |
