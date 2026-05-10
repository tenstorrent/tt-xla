# Exit code 13 in tt-xla pytest runs

Exit 13 from `pytest` is the standard `EACCES` (permission denied) errno. In
practice on tt-xla machines it most often shows up as a *symptom* of one of
these underlying problems, not the cause itself:

- A previous test crashed and left the device in a held / locked state — the
  next run can't open `/dev/tenstorrent/0`.
- Another process is currently holding the device (zombie pytest, orphaned
  Python REPL, etc.).
- A driver hang surfaced as a permission failure when the runtime tried to
  re-acquire the device.
- Genuine permissions issue on `/dev/tenstorrent/*` (rare on a configured
  machine, but possible after a package upgrade or a fresh image).

## What to do when exit 13 shows up

The dump is still the source of truth — read `run.log` end-to-end via
`Read offset/limit` (never `tail`). The interesting bits are usually:

1. **The earliest TT_FATAL / TT_THROW / driver error** in the run, even if it
   appears far before the final EACCES. That's the original cause.
2. **Any "device busy", "could not open", "in use by another process"** lines.
3. **The full traceback** of the EACCES — sometimes it points directly at
   `open()` of a device path, sometimes at an mmap. The path tells you which
   device subsystem is stuck.

Then, in addition to the normal triage report:

- Note exit-13 explicitly in the report so the developer can decide whether to
  re-run the test on a clean machine before treating the extracted op as the
  real failing op.
- If the dump shows a device-held / driver-hang signature with no preceding
  TT_FATAL, this is **not** a single-op-repro situation. Switch the artifact to
  `triage-log.md` and recommend the developer reset the card / reboot and
  re-run before filing.

## What NOT to do

- Don't `chmod` `/dev/tenstorrent/*`. The skill never modifies device perms.
- Don't kill other processes on the box without explicit user confirmation.
- Don't re-run the test more than once automatically — if exit 13 persists,
  hand back to the developer.
