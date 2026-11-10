# Review round 2, display delta

Reviewer: luna max child, read-only, brief at the phase 2 display amendment.
Scope: `spec.md` S10 and D12, the config rows and testing decisions, `skeleton.md` R12, modules M11 to
M14, contract C9, `tickets/T8-display-names.md`, `tickets/T7-readme.md` item 11.
Verdict as returned: `VERDICT: CHANGES`, nine findings plus one statement that feature behaviour could
not be run because no code exists yet.

## Dispositions

1. `Config` contract omitted `show_suggestions` and `suggestions_path`. Accepted. C2 now carries
   `show_suggestions`, `suggestions_path` and `max_display_names`, with `1 <= max_display_names <= 50`
   in its invariants, and the spec table defines all three.
2. The TUI half had no stated way to get a custom path or learn the disabled state, and M11 was called
   the only reader while the widget reads the file itself. Accepted. The record writer enforces
   `maxDisplayNames`, so the widget renders what the record holds. `showSuggestions` false now removes
   the record, which is the disabled state for both clients. The widget takes the path as
   `/skills-routed <path>`, defaults to the `suggestionsPath` default, never reads plugin config, and
   M11 is described as the only Python reader and writer.
3. The desktop half needs two independent enable steps. Accepted. T8 and T7 now name the profile
   `plugins.enabled` entry and the desktop Capabilities toggle, and the pane has to say which one is
   off. The live check turns both on and reads one routed turn.
4. The validate claim overstated the desktop lint. Accepted. The spec and T8 now call it a static
   admission lint that refuses selected constructs rather than an SDK allowlist, and T8 requires the
   file to be read against the SDK docs as well.
5. The zero-token test could pass with display data in both payload variants. Accepted. The spec and T8
   now require the enabled rewrite to be deep-equal to the same rewrite with the record writer replaced
   by a no-op, message count, roles and top-level keys included, and no record field in any message.
6. "Short" had no display bound. Accepted. `maxDisplayNames` defaults to 12 and caps what the record
   carries, so both surfaces list at most 12 names with the count in the collapsed title, and the test
   drives 50 injected names.
7. The TUI file-read uncertainty was stale. Accepted. U4 is removed; T8 records the mechanism that
   settles it: user widget modules are imported in the TUI process and run with its privileges, so
   `node:fs` reads work. The live TUI check remains the proof.
8. The record scope was ambiguous. Accepted. D12 now states that the newest write wins globally, the
   record carries the session and turn ids, the display says which turn it came from, and a test covers
   a fork turn after a main turn. U5 is removed because the scope is a decision, not an open question.
9. D12's record shape omitted `schema`. Accepted. D12, C9 and T8 all carry the same record:
   `{schema: 1, at, session_id, turn_id, status, names}`.
10. Feature behaviour could not be run. Correct and expected: this round reviewed documents, every
    implementation file is a phase 3 deliverable. The verification items stay unchecked until then.

## Result

All nine findings were reproduced against the files and fixed in one pass. The changes were published
to kata (`93w7`, `cnmg`, `d3f6`), and the same reviewer re-read the current files and returned
`VERDICT: APPROVE`, confirming every finding closed. That approval covers this delta only; the whole
document set still needs one verdict after the main round-2 findings are fixed.
