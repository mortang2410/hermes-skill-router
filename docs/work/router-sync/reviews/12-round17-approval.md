# Review round 17: approval after the one-shot header delta

Reviewer: the same independent luna max child. Scope: the two-header contract that came out of the
phase 4 code review, plus the README, guard and tests that had to name both variants.
Verdict: APPROVE, no open finding.

## Findings closed in this round

- R17-1: `one_shot` had no declared home: C3's `Partition` listed only kept, routed and unknown names
  while C8 selected the header from `kept.one_shot`. `Partition` now declares `one_shot`, render takes
  it as an explicit argument, and T5 passes `Partition.one_shot`.
- R17-2: both headers used to be one fixed string. They are now `HEADER` and `ONE_SHOT_HEADER`, sharing
  their first line so the C9 double-injection guard still recognises either, and differing only after
  it. The one-shot variant keeps the restriction its prose carried: no general process skills for work
  already known, and no creating or editing skills.
- R17-3: T7 and the README quoted only the normal header. Both are now quoted verbatim with the shared
  first line named, and `tests/test_readme.py` asserts both texts against the live constants.

## Consequences

The document set is approved again. The phase 4 second review round (`code-02.md`) verifies the three
code fixes; phase 5 verification, commit, push and the completion gate follow.
