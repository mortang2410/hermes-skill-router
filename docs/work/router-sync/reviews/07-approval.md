# Review round 7: approval

Reviewer: an independent luna max child session with no part in writing the documents or the code.
Scope: spec.md, skeleton.md, tickets T1 to T8 and STATE.md, including the two implementation
findings that reopened the document set and the one ticket-completeness finding raised in round 7.
Verdict: APPROVE, no open document finding.

## Findings closed in this round

- R7-1: `plugin.yaml` must declare `provides_middleware: ["llm_request"]`, because the pinned
  `hermes plugins validate` fails a plugin whose `register()` registers middleware it does not
  declare (`hermes_cli/plugin_validate.py:400-417`); this build's manifest parser logs one
  `unknown manifest field(s) ignored` warning per load (`hermes_cli/plugins_manifest.py:32-39,174-179`),
  which the ticket now names as expected and harmless. Accepted and written into T2.
- R7-2: a coding posture demotes some categories to `category [names only]: a, b` lines inside
  `<available_skills>` (`agent/prompt_builder.py:1378`), and the approved strip dropped those names.
  Accepted: both row shapes are parsed, demoted names enter `kept` with an empty description, are never
  routed even when `deferredSkills` names them, and C8 renders an empty description as `- name`.
  Reproduced against the installed Hermes build, fixed, and retested.
- R7-3: T2's manifest field list omitted the non-empty `description` that the validator requires
  (`hermes_cli/plugin_validate.py:96-107`). Accepted and added, with the reason.

## Evidence the reviewer read

T2 lines 33-37 (description requirement), 38-44 (middleware declaration and the expected warning),
55-69 (both row shapes and the deferred override), 89-93 (the validator compares the manifest key).
Skeleton C3 post and invariants, C8 rendering, T3, T4, T5 and T7 sections named in the finding.
The reviewer did not rerun `hermes plugins validate`; the orchestrator ran it after the edits and
recorded 14 of 14 checks with exit 0.

## Consequences

The document set is approved for phase 3. Any later edit to the spec, the skeleton or a ticket voids
this approval and needs a fresh round. Implementation may continue from T2 onward; T3 is the next
ticket whose prerequisites are met.
