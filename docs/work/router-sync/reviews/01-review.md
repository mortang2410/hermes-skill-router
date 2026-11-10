# Review 01: spec, skeleton and tickets

Scope: `docs/work/router-sync/spec.md`, `docs/work/router-sync/skeleton.md`, tickets T1 to T7, README.
Reviewer: provider `codex`, model `codex/luna-bifrost`, effort `max`. Read-only.
Round: 1. Verdict: CHANGES.

## Findings

1. [P1] Preserve the complete provider request. Exact text, `tickets/T5-compose-live.md:32-35`:
   "Return `{"request": {"messages": [...stripped, {"role": "system", "content": block}]}}`." Why:
   middleware replaces the whole payload with `next_payload`, not a merge
   (`hermes_cli/middleware.py:69-76`); Hermes kwargs include `model` and `tools`
   (`agent/chat_completion_helpers.py:1470-1479`). Following the sentence drops them and breaks the
   call. Replacement: "Return `{"request": {**request, "messages": [...]}}`, preserving all original
   top-level fields; test that model, tools, and an unrelated field survive."

2. [P1] Index the same skills Hermes exposes. Exact text, `tickets/T3-index-embed-cache.md:28-31`:
   "find every `SKILL.md` under `cfg.skill_roots` when it is non-empty, else under the Hermes roots
   `skills.create_dir` plus `skills.external_dirs` read from the profile config ... Skip a file with no
   `name`". Why: Hermes also includes local/project roots and enabled plugin skills, applies
   visibility gates, and emits canonical load names (`agent/prompt_builder.py:1201-1210,1268-1274,1395-1470`);
   `skill_view` can fall back to a folder name (`tools/skills_tool.py:625-647`). The scan omits
   routable skills and may suggest unopenable names. Replacement: "Index the same visible catalog and
   canonical `load_name` values Hermes uses, including local/project/create/external roots and plugin
   skills with the same gates; name the supported API or narrow the coverage claim."

3. [P1] Support the actual provider wire shapes. Exact text, `tickets/T2-plugin-config-strip.md:46-49`:
   "leaves `messages` untouched when the payload has no `messages` list". Why: middleware runs after
   transport serialization (`agent/turn_api_request.py:122-151`); Codex Responses uses
   `instructions`/`input` (`agent/transports/codex.py:775-783`), Anthropic uses `system` plus
   `messages` (`agent/anthropic_adapter.py:676-678`). Responses no-ops; Anthropic's system text is not
   stripped and an appended system-role message can invalidate the request. Replacement: "Define and
   test adapters for Chat Completions, Responses, and Anthropic, or explicitly restrict supported
   `api_mode` values."

4. [P1] Do not softmax DSH probabilities twice. Exact text, `tickets/T4-rerank-select-inject.md:35-37`:
   "softmax over `scores` ... keep rows whose share is at least `cfg.cutoff`". Why: shared 8010 returns
   softmax probabilities (`/home/wilder/.scripts/decider-rerank-server.py:97-118`); DSH clamps and
   thresholds them directly (`jev-client.ts:542-551`; `suggestion-selector.ts:77-91`). With 50
   candidates, cutoff .01, scores `[1,0,...,0]` become second-softmax shares >= .0193, so all 50 pass
   and the cap returns 8, versus one in DSH. Replacement: "Compare `relevance_score` directly and
   inclusively to `cutoff`; add fixed-score parity tests against the DSH selector."

5. [P2] Match the shared rerank document format. Exact text, `tickets/T5-compose-live.md:29-31`:
   "rerank `name + "\n" + description` for those rows". Why: DSH sends
   `This skill supplies ... Skill: <name> - <description>` (`jev-client.ts:149-151,459-463`); the local
   service parses that prefix (`decider-rerank-server.py:84-118`). T5's string becomes an unnamed
   candidate, so the same endpoint gets different effective inputs. Replacement: "Use the DSH prefix
   and exact name/description framing; assert the outgoing request body matches."

6. [P1] Require TLS for non-loopback endpoints. Exact text at `spec.md:97,101`: the URL bound is
   "http(s) URL". Why: request text and skill descriptions are sent to the configured services
   (`spec.md:28-32`); remote HTTP exposes them in cleartext. The plugin guidance says HTTPS except
   loopback (`hermes-plugin-development/references/tools-and-hooks.md:141-142`). Replacement: "Allow
   HTTP only for loopback; require HTTPS elsewhere, reject URL credentials and downgrade redirects, and
   disclose what request data is sent."

7. [P2] Include endpoint identity in the DSH cache key or narrow the claim. Exact text,
   `tickets/T6-dsh-http-embedder.md:40-41`: "so an endpoint or model change invalidates the cached
   vectors." Why: the current fingerprint stores model id and catalog hash only
   (`packages/dsh-skill-router/src/index/embedding-index.ts:161-169`); changing base URL/path with the
   same id leaves it unchanged. Replacement: "Include selected HTTP base URL and path in the
   fingerprint and test endpoint-only changes, or state that only model-id changes invalidate the
   cache."

8. [P2] Specify systemd unit installation. Exact text, `tickets/T1-embeddings-service.md:74-75`:
   "`systemctl --user start hermes-skill-router-embed` ... and the unit is `enabled` at user level."
   Why: the exclusive unit file is in the checkout (`:16-20`), with no install/link or `daemon-reload`
   step; the acceptance command will not find it from a clean install. Replacement: "Install/link the
   tracked unit into the user unit search path, run `daemon-reload`, then enable and start it."

9. [P2] Fix the impossible manifest assertion. Exact text, `tickets/T2-plugin-config-strip.md:63-65`:
   "asserts the registered middleware kinds equal the manifest's declared surface". Why: T2:37-38 says
   `provides_middleware` is not a manifest field; it is a catalog capability
   (`hermes-plugin-development/references/plugin-yaml.md:49-53`). Replacement: "Assert `register()`
   records exactly one `llm_request` middleware and no tools/hooks; compare catalog capability only if
   a catalog entry exists."

10. [P2] Preserve skill-use guidance for an empty shortlist. Exact text, `skeleton.md:97-99`: "both the
    prose section and the block are removed". Why: Hermes's `## Skills` prose instructs the model to
    load relevant skills via `skill_view` (`agent/prompt_builder.py:1364-1382`); `render([])` returns
    `""` (`skeleton.md:183-189`) and `maxSuggestions: 0` is valid (`spec.md:108`). Replacement: "Strip
    catalog rows but retain equivalent static skill-use guidance on every request; do not append an
    empty system message."

## Claims checked and found correct

- `llm_request` is a valid middleware kind and `{"request": {...}}` replaces the provider payload:
  `hermes_cli/middleware.py:21-26,88-99`; registration at `hermes_cli/plugins.py:933-938`.
- Disabled skills can be refused by `skill_view` and are hidden from the catalog:
  `tools/skills_tool.py:625-631`; `agent/prompt_builder.py:1395-1462`.
- Active DSH config uses shortlist 50, cutoff .01, endpoint 8010, `tasksource-decider-nano`,
  `cohere-compatible`: `profiles/web/cordis.patch.yml:455-466`; live 8010 health returned `status: ok`
  and that model (`/home/wilder/.scripts/decider-rerank-server.py:135-161`).
- DSH's embedder contract is one text per call and its index normalizes vectors:
  `packages/dsh-skill-router/src/index/embedding-index.ts:98-101,172-200`.

VERDICT: CHANGES
