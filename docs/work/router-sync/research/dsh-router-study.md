# How the DSH skill router works, and what the Hermes port must mirror

Written by the orchestrating session after reading the source directly. Every claim below carries the
file and line it came from. Claims marked `inferred` are design choices for the Hermes port, not
observations of DSH.

Source: `/home/wilder/git-repos/skill-router-and-jev/packages/dsh-skill-router/src`.

## 1. Shape of the plugin

Five stage directories, one concern each: `index` (embedder, persisted index, cosine shortlist),
`judgment` (the reranker client), `selection` (cutoff and render), `injection` (config and the
pre-step listener), `suppress` (hiding the skill tool, refreshing the catalog). `src/index.ts` wires
them (1419 lines).

The router is a per-agent plugin that listens on `agent/pre-step` and `system-prompt/assemble`.

## 2. How the skill catalog is suppressed

DSH does not rewrite prompt text. It removes the model-facing `skill` tool for that agent, and
`dsh-tool-skill` publishes its catalog only while that tool is visible
(`suppress/tool-restriction.ts:1-30`, citing `dsh-tool-skill/lib/index.js:203-217`). One restriction
removes two costs at once: the schema and the roughly 12,500-token catalog message.

The restriction must be reached through `agent.ctx`; on the plugin's own context it throws by design
(`tool-restriction.ts:30-52`). The disposer it returns must be called, or the agent scope keeps an
orphaned restriction.

## 3. Tools: eager, deferred, warm

- `eagerTools` is a configured list that REPLACES the default list rather than extending it
  (`injection/router.ts:96-101`, default at `:141` = `[...DEFAULT_EAGER_TOOLS]`).
- The kept set after the assembly transform is: the eager list, the bridge tools
  (`tool_search`, `tool_describe`, `tool_call`), this session's warm tools, `run_code` when present,
  and `structured_output` when present (`tools/assembly.ts:20-40`).
- The warm set is least-recently-used with a cap of 8 by default (`maxWarmTools`,
  `router.ts:102-107`), and warming happens inside the bridge calls themselves.
- The transform runs after `next()`, and nothing is hidden unless the guard passes: plugin enabled,
  the three bridge tools registered, the tool index ready, and at least one deferrable tool
  (`assembly.ts:20-40`).

Consequence worth stating plainly: the tools array changes whenever the warm set or the eager list
changes, which changes the prompt prefix and therefore the provider's prompt cache.

## 4. Skills: there is no eager/deferred split

The config schema has no skill-level eager or deferred key (`injection/router.ts:82-148`). The only
`eagerTools`-shaped key is for tools. Skills are handled entirely by the routing pass: shortlist,
judge, cut, inject.

`inferred`: an eager-skills and deferred-skills list is therefore a new concept for the Hermes port,
not a port of an existing one. It is a reasonable concept because Hermes cannot hide `skill_view`
without also disabling skill loading, so the only lever is which rows stay in the prompt.

## 5. The retrieval path

- Shortlist stage: `shortlist()` embeds `queryText(request, recent)` and keeps the top `k` by cosine
  over the index (`index/shortlist.ts:195-243`), default `k` 32 (`:22`), live config 50. Ranking is a
  plain dot product over normalised vectors, sorted descending (`:166-190`). A vector width mismatch
  throws rather than scoring a wrong ranking.
- The shortlist is a hard admission gate: a skill it drops cannot be recovered later.
- `isOfferable()` fails closed: when a catalog is supplied and the skill is absent from it, the skill
  is refused, because the index carries no invocation policy; `modelInvocable === false` is also
  refused (`:152-163`). A caller with no catalog at all stays permissive.
- Judgment stage: one rerank request scores the shortlist; documents are
  `This skill supplies ... Skill: <name> - <description>` (`judgment/jev-client.ts:149-151,459-463`),
  and the local server parses exactly that prefix (`/home/wilder/.scripts/decider-rerank-server.py`,
  `SKILL_PREFIX` and `parse_document`).
- Selection: `select()` keeps `probability >= cutoff` (inclusive), drops unscored candidates, drops
  names already suggested, and sorts descending with ties keeping shortlist order
  (`selection/suggestion-selector.ts:70-92`). Default cutoff 0.3 (`:41`), live config 0.01.
- Rendering: one line per skill, `- <name>: <description>` with the description collapsed to one line
  and the route appended as ` [read: <path>]` when a path is known and the description does not
  already contain the marker (`:106-159`). Empty selection renders as an empty string.
- Failure at any stage returns no suggestions plus a warning; it never throws into the turn.

## 6. Index and cache

The index is persisted, and its identity is the embedding model id plus a hash of the catalog text
(`index/embedding-index.ts:161-169`). Note what is absent: the endpoint URL. For the Hermes port with
a shared HTTP embedder that omission matters, because repointing the URL while keeping the model id
would silently reuse vectors from the other service.

A partial-view guard refuses a catalog read that returns fewer than half the cached index entries, and
a known-good catalog is retained rather than replaced by a short read.

## 7. Session dedup

Prior suggestions are recovered from the session's own messages: every message the router built is
stamped `source.kind = "skill-router"`, and only dash-prefixed lines in those messages are parsed
back into names (`selection/suggestion-selector.ts:174-218`). No separate ledger is kept.

Hermes has no equivalent message source tag, so the port must recognise its own injected block by a
marker instead. `inferred`.

## 8. Mapping table

| DSH mechanism | Observed source | Hermes port |
|---|---|---|
| Deny the `skill` tool, catalog disappears | `suppress/tool-restriction.ts:1-30` | Not possible: Hermes needs `skill_view`. The port strips catalog rows from the outgoing prompt text instead. |
| `eagerTools` kept visible, rest deferred | `injection/router.ts:96-101`, `tools/assembly.ts:20-40` | `eagerSkills`: rows never stripped and never routed. |
| Tools not eager are deferred and searchable | `tools/assembly.ts:20-40` | `deferredSkills`: the routable pool. Empty means every non-eager skill is routable. `inferred`. |
| Warm set, LRU 8, warmed by bridge calls | `router.ts:102-107` | No analogue: Hermes has no tool loading. |
| Shortlist top-k cosine | `index/shortlist.ts:166-243` | Same stage over the parsed catalog rows, `shortlistSize` 50. |
| `isOfferable` fail-closed on absent or non-invocable skills | `index/shortlist.ts:152-163` | No analogue: Hermes has no `modelInvocable`, and the index is built from the catalog itself, so every indexed skill is by construction present. |
| Rerank documents with `Skill: <name> - <description>` | `judgment/jev-client.ts:149-151,459-463` | Same prefix as the default `documentPrefix`. |
| Inclusive cutoff on judged probability | `selection/suggestion-selector.ts:70-92` | Same, no second softmax. |
| `- name: description [read: path]` | `selection/suggestion-selector.ts:106-159` | `- name: description` only, because `skill_view` resolves the name. `inferred`. |
| Dedup via `source.kind = skill-router` | `selection/suggestion-selector.ts:174-218` | Marker on the injected block. `inferred`. |
| Index identity = model id + catalog hash | `index/embedding-index.ts:161-169` | Model id + catalog hash + endpoint URL. `inferred`. |
| Partial-view guard | `suppress/catalog-refresh.ts` | No analogue: the port indexes the catalog it just received, so there is no second opinion to disagree with. `inferred`. |

## 9. What a naive port gets wrong

1. Two softmaxes. The reranker already returns probabilities; normalising again turns one confident
   skill into a full page of them.
2. Rewriting the payload instead of extending it. Hermes middleware replaces the whole request, so a
   returned `{"messages": ...}` without `model` and `tools` breaks the turn.
3. Rebuilding the catalog by scanning the filesystem. Hermes already decided what the model may see;
   the honest source is the catalog text in the request.
4. Assuming an eager-skill concept exists upstream. It does not; only tools have one.

## 10. Observed vs inferred

- Observed: sections 1 to 7 and the left column of section 8, each with a file and line.
- Inferred: every right-column entry for a Hermes mechanism that has no DSH counterpart, the
  `eagerSkills`/`deferredSkills` semantics, the block marker for dedup, and the endpoint in the index
  fingerprint.
- Not verified here: whether Hermes ever rebuilds its catalog inside one session (tracked as U3 in
  the skeleton), and how a provider accepts a trailing second system message (U1).
