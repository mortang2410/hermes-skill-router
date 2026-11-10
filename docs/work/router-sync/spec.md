# hermes-skill-router

Status: draft for review.

Skeleton: `docs/work/router-sync/skeleton.md` (lint status OK).
Tickets: `docs/work/router-sync/tickets/` (T1 to T8).

## Problem

Hermes renders every skill it can find into the system prompt. Measured on this
machine with the live profile: 427 rows, 47,981 characters inside
`<available_skills>`, about 12,400 tokens on every request, and the fork agents
(`/btw` side questions and the after-turn background review) replay the same
prompt. Hermes has no switch that hides the catalog while leaving skill loading
usable: `skills.disabled` removes a skill from the catalog and makes
`skill_view` refuse it (`tools/skills_tool.py:629`), so hiding it that way also
removes the model's only way to open a skill.

DSH already solved this with a two-stage router: an embedding shortlist, then a
rerank judgment pass, injected as a short block of names and descriptions. The
same two models run on this host. The port is worth doing because the catalog is
the largest fixed cost in every Hermes session.

## Solution

A Hermes plugin that rewrites each outgoing model request:

1. Remove the whole `## Skills` section from the request's system message: its heading, the prose
   before and after the block, and the `<available_skills>` catalog rows. The block yields two row
   shapes: `- name: description` rows and `category [names only]: a, b` lines that a coding posture
   demotes. The routed rows become the index; the kept rows, meaning the eager rows, any row a
   non-empty deferred list leaves listed, and every name on a `[names only]` line, are listed again in
   the injected block, so they never disappear.
2. Embed that request's last user text against a shared HTTP embeddings endpoint. The index is
   built from the routed rows only, under the names `skill_view` accepts, so eager rows and rows
   outside a non-empty deferred list are never suggested from the index.
3. Keep the nearest candidates, rerank them against a shared HTTP rerank
   endpoint, and select those at or above the cutoff.
4. Append one compact block with the suggested rows and the kept rows, names and single-line
   descriptions, behind the guidance the removed section carried.

The rewrite runs only on OpenAI-compatible Chat Completions requests (`api_mode ==
"chat_completions"`). Codex Responses and Anthropic Messages serialize the prompt into other fields
(`instructions`/`input`, `system`), so those requests pass through untouched and keep the catalog.
`skill_view` stays untouched, so the model opens whatever the block names. If
either endpoint is unreachable, or the embedding index cannot be built, the
plugin changes nothing and the catalog stays in place, so a broken router can
never leave a session unable to reach a skill.

Both endpoints are the ones DSH uses, reachable by URL, and documented well
enough that a third party can point the plugin at their own models.

## User stories

- S1 As the operator, I stop paying 12,400 tokens of skill list on every request.
- S2 As the operator, the model still gets the skills a turn needs, by name.
- S3 As the operator, the model can still open any skill itself with `skill_view`.
- S4 As the operator, a turn whose router is down still works, with the catalog back.
- S5 As the operator, `/btw` and the background review route like any other turn.
- S6 As the operator, Hermes and DSH ask the same endpoint for the same vectors.
- S7 As the operator, every knob (candidate count, cutoff, endpoints, limits) is
  configurable and documented.
- S8 As a third-party user, I can point the plugin at my own embeddings and
  rerank endpoints and publish nothing of mine.
- S9 As the operator, I can pin skills that must always stay listed, and I can
  limit routing to a named set, the same way DSH splits eager from deferred.
- S10 As the operator, I can see which skills the router suggested on the last turn, as skill names
  with the record's turn id and time in a caption and no descriptions or request text, in a collapsible
  display in Hermes Desktop and the Ink TUI, and that display costs no tokens.

## Implementation decisions

- D1 Hiding happens in a plugin `llm_request` middleware. A callback returning
  `{"request": {...}}` replaces the provider payload before it is sent
  (`agent/turn_api_request.py:143`, `hermes_cli/middleware.py:88`, contract in
  `tests/hermes_cli/test_plugins_middleware_chain.py`). No upstream patch, and a
  Hermes update cannot revert it.
- D2 One rule for every spawn: the router runs on every Chat Completions request that reaches the
  middleware, using that request's last user-role text as the query. Fork agents
  replay the parent's `api_content` sidecars (`agent/turn_context.py:1305`), so
  their own text is what ships; fresh spawns (cron, kanban workers, `hermes -z`,
  the `/btw` one-shot fallback) route like a normal turn. No spawn-type list is
  maintained.
- D3 The injected block carries the suggested rows and the kept rows (eager rows plus anything a
  non-empty deferred list leaves listed), names plus descriptions, one line each, descriptions trimmed
  to `maxDescriptionChars`, behind a fixed header that tells the model to open a skill with `skill_view`,
  in a normal and a one-shot variant that share their first line; the one-shot variant keeps the
  restriction its prose carried, that general process skills are not loaded for work already known
  and skills are not created or edited
  and carries the guidance the removed section provided. A kept row whose catalog line carried no
  description is listed by name only. The strip
  and the injection are atomic: the whole skills section is removed only on a request that receives a
  non-empty block, so a request with no selected suggestion and no kept row keeps the original section
  byte for byte. `skill_view` stays enabled.
- D4 Shared endpoints: both endpoints are configurable. A plain `http` URL is accepted only for a
  loopback host; any other host must be `https`, URL credentials are rejected, a base URL carrying a
  query or a fragment is rejected because the request path is appended to it, and redirects are not
  followed: both clients build requests through one opener whose `redirect_request` returns `None`, so
  urllib turns a 3xx into an `HTTPError` and no follow-up request leaves the process. Two values other
  than the defaults change where the request text goes, so the README states what leaves the process. Embeddings on `http://127.0.0.1:8012` with
  OpenAI-compatible `POST /v1/embeddings` (model `bge-m3`, the BGE-M3 ONNX
  already on disk), rerank on `http://127.0.0.1:8010` with Cohere-shaped
  `POST /v1/rerank` (model `tasksource-decider-nano`). Port 8000 stays as
  Hindsight's own `ms-marco-MiniLM-L-6-v2` service and is not touched.
- D5 The embeddings service is new, lives in this repository under `service/`,
  runs from the miniforge python that already has onnxruntime, tokenizers,
  fastapi and uvicorn, and gets a systemd user unit beside the two rerank units.
- D6 DSH's router gains an HTTP embedder implementing its existing one-text
  contract (`packages/dsh-skill-router/src/index/embedding-index.ts:98`), with `embedBaseUrl`,
  `embedPath`, `embedModelId` and `embedTimeoutMs` config keys, and the endpoint identity added to
  its index fingerprint. Its `modelPath` ONNX path stays
  as the fallback, and the profile switches to the URL once the service is up.
- D7 The plugin ships as stdlib-only Python (`urllib`), so an install needs no
  new package beyond what Hermes already vendors: onnxruntime and tokenizers are
  only used by the service, not by the plugin.
- D8 Fail-open and fail-inert: endpoint error, timeout, malformed response, empty
  index, a request with no user text, or an `api_mode` other than `chat_completions` all mean "leave
  this request exactly as it arrived". Each distinct reason logs once, not once per request. The
  replacement payload keeps every top-level key it received; middleware replaces the whole payload,
  so returning only `messages` would drop `model` and `tools`.
- D9 The index is built from `Partition.routed` only and persisted at `cachePath`, keyed by a hash of
  each routed row's name and full description, the description clip, the model id, the document
  format version and the endpoint URL. A change to a kept row alone leaves the fingerprint and the
  cache intact. A change to a routed row re-embeds, including a description edit past the clip,
  because the cached descriptions feed the rerank documents and the injected block. A model or
  endpoint change re-embeds too. The catalog is frozen per session, so a session start reuses the
  cache.
- D10 The rerank call must match DSH byte for byte in the fields that matter: one document per
  candidate, `This skill supplies a procedure, tool, or domain-specific instructions that the agent
  would actually need to carry out the current task correctly. Skill: <name> - <description>`, and the
  returned `relevance_score` is a probability compared directly and inclusively against `cutoff`. No
  second softmax.
- D11 Eager and deferred skills mirror DSH's eager/deferred split for tools, applied to skills.
  `eagerSkills` (default empty) names skills that are never routed and never removed from the system
  prompt, so they stay listed on every request exactly as today. `deferredSkills` (default empty)
  names the routable pool: empty means every skill that is not eager is routable, mirroring DSH's
  posture that everything not eager is deferred, and a non-empty list restricts routing to those names
  so every other skill stays listed. A name in both lists is eager, with one note logged; a configured
  name that the catalog does not carry is ignored and named in the config notes. The partition is
  computed once per request from the catalog rows, and the index covers routed rows only. Names on a
  `[names only]` line are always kept and never routed, even when `deferredSkills` names them, because
  the renderer omitted their descriptions.
- D12 The suggestion display is a names-only side channel, never request text. On each request it
  decides on, the router writes `{schema: 1, session_id, turn_id, at, status, names}` to
  `suggestionsPath`, with the selected suggestion names only, never a kept row, capped at
  `maxDisplayNames`, and no description text;
  `showSuggestions` false stops the writes and removes an existing record, so both clients show idle.
  The newest record wins and carries its session and turn id, which is what a second session or a fork
  replacing it looks like on screen. Two surfaces read that record: a desktop pane the plugin
  contributes, fed by the plugin's own `/api/plugins/hermes-skill-router/suggestions` route, and an Ink
  TUI ambient user widget that the plugin ships and the install step links into
  `$HERMES_HOME/tui-widgets/`, because Hermes exposes no plugin display hook in the TUI; the widget
  takes the record path as its launch argument and defaults to the `suggestionsPath` default. No
  Hermes UI code is patched and nothing is added to the provider payload, so the display costs no
  tokens. Both surfaces paint one muted caption under the heading, `turn <turn_id> · <local HH:MM>`,
  from the record alone and only for a routed record, suppressed while a surface is loading, when its
  route is missing and when the read fails, so a reader can tell which turn produced the names; the
  classic prompt_toolkit TUI exposes no plugin display surface and stays out of scope.

### Config keys, defaults and bounds

| Key | Default | Bound | Effect |
| --- | --- | --- | --- |
| `enabled` | `true` | bool | Master switch. False changes nothing. |
| `eagerSkills` | empty | list of catalog names | Skills always listed and never routed. |
| `embedBaseUrl` | `http://127.0.0.1:8012` | http for loopback only, https elsewhere | Embeddings endpoint root. |
| `embedPath` | `/v1/embeddings` | path | Endpoint path appended to the base URL. |
| `embedModelId` | `bge-m3` | string | `model` field sent to the endpoint. |
| `embedTimeoutMs` | `6000` | 100..60000 | Per-request embeddings timeout. A cold build sends
  16-row chunks that take 1.2 to 2.0 s on this CPU host, so 2000 was marginal and made the first
  turn inert. |
| `rerankBaseUrl` | `http://127.0.0.1:8010` | http for loopback only, https elsewhere | Rerank endpoint root. |
| `rerankPath` | `/v1/rerank` | path | Endpoint path appended to the base URL. |
| `rerankModel` | `tasksource-decider-nano` | string | `model` field sent to the rerank endpoint. |
| `rerankProvider` | `cohere` | `cohere` | Request and response shape. |
| `documentPrefix` | the DSH skill prefix | string | Exact text before `<name> - <description>` in a rerank document. |
| `deferredSkills` | empty | list of catalog names | Routable skills. Empty means every non-eager skill is routable. |
| `rerankTimeoutMs` | `20000` | 100..60000 | Per-request rerank timeout. |
| `routingTimeoutMs` | `30000` | 1000..60000 | One whole routing pass, from the query embed through
  the rerank reply. The budget starts after the index is loaded or built, so the cold build is never
  cut, and each per-request timeout is capped at the time left. |
| `shortlistSize` | `50` | 1..383 | Candidates carried from embeddings into rerank. |
| `cutoff` | `0.01` | 0..1 | Rerank `relevance_score` a candidate needs, compared inclusively. |
| `maxSuggestions` | `8` | 0..50 | Hard cap on suggested rows; 0 selects none, while kept rows are still listed. |
| `maxDisplayNames` | `12` | 1..50 | Names the pane and the widget list; the count stays visible when collapsed. |
| `maxDescriptionChars` | `400` | 3..4000 | Description characters kept per row, in the
  injected block and in the embedding document text; both sides embed the same text when their
  values agree, and both default to 400. |
| `cachePath` | `~/.cache/hermes-skill-router/index.json` | path | Persisted index; empty disables persistence. |
| `showSuggestions` | `true` | bool | Let the desktop pane and the TUI widget show the last suggestion names. False stops the record and removes an existing one. |
| `suggestionsPath` | `~/.cache/hermes-skill-router/suggestions.json` | path | Display record both surfaces read; names only, never request text. |
| `healthCheckMs` | `60000` | 0..3600000 | Minimum gap between endpoint health probes; 0 probes every request. |

## Testing decisions

- Every unit test that needs a model uses a local fake HTTP server, so tests run
  with no GPU, no model files and no network.
- The catalog strip is covered by a mutation check: with the strip disabled, the
  test that asserts the catalog is absent must fail.
- The strip is tested against a real-shaped system message: the `## Skills` heading, the prose before
  and after the block, the tags, the single-query variant, and a demoted `[names only]` line. On
  success the routed rows are gone, the guidance and the kept rows are in the block, the demoted names
  are listed by name and absent from the index, and on failure the message is byte-identical.
- Both HTTP clients are tested against a fake endpoint that answers 302 and 307: the call fails and the
  fake records no second request.
- Selection parity is covered against the DSH selector on fixed score vectors, including the
  all-but-one case at cutoff 0.01, so a second softmax or an exclusive comparison fails a test.
- The rerank request body is asserted against the DSH document text, and the rewrite is asserted to
  preserve `model`, `tools` and an unrelated top-level key. A `codex_responses` payload must come out
  unchanged.
- The eager/deferred partition is tested at its boundaries: empty both lists, an eager name, a
  deferred name, a name in both, an unknown name, and a deferred list that excludes a skill so it
  stays listed.
- The display cannot cost tokens: the rewritten payload with the display enabled is deep-equal to the
  payload the same rewrite produces with the record writer replaced by a no-op, message count and roles
  included, and no record field ever appears in any message. A test that only compares
  `showSuggestions` true with false is not enough, because an addition present in both paths would
  pass it.
- The display surfaces are checked at their seams: the route returns the record or an idle answer, a
  pre-existing record is removed when `showSuggestions` is false, the caps hold with 50 names, the
  newest record wins after two rewrites from different sessions, the widget file passes as ES module
  syntax, `desktop/plugin.js` passes the static admission lint inside `hermes plugins validate` (a
  tripwire for selected constructs, not an SDK allowlist), both enable gates are documented and
  exercised, and one live routed turn is read in both clients.
- The service is verified by running it and comparing one vector against a
  reference computed directly from the ONNX model, plus a shape and norm check.
- The unit is installed by linking the tracked file into the user unit directory, `systemctl --user
  daemon-reload`, then `enable --now`, so a clean checkout can be started twice.
- The end-to-end path is verified on this machine against the live 8012 and 8010
  endpoints, with a real Hermes one-shot turn showing the catalog gone and the
  block present, then a turn with the service stopped showing the catalog back.
- DSH's HTTP embedder is verified by its own suite plus one router boot against
  the live endpoint, comparing the shortlist to the pre-change one for a fixed
  query.

## Out of scope

- Changing DSH's catalog suppression, tool deferral or injection format.
- Any Hermes UI code change: the display is a plugin pane plus an Ink TUI widget.
- The classic prompt_toolkit TUI, which exposes no plugin display surface.
- Publishing to PyPI or a Hermes skill registry.
- macOS, Windows, and non-CPU embedding backends.
- Any change to the Hindsight reranker on port 8000.

## Directive table

| His words | Where this spec carries it |
| --- | --- |
| "port our dsh skill router to hermes" | Solution, D4, D6 |
| "I want to be able to hide hermes skill catalog and route relevant skills when needed" | D1, D3 |
| "The hermes plugin should have config settings for setting the embedder and reranker, as well as relevant parameters (like how many candidates etc.)" | D4, config table |
| "and a github README.md" | S8, ticket T7 |
| "go with option 1" | D1 |
| "make sure that all incidental spawns like btw or background reviews etc. in hermes can also use skill router" | D2, S5 |
| "this fact should be documented" | ticket T7 README section on spawns |
| "I wonder if background reviewer can still work properly" | D3 plus README note: `skills_list` and `skill_view` stay available to the review fork |
| "isn't it easier to enable for all spawns ? then there is no need to worry about cases" | D2 |
| "can use the same instances of embedder and reranker that are currently running for dsh (through config setting pointing at endpoints)" | D4, D6 |
| "Eventually we want to publish this plugin for others to use and they can bring their own models" | S8, D7, T7 |
| "serve everything with proper endpoints so hermes and dsh can make the same requests to the same endpoint" | D4, D5, D6, T1 |
| "Same. but allow the parameters to be overridden in config with full documentation" | D3, config table, T7 |
| "note that I want to mimic dsh skill router and that means there should be a list of eager skills, and list of deferred skills in config" | D11, config table `eagerSkills` and `deferredSkills`, T2, T7 |
| "I also want there to be a short collapsible display in the UI, for instance in hermes desktop and hermes TUI, showing that skill router works, and listing the names of the skills that got suggested" | S10, D12, T8 |
| "just names, no descriptions" | D12, T8 |
| "without incurring any extra token use (it's just a pure display thing)" | D12, testing decision on the record, T8 |
| "do hindsight recall to actually learn" | research/dsh-router-study.md, and D11 mirrors the eager/deferred split it documents |
