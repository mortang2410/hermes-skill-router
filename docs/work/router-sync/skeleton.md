# Skeleton: hermes-skill-router

- status: draft
- mode: design
- spec: docs/work/router-sync/spec.md (working tree, uncommitted until phase 5)
- scope: hermes-skill-router repository (service/, hermes_skill_router/, dashboard/, desktop/, tui/, plugin.yaml, README.md, tests/) plus packages/dsh-skill-router in the DSH router repository
- grill: none (settled in conversation, decisions recorded in the spec directive table)

## Requirements map

| Id | Requirement | Spec anchor | Kind |
|----|-------------|-------------|------|
| R1 | Remove the `## Skills` catalog from the request that is actually sent | Solution step 1, D1 | behaviour |
| R2 | Route every spawn by that request's last user text, with no spawn-type list | D2, S5 | behaviour |
| R3 | Put a compact block of skill names and descriptions in the request | Solution step 4, D3, S2 | behaviour |
| R4 | `skill_view` keeps working, so the model can open anything the block names | D3, S3 | boundary |
| R5 | Any endpoint or index failure leaves the request byte-identical | D8, S4 | error |
| R6 | Hermes and DSH use one embeddings endpoint and one rerank endpoint, over HTTP | D4, D5, D6, S6 | interface |
| R7 | Every knob is configurable, bounded, and documented | Config table, S7 | interface |
| R8 | The skill index is persisted and rebuilt only when the routed rows, the description clip, the model id, the endpoint, or the cache or document format version change | D9 | non-functional |
| R9 | DSH's router reaches the same embeddings endpoint through its existing one-text contract | D6 | interface |
| R10 | A third party can point the plugin at their own endpoints and read how | S8, T7 | documentation |
| R11 | Eager skills always stay listed, and routing is limited to the deferred set | D11, S9 | behaviour |
| R12 | The last suggested skill names are readable in a collapsible desktop pane and TUI widget, with the record's turn id and time in a caption and no descriptions or request text, with no change to any request | D12, S10 | behaviour |

## Modules

| Id | Path | Responsibility | Allowed deps | Forbidden deps | Exposed surface |
|----|------|----------------|--------------|----------------|-----------------|
| M1 | `service/embed_server.py` | Is the only place that loads the BGE-M3 ONNX model and turns text into vectors | none | M2, M3, M4, M5, M6, M7, M8, M9, M10 | `POST /v1/embeddings`, `GET /health` |
| M2 | `hermes_skill_router/config.py` | Is the only place that decides what a configuration value means | none | M1, M3, M4, M5, M6, M7, M8, M9, M10 | `Config`, `load`, `DEFAULTS` |
| M3 | `hermes_skill_router/catalog.py` | Is the only place that recognises the skill catalog, partitions it into kept and routed rows, and removes the routed ones | M2 | M1, M4, M5, M6, M7, M8, M9, M10 | `strip_catalog`, `Partition` |
| M4 | `hermes_skill_router/library.py` | Is the only place that decides which skills exist and when the index is stale | M2, M3, M5 | M1, M6, M7, M8, M9, M10 | `load_or_build`, `fingerprint` |
| M5 | `hermes_skill_router/embed.py` | Is the only place that talks to the embeddings endpoint | M2 | M1, M3, M4, M6, M7, M8, M9, M10 | `embed_texts`, `opener`, `HealthGate` |
| M6 | `hermes_skill_router/rerank.py` | Is the only place that talks to the rerank endpoint | M2, M5 | M1, M3, M4, M7, M8, M9, M10 | `rerank` |
| M7 | `hermes_skill_router/select.py` | Is the only place that decides which candidates are suggested | M2 | M1, M3, M4, M5, M6, M8, M9, M10 | `select` |
| M8 | `hermes_skill_router/inject.py` | Is the only place that decides what the injected block looks like | M2, M3, M4, M7 | M1, M5, M6, M9, M10 | `render` |
| M9 | `hermes_skill_router/router.py` | Is the only place that composes one request rewrite and owns its failure policy | M2, M3, M4, M5, M6, M7, M8, M11 | M1, M10 | `rewrite` |
| M10 | `packages/dsh-skill-router/src/index/http-embedder.ts` | Is the only DSH code that turns text into a vector over HTTP | none | M1, M2, M3, M4, M5, M6, M7, M8, M9 | `createHttpEmbedder` |
| M11 | `hermes_skill_router/display.py` | Is the only Python place that writes and reads the display record | M2 | M1, M3, M4, M5, M6, M7, M8, M9, M10, M12, M13, M14 | `record`, `read` |
| M12 | `dashboard/plugin_api.py` | Is the only place the desktop half reads the record through | M2, M11 | M1, M3, M4, M5, M6, M7, M8, M9, M10, M13, M14 | `GET /suggestions`, answered with the record or `{"status": "idle", "names": []}` |
| M13 | `desktop/plugin.js` | Is the only desktop UI surface that shows the names | M12 | M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M14 | `panes` contribution |
| M14 | `tui/skill-router.mjs` | Is the only TUI surface that shows the names, as a user widget the plugin ships | M11 | M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M12, M13 | ambient widget `skills-routed` |

## Contracts

### C1: embeddings endpoint

- kind: data shape
- owner: M1
- stability: stable
- satisfies: R6, R9

`POST /v1/embeddings` body `{"model": str, "input": [str, ...]}` returns
`{"model": str, "data": [{"index": int, "embedding": [float, ...]}, ...]}` with one
entry per input, in input order. `GET /health` returns
`{"status": "ok", "model": str, "dim": int}`.

- pre: the service is running and the model directory exists
- post: vectors are L2-normalised float lists of length `dim`
- errors: 400 `{"error": str}` for a body with no input or a non-string entry; 503 `{"error": str}` while the model is still loading
- invariants: the vector for one text is identical across calls
- state: model loading is `starting -> ready`; requests during `starting` get 503

### C2: config

- kind: function
- owner: M2
- stability: stable
- satisfies: R7, R11

```python
@dataclass(frozen=True)
class Config:
    enabled: bool; embed_base_url: str; embed_path: str; embed_model_id: str
    embed_timeout_ms: int; rerank_base_url: str; rerank_path: str
    rerank_model: str; rerank_provider: str; rerank_timeout_ms: int
    shortlist_size: int; cutoff: float; max_suggestions: int
    max_description_chars: int; cache_path: str; health_check_ms: int
    show_suggestions: bool; suggestions_path: str; max_display_names: int
    document_prefix: str
    eager_skills: tuple[str, ...]; deferred_skills: tuple[str, ...]
    notes: tuple[str, ...]

def load(raw: Mapping[str, Any] | None) -> Config: ...
```

- pre: `raw` is the plugin config mapping, possibly empty
- post: every field is present; an out-of-bound value is replaced by its default and a note naming it is appended to `notes`; an unknown key appends a note and is otherwise ignored
- errors: none; a malformed value never raises
- invariants: `0 <= cutoff <= 1`, `1 <= shortlist_size <= 383`, `0 <= max_suggestions <= 50`, `1 <= max_display_names <= 50`; a plain `http` endpoint URL is accepted only for a loopback host and the value otherwise falls back to its default with a note
- state: none

### C3: catalog strip and partition

- kind: function
- owner: M3
- stability: stable
- satisfies: R1, R11

```python
@dataclass(frozen=True)
class CatalogRows:
    names: tuple[str, ...]; descriptions: tuple[str, ...]

@dataclass(frozen=True)
class Partition:
    kept: CatalogRows; routed: CatalogRows; unknown_names: tuple[str, ...]
    one_shot: bool

def strip_catalog(payload: Mapping[str, Any], *, eager: Sequence[str] = (),
                  deferred: Sequence[str] = ()) -> tuple[Mapping[str, Any], Partition | None]: ...
```

- pre: `payload` is a provider request with a `messages` list, or any mapping
- post: when the first system message holds the skills section (the `## Skills` heading, the prose before and after the block, the `<available_skills>` block and the trailing sentence, or the single-query variant guidance), the whole section is removed and the parsed rows are returned in catalog order under `kept` and `routed`; both row shapes are parsed, `- name: description` rows and `category [names only]: a, b` lines whose names enter `kept` with an empty description; the variant is recorded on the partition (`one_shot` false for the normal prose, true for the single-query guidance) so C8 can match the instruction it removed; otherwise the payload is returned unchanged with `None`
- errors: none; a payload without `messages`, or with no catalog block, returns `(payload, None)`
- invariants: idempotent; the stripped payload is only usable with a block that carries the guidance the section provided; names from a `[names only]` line are always kept, even when `deferredSkills` names them; duplicate catalog names stay in `routed` in catalog order and C4 collapses them; no row appears in both `kept` and `routed`; a name in both configured lists is kept; a configured name the catalog does not carry is reported in `unknown_names`; the names are the catalog's own load names, so `skill_view` accepts each one
- state: none

### C4: index load or build

- kind: function
- owner: M4
- stability: provisional
- satisfies: R8, R2, R11

```python
def load_or_build(cfg: Config, rows: CatalogRows, embedder: Embedder) -> Index | None: ...
def fingerprint(rows: CatalogRows, cfg: Config) -> str: ...

class Index:
    names: list[str]; descriptions: list[str]; vectors: list[list[float]]
    dim: int; fingerprint: str
```

- pre: `rows` is the `routed` half of C3; `embedder` is `M5.embed_texts`
- post: returns an `Index` whose length equals the number of distinct routed names, first occurrence wins, with duplicate names collapsed before embedding; when `cfg.cache_path` names a readable cache whose `version` is the cache version, whose `fingerprint` matches, whose `dim` is a positive integer equal to the width of every stored vector, and whose entry count equals the live deduped routed row count, returns that cache unembedded; otherwise embeds every row in chunks of at most 16 rows per request, so the configured timeout bounds each request, and writes the cache
- errors: returns None when `rows` is empty or embedding returns None; a cache write failure is logged and the in-memory index is still returned
- invariants: `len(names) == len(descriptions) == len(vectors)` equals the unique name count, never the raw routed row count; every vector has length `dim`; the document text is `name + "\n" + description` clipped to `cfg.max_description_chars`; the fingerprint is a hash of every row's document text and full description plus `cfg.embed_model_id`, the document format version and `cfg.embed_base_url + cfg.embed_path`, so a description edit past the clip still invalidates the cache that supplies the rerank and block text
- state: in-process memo keyed by fingerprint; one build per fingerprint per process

### C5: embed texts

- kind: function
- owner: M5
- stability: stable
- satisfies: R6, R5

```python
def embed_texts(cfg: Config, texts: Sequence[str]) -> list[list[float]] | None: ...
```

- pre: `texts` non-empty
- post: one vector per text, in order, each of equal length
- errors: none raised; returns None on connection error, non-2xx, timeout, or a response whose `data` length differs from `len(texts)`
- invariants: no retry, no partial result; the module also exposes the redirect-refusing opener, and M6 reuses the same one, so both clients share one redirect policy
- state: a health flag with `health_check_ms` gating so a down endpoint is probed at most once per window

### C6: rerank

- kind: function
- owner: M6
- stability: stable
- satisfies: R6, R5

```python
def rerank(cfg: Config, query: str, documents: Sequence[str]) -> list[float] | None: ...
```

- pre: `documents` non-empty; each one is `cfg.document_prefix + name + " - " + description`
- post: one score per document in input order, already a probability
- errors: none raised; returns None on connection error, non-2xx, timeout, or a result whose length differs from `len(documents)`
- invariants: Cohere shape `{"model", "query", "documents", "top_n"}` request, `{"results": [{"index", "relevance_score"}]}` response, re-ordered to input order
- state: shares the health-flag shape with C5, keyed by its own URL

### C7: select

- kind: function
- owner: M7
- stability: stable
- satisfies: R3, R7

```python
def select(scores: Sequence[float], names: Sequence[str], cfg: Config) -> list[tuple[str, float]]: ...
```

- pre: `len(scores) == len(names)`; scores are rerank probabilities; `names` are routed rows only
- post: rows whose score is at or above `cfg.cutoff` are kept, sorted descending by score, truncated to `cfg.max_suggestions`; no second softmax
- errors: none; empty input returns `[]`
- invariants: never returns a name absent from `names`; never returns more than `max_suggestions` rows
- state: none

### C8: render

- kind: function
- owner: M8
- stability: provisional
- satisfies: R3, R4, R11

```python
def render(rows: Sequence[tuple[str, float]], kept: CatalogRows, index: Index, cfg: Config, one_shot: bool) -> str: ...
```

- pre: `rows` came from C7 and holds at most `cfg.max_suggestions` rows (possibly zero); `kept` is the kept half of C3; `index` is C4's index and the description source for suggested rows
- post: one of two fixed headers chosen by `one_shot`, sharing their first line so the C9 double-injection guard recognises either, that tells the model to open a skill with `skill_view` and to call `skills_list` when nothing fits; the one-shot header also keeps the restriction its prose carried, that general process skills are not loaded for work already known and skills are not created or edited; then one `- name: description` line per suggested row with the description looked up in `index` and clipped to `max_description_chars`, then one line per kept row under an `Also listed:` marker; a row with an empty description is listed as `- name`; returns `""` when `rows` and `kept` are both empty
- errors: none
- invariants: `max_suggestions` caps suggested rows only, so `0` selects none and `kept` rows still render; the block never names a skill twice, suggested rows first; every kept name appears in the block whenever the block is non-empty, so no listed skill is lost by the strip
- state: none

### C9: rewrite

- kind: function
- owner: M9
- stability: stable
- satisfies: R1, R2, R3, R5, R11, R12

```python
def rewrite(payload: Mapping[str, Any], *, cfg: Config, index: Index | None,
            embed: Callable, rerank: Callable) -> Mapping[str, Any]: ...
```

- pre: called once per outgoing provider request whose `api_mode` is `chat_completions`
- post: when a block renders to a non-empty string (at least one row at or above `cutoff`, or at
  least one kept row), returns `{**request, "messages": [...stripped, {"role": "system", "content":
  block}]}` where `stripped` has the whole skills section removed; on any failure, on an empty block,
  or on no routed rows, returns the payload unchanged with the section byte-identical. Either way, when the request carried the catalog on
  `chat_completions`, it writes the display record through M11:
  `{"schema": 1, "at": <ISO 8601 UTC>, "session_id": str, "turn_id": str, "status": "routed" | "inert",
  "names": [str, ...]}` with the selected suggestion names only, at most `cfg.max_display_names` long or
  an empty list for `inert`, through a temporary file and a rename; kept rows are listed in the block
  but never recorded, so the display cannot call a pinned skill a suggestion; an identical record skips the write,
  `show_suggestions` false removes an existing record instead of writing, and the newest write is the
  record both clients read
- errors: never raises; a record write failure logs once and leaves the returned payload alone
- invariants: every top-level key of `request` survives the rewrite; the returned payload is either
  unchanged or both stripped and appended, never one of the two; a non-`chat_completions` payload is
  returned untouched; the payload with the display enabled is deep-equal to the same rewrite with the
  record writer replaced by a no-op, message count and roles included, and no record field appears in
  any message, so the display costs no tokens; the record holds names only and never the user's text;
  deterministic for the same input and index
- state: one record file at `suggestions_path`, read by M12 and M14 and never added to a request

### C10: DSH http embedder

- kind: function
- owner: M10
- stability: stable
- satisfies: R9, R6

```typescript
export interface HttpEmbedderOptions {
  readonly baseUrl: string
  readonly modelId: string
  readonly timeoutMs: number
  readonly path?: string
}
export function createHttpEmbedder (opts: HttpEmbedderOptions): TextEmbedder
```

- pre: `baseUrl` is reachable, or the router keeps its ONNX embedder as the fallback
- post: `embedOne(text)` resolves to one vector for one text, the same shape `createBgeEmbedder` returns
- errors: rejects rather than returning a wrong vector, which `shortlist` already turns into no suggestions
- invariants: one HTTP request per text, no batching, no retry
- state: one request per call, no connection reuse policy of its own

## Data flow

A provider request reaches `apply_llm_request_middleware`, which calls M9.rewrite.
M9 reads the last user-role text, asks M3 to remove the catalog, asks M4 for the
index (which calls M5 on a cold build), calls M5 for the query, calls M6 for the
shortlist documents, calls M7 and M8, and returns the rewritten payload. It also
writes the display record through M11, which M12 serves to the desktop half and
M14 reads in the TUI. The service M1 is reached only through M5 and through M10 on
the DSH side.

## State and concurrency

The index is shared process state: one build per fingerprint, guarded by a lock so
two concurrent requests do not embed the library twice. The health flags in M5 and
M6 are process state with a monotonic timestamp. Nothing in the plugin mutates
persisted history; the rewrite is applied to the outgoing payload only, which
Hermes replays byte-for-byte on later turns, so the rewrite must stay
deterministic for the same payload and index. The display record is the only state the two
clients share: the rewrite rewrites it atomically and the last writer wins.

## Brownfield delta

| Path | Existing fact | Proposed change | Evidence |
|------|---------------|-----------------|----------|
| `agent/prompt_builder.py` | Renders the catalog into the system prompt; 47,981 characters measured on this profile | untouched; the plugin parses and removes the rows it renders, so plugin skills and gated rows keep their canonical names | `build_skills_system_prompt()` measured 427 rows |
| `tools/skills_tool.py` | `skill_view` refuses a disabled skill | untouched; nothing is disabled | `tools/skills_tool.py:629` |
| `hermes_cli/middleware.py` | Applies plugin request middleware and takes the returned `{"request": ...}` | used as the only entry point | `hermes_cli/middleware.py:88` |
| `packages/dsh-skill-router/src/index/bge-embedder.ts` | Loads ONNX in process from `modelPath` | an HTTP embedder implements the same contract beside it | `src/index/bge-embedder.ts:85` |
| `~/.scripts/decider-rerank-server.py` | Serves Cohere rerank on 8010 | used unchanged | `GET /health` returns `tasksource-decider-nano` |

## Verification hooks

| Contract | Check | Command |
|----------|-------|---------|
| C1 | endpoint shape test plus a live health probe | `python3 -m pytest tests/test_embed_server.py -q` and `curl -s localhost:8012/health` |
| C2 | bounds and unknown-key notes | `python3 -m pytest tests/test_config.py -q` |
| C3 | strip test on a real-shaped message with the heading, prose and tags, partition boundaries, the single-query variant, the demoted `[names only]` line, and a mutation that disables the strip | `python3 -m pytest tests/test_catalog.py -q` |
| C4 | cache hit avoids embedding; duplicate names collapse to the first occurrence; a changed row, model id or endpoint changes the fingerprint | `python3 -m pytest tests/test_library.py -q` |
| C5 | None on 500, timeout and length mismatch | `python3 -m pytest tests/test_embed_client.py -q` |
| C6 | order restored from `results[].index`, and the document text equals the DSH format | `python3 -m pytest tests/test_rerank_client.py -q` |
| C7 | cutoff and cap boundaries plus DSH parity on fixed scores | `python3 -m pytest tests/test_select.py -q` |
| C8 | block shape, clipping, kept rows listed and the empty-block case | `python3 -m pytest tests/test_inject.py -q` |
| C9 | unchanged payload on every failure; never half-applied | `python3 -m pytest tests/test_rewrite.py -q` |
| C10 | DSH suite with an endpoint-sensitive fingerprint plus a boot against the live endpoint | `npm test` in `packages/dsh-skill-router` |
| C9 | display record shape, atomic write, dedupe, removal on `show_suggestions` false, the caps with 50 names, newest-wins after two sessions, a payload deep-equal to the no-display build, the route answer for a record and for a missing record, and the desktop admission lint | `python3 -m pytest tests/test_display.py -q` and `hermes plugins validate .` |

## Traceability

| Requirement | Contracts | Tickets |
|-------------|-----------|---------|
| R1 | C3, C9 | T2 |
| R2 | C4, C9 | T3, T5 |
| R3 | C7, C8, C9 | T4 |
| R4 | C8 | T4 |
| R5 | C5, C6, C9 | T2, T3, T4, T5 |
| R6 | C1, C5, C6, C10 | T1, T3, T4, T6 |
| R7 | C2 | T2, T7 |
| R8 | C4 | T3 |
| R9 | C10 | T6 |
| R10 | C1, C2 | T7 |
| R11 | C2, C3, C4, C8, C9 | T2, T3, T4, T5 |
| R12 | C9 | T5, T8 |

## Uncertainties and tradeoffs

| Id | Uncertainty | Why unresolved | Settle by | Owner |
|----|-------------|----------------|-----------|-------|
| U1 | Whether a second system-role message is accepted on every provider route | needs a live probe against the configured routes | run one `hermes -z` turn per route with the block appended and read the response | orchestrator |
| U2 | Whether removing 12.4k tokens of system prompt changes prompt-cache accounting | needs a before and after turn comparison on one route | compare cached_read_tokens across two turns | orchestrator |
| U3 | Whether Hermes ever rebuilds the catalog inside one session | needs a live probe; the design assumes it is frozen per session | change a skill file mid-session and compare the next request's system message | orchestrator |

## Revision log

| Date | Change | Trigger | Approved by |
|------|--------|---------|-------------|
| pre-approval | Review round 7 fixed two implementation findings: the manifest must declare `provides_middleware` for the pinned validate check, and demoted `[names only]` catalog names are parsed and always kept so the strip cannot drop them | implementation findings, review round 7 | pending round 7 |
| pre-approval | Review round 4 fixed six findings: the index is routed-only, `maxSuggestions` caps suggestions only, the record carries selected names only, M8 and M9 allow the dependencies C8 and C9 need, and R11 is traced through the block with an end-to-end criterion | review round 4 | pending round 5 |
| pre-approval | Review round 3 fixed ten findings: middleware kwargs are unfiltered, the record writer moved to T5, the strip removes the whole skills section and the block lists eager rows, both clients refuse redirects, T6 owns `embedding-index.ts` and names the profile row, T2 and T5 declare a sequential handoff, C4 counts unique names, traceability and README checks are extended, and the desktop-toggle message is gone | review round 3 | pending round 4 |
| pre-approval | Review round 3 fixed the display delta: C2 now carries the three display config fields, the record gained `schema`, `maxDisplayNames`, the removal on `showSuggestions` false and newest-wins semantics, the zero-token test compares against the no-display build, and the settled uncertainties were dropped | review round 3 on the display delta | pending round 3 |
| pre-approval | Display of the suggested names added (S10, D12, R12, T8) as a names-only record written by the rewrite, served by the plugin backend and read by a desktop pane and an Ink TUI widget; it lives inside C9 to keep the contract count at the template limit | operator request, mid-run addition | pending round 3 |
| pre-approval | Eager and deferred skill partition added (D11), from the operator's request to mirror DSH's eager/deferred split; index covers routed rows only | operator request, review round 2 | pending round 2 |
| pre-approval | Index built from the stripped catalog rows instead of a filesystem scan; payload preservation, `chat_completions` scope, direct cutoff comparison, TLS bound and DSH document format written into the contracts | review round 1, findings 1 to 10 | pending round 2 |
