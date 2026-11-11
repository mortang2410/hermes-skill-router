# Technical documentation

This is the full reference for hermes-skill-router: the request rewrite, the suggestion record, both
display protocols, every setting with its bound, both endpoint protocols, failure behaviour and the
measurements. [README.md](README.md) is the short version.

## How routing works

The rewrite runs six steps for one Chat Completions request:

1. Strip the skills section and partition its rows with `eagerSkills` and `deferredSkills`.
2. Build or load the index over the routed rows only. The index is the routed rows' names and
   descriptions, embedded through the embeddings endpoint.
3. Build the query from the newest typed user message and the typed user messages before it, then
   embed it and fuse three channel orders: the `shortlistSize`-independent cosine order over the
   index, the BM25 order over the same documents, and the association order over the learned store.
   Fuse them with reciprocal rank fusion, apply the bounded recency bump, and only then cut to
   `shortlistSize`. The query is `<newest message> Context: <older message> <older message>`, with
   the older messages joined oldest first; only typed user messages count, so a system message, an
   assistant message and an earlier injected block never join it. The preceding context is capped at
   `RECENT_CONTEXT_CHARS = 10000` characters, counted over whole messages from the newest backwards
   and stopping at the first message that would cross the cap, so a message is never cut. The newest
   message is not part of the cap and is embedded in full, and with no typed message before it the
   query is exactly the newest message.
4. Send those candidates to the rerank endpoint with the same query string, one document per
   candidate, in the rerank document format `documentPrefix + name + " - " + description`.
5. Resolve every reranked candidate's read path through the host lookup the `skill_view` resolver
   uses, recover the names the request's own earlier blocks already suggested, then keep the
   candidates that are not suppressed, score at or above `cutoff` and carry a path, sort them by
   score, and cap them at `maxSuggestions`. A cap of `0` means unlimited.
6. Render the block with those suggestions and the kept rows, and append it as one system message.

The model can also search the catalog the prompt showed. The plugin registers one tool,
`skill_router_search`, which ranks the routed half of the same partition through the same embed,
shortlist, rerank, cutoff and cap, and returns `{"skills": [{"name", "description", "path"?}]}`. The
prompt's shortlist is already in front of the model, so the tool description tells it not to call the
tool for those rows. A search runs inside `routingTimeoutMs` and finds the index the turn already
built; it writes the same display record with origin `search`, so both surfaces show it. The tool
never searches the skill library for candidates: it resolves every reranked candidate's `SKILL.md`
path through the host lookup before the cut, at most `shortlistSize` lookups per call, and drops a
name the lookup refuses before the cap counts rows, so the next admissible row takes its slot. A
session with no parsed catalog answers `no-catalog`, and with `showSuggestions` off the record stays
hidden exactly as for a turn.

The persisted index at `cachePath` is keyed by a fingerprint of every routed row's text and full
description, `embedModelId`, the document format version, and `embedBaseUrl + embedPath`. A change to
a routed skill, the model, the endpoint, or the description clip rebuilds it. A change to a kept row
alone leaves it intact, and the cache is reused across sessions and across turns. The catalog is
frozen per Hermes session, so a session start normally reuses the cache.

A catalog that arrives with fewer than half the rows of the saved index is a different case: the
plugin treats it as a truncated view rather than an edit. The build is refused before it embeds
anything, the saved index and its vectors stay in place, and the turn routes from them. One warning
per refused catalog names the arrived count, the saved count and the cache path, and it names the
recovery: remove the cache file to force a rebuild. The trade is deliberate: a real catalog shrink of
more than half keeps routing against the saved list until that file is removed.

### The three channels

The semantic channel is the cosine order over the index vectors, and it is the rollback order: with
`fusionMode: cosine` the plugin returns it untouched.

The BM25 channel indexes the exact document text the embedder sees, so both channels read one
document format. Tokens are lowercased, split on every character that is not a letter or a digit,
dropped below two characters, dropped when they are in one short English stopword list, and never
stemmed. Scoring is Okapi BM25 with `k1` 1.2 and `b` 0.75, with document frequency and average
document length computed over the same catalog. The constants are module constants, not settings.

The term index is built during an index build and written atomically to `bm25.json` beside
`cachePath`, so a restart does not re-tokenise the catalog. It carries a digest over the document
texts and the scoring constants, and a loader refuses a file whose digest does not match the rows in
hand. It is loaded once per process. A missing, corrupt or foreign term index contributes no BM25
ranks, logs one warning, and leaves the other channels routing. A term index that cannot be built
does not fail the index build: the vector index stays usable, the BM25 channel is disabled for that
pass with one warning, and the remaining channels are fused.

The association channel reads the learned load store at `associationDbPath` open read-only. It
normalises the composed query with the module's own term rule, reads the decayed weights for those
terms, keeps only `(term, skill)` rows whose support clears the floor of three distinct prompts,
excludes ambient and soft-deleted skills, maps skill ids to catalog names and contributes ranks only
for names the catalog carries. A missing or unreadable store contributes no association ranks,
logs one warning, and leaves the other channels routing. No channel is a precondition.

The three orders are fused with reciprocal rank fusion: `sum(weight / (20 + rank))` over the channel
ranks, with weights 1.0 for semantic, 0.5 for BM25 and 1.0 for associations, and ties broken by
name. Only ranks are fused, never the three scores, so a channel whose scores live on another scale
cannot dominate the list. A disabled channel and a zero weight produce the same order.

The recency bump is a boost, not a fourth channel. It moves a candidate up by
`round(frecencyCapPositions * min(usage, 1))` places, once, and it only reorders candidates a
channel already returned. The usage score is `sum over distinct session turns of
0.5 ** (age_days / frecencyHalfLifeDays)`, so three loads in one turn count once, and ambient and
soft-deleted skills are absent. The bump is gated on a resolvable catalog path: a candidate whose
path the host lookup does not resolve keeps its fused position, which is the same gate the reference
router applies to its usage scores. The shortlist is cut to `shortlistSize` only after the bump, so
a candidate just below the cut can still enter it. The defaults are 14 days and 5 positions.

### Fusion modes

`fusionMode: live` is the default: the shortlist is the fused, bumped, cut order.

`fusionMode: shadow` computes the same fused, bumped, cut order and logs one line per turn carrying
each candidate's three channel ranks, its movement and its final position, then returns the cosine
order cut to `shortlistSize`. It is the side-by-side comparison mode. The line carries skill names
and ranks only, never request text.

`fusionMode: cosine` returns the cosine order cut to `shortlistSize` with no BM25 score, no
association store open and no bump. It is the single rollback switch: no rebuild and no config
change beyond this key.

### The association backfill

The association store is filled from the Hermes session store by the module's own backfill, run as a
command from the repository root:

```sh
python3 -m hermes_skill_router.associations --max-sessions 200
```

The command reads `~/.hermes/state.db` read-only, writes only the association store, and accepts a
bounded session count as above; omit `--max-sessions` to scan the whole store. `--db` names a
different store file, `--state-db` a different session store, and `--catalog` an optional JSON list
of catalog names (or an index file carrying a `names` list). It prints a JSON report with the
session, prompt and load counts.

Run it after the catalog has been indexed and after a batch of work that loaded skills you want the
channel to learn from; it is safe to rerun because uniqueness in the store, not a check, is what
makes a second run insert nothing. Until it has run, the store is missing or empty and the
association channel contributes no ranks, so the fused order is the semantic and BM25 channels only.
That costs no correctness: the channel is a ranking signal, not a precondition, and the turn logs
one warning naming the store path.


## The request rewrite and the prompt block

The plugin removes the whole `## Skills` section from the first system message: the heading, the
prose before and after the `<available_skills>` block, the block itself, the trailing sentence, and
any single-line parenthetical notes the renderer appends after the block.

The catalog carries two row shapes:

- `- name: description` rows.
- `category [names only]: a, b` lines, which a coding posture demotes to a list of names with no
  descriptions.

Every row is partitioned once per request into kept rows and routed rows:

- Kept rows stay listed in the block that carries them and are never suggested: every name in `eagerSkills`,
  every row a non-empty `deferredSkills` list does not name; and every name on a demoted
  `[names only]` line, even when `deferredSkills` names it. A demoted name is always listed by name
  only and never routed, because the renderer never emitted its description.
- Routed rows are everything else. Only routed rows enter the embeddings index and can be
  suggested.

Hermes also appends a parenthetical note for shadowed skills when two tiers carry a skill with the
same name and relative path. That note names copies that `skill_view` cannot load, so the plugin
removes the note with the section and drops those names from the block. They were not loadable before
the plugin ran either.

The plugin then appends one extra `system` message at the end of the request. It has two fixed
headers, and both open with the same marker line. The rewrite recognises its own block by that line,
so a replayed request is not injected twice with either header:

```
Skills routed for this request
```

The strip records which prose variant it removed, and that choice picks the header. A normal turn
uses the normal header:

```
Skills routed for this request
These are the skills that may help; the rest of the catalog is not listed. Scan them before replying. If one matches or is even partially relevant, load it with skill_view(name) and follow its instructions: skills carry API endpoints, tool-specific commands, conventions and pitfalls that outperform a general approach, so load one rather than guess, and load one that defines how a task should be done here even when you already know the basics. Call skills_list when none of these fits, to see every skill available. Only proceed without loading a skill if genuinely none are relevant.
```

A one-shot session uses the one-shot header instead, because its catalog prose carries the
single-query restriction:

```
Skills routed for this request
These are the skills that may help; the rest of the catalog is not listed. Load one with skill_view(name) only when it carries domain knowledge you lack for this task: an API, a tool's commands, or a project's conventions. Do not load general process skills (testing, debugging, review methodology) for work you already know how to do, and do not create or edit skills: this is a one-shot run with no later session to reuse them. Call skills_list when none of these fits, to see every skill available.
```

The one-shot header keeps that restriction: do not load general process skills for work you already
know, and do not create or edit skills, because a one-shot run has no later session to reuse them.
The normal header invites a load for any partially relevant skill. Both headers carry the skill-use
guidance the removed prose provided, including the call to `skills_list` when none of the suggestions
fits, and both name the loader tools `skill_view` and `skills_list`.

Below the header the block carries the selected suggestions, best first. Every suggestion carries its
name, its description and the absolute read path the host lookup resolved. The opening block of a
compaction window adds the pinned rows in their own section, and a later turn in that window carries
the suggestions only:

```
Suggestions:
- <suggested name>: <description>
  path: <absolute path to the skill's SKILL.md>
- <suggested name with no description>
  path: <absolute path to the skill's SKILL.md>

Pinned skills:
- <pinned name>: <description>
  path: <absolute path to the skill's SKILL.md>

Also listed:
- <kept name>: <description>
```

Every suggestion and every pinned row carries the file `skill_view` opens. A suggestion whose path
does not resolve is not emitted, and the selector has already dropped it before the cap counted rows,
so the next admissible candidate takes the slot. A pinned row whose path does not resolve is omitted
with one warning per process. Kept rows that are neither pinned nor suggested keep their earlier shape
under `Also listed:`: `- name: description`, or `- name` for a demoted `[names only]` row, and they
carry no path. Descriptions are collapsed to one line and clipped to `maxDescriptionChars`; a
suggestion the index no longer describes is listed as `- name` and still carries its path.

The cap is applied once, in the selector. `maxSuggestions` defaults to `10` and accepts `0` to `50`,
and `0` means unlimited: the renderer lists every row it is handed. The same selector drops a name
that the request's own earlier blocks already suggested, so a later turn in the same compaction
window never repeats it. A request whose messages carry no earlier block is the window's opening block
and lists the pinned rows; a request carrying an earlier block omits them; a request whose earlier
block a compaction removed starts a new window and lists them again. A name that is both kept and
suggested appears only once, as a suggestion.

The first line of the block is a marker, not the removed `## Skills` heading, so a request dump shows
whether the plugin ran. The rewrite reads a prior block back with the renderer's own recogniser, which
returns the names in the suggestions section only, never a pinned or also-listed name.

The strip and the injection are atomic. The plugin removes the section only when it can replace it
with a non-empty block, so a request with no suggestion and no kept row keeps its original section
byte for byte. The first line of the block is a marker, not the removed `## Skills` heading, so a
request dump shows whether the plugin ran.

The plugin rewrites only OpenAI-compatible Chat Completions requests, with
`api_mode: "chat_completions"`. Codex Responses (`api_mode: "codex_responses"`) carries the prompt in
`instructions` and `input`, and Anthropic Messages (`api_mode: "anthropic_messages"`) carries it in
`system`, so a `messages` rewrite would do nothing or corrupt the call. Those two modes pass through
untouched and keep the catalog; the unsupported mode is logged once.

The selected suggestion names are also written to a names-only display record after the payload is
built. The record is never part of a request.

## Suggestion record schema

The record is one JSON file at `suggestionsPath`:

```json
{
  "schema": 2,
  "at": "2026-11-10T07:20:31Z",
  "session_id": "<session id when the dispatcher supplies one, else an empty string>",
  "turn_id": "<turn id when the dispatcher supplies one, else an empty string>",
  "status": "routed",
  "origin": "turn",
  "names": ["github-pr-workflow", "code-review"]
}
```

Fields:

- `schema` is 2. A schema 1 file still reads, with origin `turn`.
- `at` is the decision time in UTC, written as `YYYY-MM-DDTHH:MM:SSZ`.
- `session_id` and `turn_id` carry the ids the dispatcher supplies, and an empty string otherwise. A
  search record never carries a turn id.
- `status` is `routed` when the block was injected and `inert` when the catalog stayed in place. An
  inert record carries no names.
- `origin` is `turn` for the automatic pass and `search` for a model call to the search tool.
- `names` holds the selected suggestion names only, capped at `maxDisplayNames`. A kept row is never
  recorded, so the display cannot present a pinned or deferred-excluded skill as a suggestion.

The record never carries request text. The newest record wins, and it carries the session and turn
ids the dispatcher supplies so a second session or a fork replacing it is visible on screen when the
ids are present.

Write rules:

- A write goes through a temporary file in the target directory and one `os.replace`, so a reader
  never sees a half-written record and a crashed writer leaves nothing behind.
- A record identical to the one on disk is not rewritten. Identity ignores `at`, so a turn that
  routes to the same names keeps the timestamp of the decision that first produced them. `origin` is
  part of that identity, so a search with the same names still rewrites the marker.
- `showSuggestions: false` stops the writes and removes an existing record, which turns both
  surfaces idle. When that removal is refused, the plugin writes an inert, empty record instead, so
  the file-reading widget shows no names.
- Every failure is swallowed and logged once per process. The rewrite result never depends on the
  display, and an unreadable file reads as idle rather than raising in a reader.

## Display surfaces

The last routed decision is readable in two places, and neither place costs a token. The plugin
writes the record after it has built the payload it returns, and nothing from the record is ever
added to a request.

### Desktop pane

`Skills routed` is a collapsed-by-default pane on the right side, 260 px wide. Its heading carries the
count, its list shows the names, and for a routed record one muted caption line reads
`turn <turn_id> · <local HH:MM>` from the record's own `turn_id` and `at`, or
`model search · <local HH:MM>` when a model search produced the names. The caption is shown only for
a routed record, so a stale turn cannot sit under an inert or idle state.

It refreshes on mount and when the plugin broadcasts
`plugin.hermes-skill-router.suggestions.updated`. It reads the plugin's own route,
`GET /api/plugins/hermes-skill-router/suggestions`, which returns the record or
`{"status": "idle", "names": []}` when the file is absent or unreadable. The route never returns the
user's text. The pane is copied into `$HERMES_HOME/desktop-plugins/hermes-skill-router/` when the
plugin folder is in a plugins root.

### TUI widget

Hermes exposes no plugin display hook in the Ink TUI, so the plugin ships an ambient user widget
instead. Install it with a symlink:

```bash
mkdir -p "${HERMES_HOME:-$HOME/.hermes}/tui-widgets"
ln -sfn /path/to/hermes-skill-router/tui/skill-router.mjs "${HERMES_HOME:-$HOME/.hermes}/tui-widgets/skill-router.mjs"
```

This is `$HERMES_HOME/tui-widgets/`, which defaults to `~/.hermes/tui-widgets/`. In `hermes --tui`,
run `/skills-routed` to dock the widget and run it again to collapse it. The command accepts a path
argument, `/skills-routed <path>`, and defaults to
`~/.cache/hermes-skill-router/suggestions.json`. The widget reads that file every 500 ms. For a
routed record it carries the count in the `Suggestions` heading, lists the names one per line, and
paints one muted caption line from the record's own `origin`, `turn_id` and `at`:
`turn <turn_id> · <local HH:MM>` for a turn record, and `model search · <local HH:MM>` for a record a
model search wrote, which has no turn id. The caption is shown only for a routed record. A missing,
corrupt or `inert` record shows one idle line instead. It uses theme tones only and never reads
plugin config, which is why the path is an argument.

### Enabling the display

The display has two independent enable steps:

1. The profile's `plugins.enabled` entry enables the Python backend, including the route.
2. The plugin's own toggle in the desktop app's Capabilities panel enables the desktop half. Hermes
   never calls the desktop half's `register()` when that toggle is off, so the pane does not exist.
   With the backend disabled instead, the pane shows the enable line for the missing route.

The three display keys are `showSuggestions`, `suggestionsPath` and `maxDisplayNames`, documented in
the config table below.

The classic prompt_toolkit TUI exposes no plugin display surface, so it is not covered.

## Configuration

Every value in this section is read from `plugins.entries.hermes-skill-router.settings` in the
profile config. An out-of-bound or malformed value falls back to its default and appends a note
naming the key. Each rejected setting is logged once per process, on the first load that rejects it,
and an endpoint URL in that message is reduced to its scheme and host, so credentials never reach the
log. An unknown key appends a note and is otherwise ignored. A malformed value never aborts a
request.

### Eager and deferred lists

`eagerSkills` and `deferredSkills` split the catalog into a pinned list and a routable pool. The
partition is computed per request from the catalog rows.

- `eagerSkills` names skills that are listed once, in the opening block of a compaction window,
  under `Pinned skills:`, each with its name, description and read path, and that are never routed. A
  later turn in the window carries the suggestions only, and the next window opens with them again.
  Example: `eagerSkills: [writing-for-agents]` keeps that skill in that opening block and out of the
  index, the rerank call and the display record.
- `deferredSkills` names the routable pool. An empty list, the default, means every skill that is not
  eager is routable. A non-empty list restricts routing to the named skills and leaves every other
  skill listed. Example: `deferredSkills: [github-pr-workflow, code-review]` suggests only those two
  and lists every other non-eager skill under `Also listed:`.
- A name in both lists is eager, and the plugin warns once per process for it. A configured name that
  the catalog does not carry is ignored, and an unknown eager or deferred name warns once per
  process. A name on a demoted `[names only]` line is always listed and never routed, even when
  `deferredSkills` names it.

### Every key

<!-- config-defaults:start -->
| Key | Default | Bound | Effect |
| --- | --- | --- | --- |
| `enabled` | `true` | boolean | Master switch. False leaves every request untouched. |
| `eagerSkills` | empty | list of skill names | Skills listed once in each compaction window's opening block under `Pinned skills:`, with name, description and read path, and never routed. A name in both lists is eager. |
| `embedBaseUrl` | `http://127.0.0.1:8012` | http for a loopback host only, https elsewhere, no credentials, query or fragment | Embeddings endpoint root. A rejected value falls back to the default with one warning. |
| `embedPath` | `/v1/embeddings` | path starting with `/` | Path appended to `embedBaseUrl`. |
| `embedModelId` | `bge-m3` | non-empty string | `model` field sent to the embeddings endpoint. Part of the index fingerprint, so a change rebuilds the cache. |
| `embedTimeoutMs` | `6000` | 100..60000 | Per-request embeddings timeout. Each index-build chunk of at most 16 texts gets its own budget. |
| `rerankBaseUrl` | `http://127.0.0.1:8010` | http for a loopback host only, https elsewhere, no credentials, query or fragment | Rerank endpoint root. A rejected value falls back to the default with one warning. |
| `rerankPath` | `/v1/rerank` | path starting with `/` | Path appended to `rerankBaseUrl`. |
| `rerankModel` | `tasksource-decider-nano` | non-empty string | `model` field sent to the rerank endpoint. |
| `rerankProvider` | `cohere` | `cohere` | Request and response shape. Cohere is the only implemented provider. |
| `documentPrefix` | `This skill supplies a procedure, tool, or domain-specific instructions that the agent would actually need to carry out the current task correctly. Skill: ` | non-empty string | Exact text before `<name> - <description>` in a rerank document. The default matches the format the shared rerank service parses. |
| `deferredSkills` | empty | list of skill names | Routable skills. Empty means every non-eager skill is routable; a non-empty list restricts routing to the named skills and lists the rest. |
| `rerankTimeoutMs` | `20000` | 100..60000 | Per-request rerank timeout. One rerank call carries the whole shortlist. |
| `routingTimeoutMs` | `30000` | 1000..60000 | Whole routing pass budget, from the query embed through the rerank reply. It starts after the index is loaded or built, so the cold build is never cut, and each per-request timeout is capped at the time left. On expiry the request keeps the bare catalog and gets no suggestions. |
| `shortlistSize` | `50` | 1..383 | Candidates carried from embeddings into rerank. |
| `cutoff` | `0.01` | 0..1 | Rerank `relevance_score` a candidate needs, compared inclusively. |
| `maxSuggestions` | `10` | 0..50 | Cap on suggested rows, applied once in the selector. Zero means unlimited. |
| `fusionMode` | `live` | `live`, `shadow` or `cosine` | Fusion mode. `live` returns the fused order, `shadow` logs the diagnostics and returns the cosine order, `cosine` is the rollback switch with no channel read and no bump. |
| `channelsSemantic` | `true` | boolean | Read the semantic (cosine) channel. False is the same input as a zero weight for that channel. |
| `channelsBm25` | `true` | boolean | Read the BM25 channel. A term index that is missing or cannot be built contributes no ranks either way, with one warning. |
| `channelsAssociations` | `true` | boolean | Read the association channel. The store stays read-only on the turn path. |
| `frecencyHalfLifeDays` | `14` | 1..3650 | Half-life in days of the usage decay behind the recency bump. |
| `frecencyCapPositions` | `5` | 0..50 | Most shortlist positions the recency bump can move one candidate. Zero disables the bump and keeps the channels fused. |
| `maxDisplayNames` | `12` | 1..50 | Names the desktop pane and the TUI widget list. The count stays visible when collapsed. |
| `maxDescriptionChars` | `400` | 3..4000 | Description characters kept per row, in the block and the embedding text. Part of the index fingerprint. |
| `cachePath` | `~/.cache/hermes-skill-router/index.json` | path, empty disables persistence | Persisted index. Empty means each process builds its own index and keeps it in memory. A routed catalog with fewer than half the rows of this file is refused and the file is kept, so remove it to force a rebuild. |
| `associationDbPath` | empty | path, empty means beside the index | Association store file. Empty puts `associations.db` beside `cachePath`; with `cachePath` also empty, the store is not opened. A missing or unreadable store contributes no association ranks with one warning. |
| `showSuggestions` | `true` | boolean | Let the display read the last suggestion names. False stops the record and removes an existing one. |
| `suggestionsPath` | `~/.cache/hermes-skill-router/suggestions.json` | path, empty disables the record | Display record both surfaces read. Names only, never request text. |
| `healthCheckMs` | `60000` | 0..3600000 | Minimum gap between probes of an endpoint that failed. Zero probes every request. |
<!-- config-defaults:end -->

### Endpoint URL rule

A plain `http` URL is accepted only for a loopback host, which means `127.0.0.1`, `localhost` or
`::1`. Every other host must use `https`. A URL carrying credentials, a query or a fragment is
rejected and falls back to its default with one warning, because the request path is appended to the
root. Both HTTP clients refuse to follow a redirect, so a 3xx fails the call instead of sending the
request text to another host. The defaults point at loopback: `http://127.0.0.1:8012` for embeddings
and `http://127.0.0.1:8010` for rerank.

### Data disclosure

On every routed request the plugin sends the text of the request's last user message to the
embeddings endpoint. On a cold index build it sends the name and description of every routed skill
there too. It sends the shortlisted skill names and descriptions to the rerank endpoint. Nothing else
is sent to either endpoint. Request text is never written to the display record, is never written to
the index cache and never appears in a warning. The one exception is the decision line described
below: the plugin's own log carries the query text and the scores of each routing pass.

### The decision line

The plugin writes one info line beginning `skill router: query=` for every turn whose ranking pass
completes. The line carries the query text as a JSON string, every shortlisted candidate as
`name:score` including the candidates the cutoff rejected, the cutoff, the suggested names with their
probabilities, or `-` when no candidate cleared the cutoff, and the embed, rerank and total stage
times in milliseconds. The scores are the values the selector compared with the cutoff, so a name
that is missing from the block can be explained without replaying the request. A turn whose rerank
stage failed, timed out or ran out of its budget writes no decision line: its one warning names the
reason and, like every warning, carries no user text.

## Endpoint protocols

Both endpoints are plain HTTP or HTTPS with JSON bodies. The plugin sends no API key and no
`Authorization` header, so an endpoint that needs credentials must sit behind a proxy that adds them.
Any server that answers these shapes works, including the bundled embeddings service and the author's
rerank service.

### Embeddings

The plugin sends `POST <embedBaseUrl><embedPath>`, which defaults to
`http://127.0.0.1:8012/v1/embeddings`, with a batch of texts. The query is one text; an index build
sends at most 16 texts per request.

Request:

```http
POST /v1/embeddings HTTP/1.1
Host: 127.0.0.1:8012
Content-Type: application/json

{"model": "bge-m3", "input": ["first text", "second text"]}
```

Response:

```json
{
  "model": "bge-m3",
  "data": [
    {"index": 0, "embedding": [0.0123, -0.0456, 0.0789]},
    {"index": 1, "embedding": [0.0034, 0.0567, -0.0891]}
  ]
}
```

The response must carry one entry per input, in input order. The plugin reads `data` in that order
and does not re-sort by `index`. Every vector must be a non-empty list of numbers and all vectors in
one response must share one width. A short, long, ragged or non-numeric response fails the call, and
no partial result is used. The bundled service returns 1024-wide L2-normalised vectors, but the
plugin only needs a consistent width because it computes cosine similarity itself.

Health check:

```http
GET /health HTTP/1.1
Host: 127.0.0.1:8012
```

```json
{"status": "ok", "model": "bge-m3", "dim": 1024}
```

The plugin does not call `/health`; it treats a failed request as the health signal and remembers the
failure for `healthCheckMs`. The route exists for operators and for the bundled unit's own readiness.
While the bundled service loads its model it answers `{"status": "loading"}` and returns 503 for
embeddings requests; the plugin treats that as a failure and leaves the request unchanged.

### Rerank

The plugin sends `POST <rerankBaseUrl><rerankPath>`, which defaults to
`http://127.0.0.1:8010/v1/rerank`, with the shortlist. `top_n` is the number of documents in the
call.

Request:

```http
POST /v1/rerank HTTP/1.1
Host: 127.0.0.1:8010
Content-Type: application/json

{
  "model": "tasksource-decider-nano",
  "query": "<the last user message text>",
  "documents": [
    "This skill supplies a procedure, tool, or domain-specific instructions that the agent would actually need to carry out the current task correctly. Skill: code-review - <description>",
    "This skill supplies a procedure, tool, or domain-specific instructions that the agent would actually need to carry out the current task correctly. Skill: github-pr-workflow - <description>"
  ],
  "top_n": 2
}
```

Response:

```json
{
  "results": [
    {"index": 1, "relevance_score": 0.873},
    {"index": 0, "relevance_score": 0.041}
  ]
}
```

The response must carry exactly one result per document. Results may arrive in any order; `index` is
the zero-based position of the document in the request, and the plugin restores input order from it.
`relevance_score` is a finite probability. A missing, duplicated or out-of-range index, a non-numeric
score, or a result count that differs from the document count fails the call, and no partial ranking
is used. The selector compares the score directly and inclusively with `cutoff`; it applies no
second softmax.

### Timeouts and health

| Setting | Applies to | Default | Bound |
| --- | --- | --- | --- |
| `routingTimeoutMs` | One whole routing pass: the query embed, the shortlist, and the rerank reply | 30000 ms | 1000..60000 |
| `embedTimeoutMs` | One embeddings request, including each index-build chunk | 6000 ms | 100..60000 |
| `rerankTimeoutMs` | One rerank request carrying the whole shortlist | 20000 ms | 100..60000 |
| `healthCheckMs` | How long a failed endpoint is not probed again | 60000 ms | 0..3600000 |

`routingTimeoutMs` bounds one whole routing pass. The budget starts after the skill index is loaded
or built, so it never covers that build: the build issues 27 chunked embeddings requests and takes
about 38 seconds here, and it already fails through its own path by returning no index instead of
being cut off. Each per-request timeout is capped at the milliseconds the budget has left, so a slow
embed or rerank is cut when the budget ends. The cap bounds the wait for an answer, not the transfer
of one: a server that trickles bytes within every socket timeout can still run past the budget. When
the budget runs out the plugin abandons the pass: the request keeps the bare catalog and gets no
suggestion block, the same inert fallback every other failure uses, and one warning names the budget.

The 30000 ms default rests on two facts. The two per-request timeouts allow a 26 second pass at
worst, 6000 plus 20000, so 30000 sits above that ceiling and a working but slow path is never cut. It
also has headroom over the measurements recorded on this machine against its live endpoints: a steady-state single-text embed
took 236 to 306 ms, a 50-document rerank took 2158 to 2255 ms, and the first embed after an endpoint
restart took 2700 ms, so a cold but healthy pass cost about 5 seconds, 2700 plus 2255. Those are the
slow case rather than a constant: an earlier live run on the same host measured a full warm rewrite
at 0.75 to 0.79 seconds, and a probe on 2026-11-10 measured a one-text embed at 84 ms and a
50-document rerank at 287 ms. On a host whose endpoints are slower than the recorded figures, raise
the budget above your own measured cold pass before lowering anything else.

## The bundled embeddings service

The repository ships one embeddings server for BGE-M3 ONNX on CPU. Its model directory must contain
`onnx/model.onnx` and `tokenizer.json`, and the Python environment needs `onnxruntime`, `tokenizers`,
`fastapi`, `uvicorn` and `numpy`. From the checkout root:

```bash
HSR_MODEL_DIR=/path/to/bge-m3 HSR_PORT=8012 python3 -m service.embed_server
```

The service listens on `127.0.0.1:8012` by default. It reads its settings from `HSR_*` environment
variables: `HSR_MODEL_DIR`, `HSR_PORT` (default 8012), `HSR_MODEL_ID` (default `bge-m3`) and
`HSR_MAX_LENGTH` (default 512). While it loads its model it answers `{"status": "loading"}` and
returns 503 for embeddings requests.

The tracked systemd user unit runs the same command, but it carries the author's model path in
`HSR_MODEL_DIR`, so override that value for your host, for example with
`systemctl --user edit hermes-skill-router-embed.service`. Then link and start the unit:

```bash
ln -sfn /path/to/hermes-skill-router/service/hermes-skill-router-embed.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now hermes-skill-router-embed.service
```

The rerank endpoint is a separate service, and this repository does not ship one. The author's host
runs one on `127.0.0.1:8010` with the `tasksource-decider-nano` model.

## Point it at your own models

The plugin needs no OpenAI key and ships no model. To use your own models:

1. Run an embeddings server that answers the embeddings protocol and a reranker that answers the
   rerank protocol.
2. Set `embedBaseUrl`, `embedPath` and `embedModelId` for the first, and `rerankBaseUrl`,
   `rerankPath` and `rerankModel` for the second.
3. Use `https` for any host that is not loopback. The plugin rejects a plain `http` URL for a remote
   host, a URL with credentials, and any redirect.
4. Leave `rerankProvider` at `cohere`. The plugin sends and reads that shape only.
5. Let the index rebuild. `embedModelId`, `embedBaseUrl` and `embedPath` are part of the cache
   fingerprint, so changing any of them embeds the routed catalog again on the next request.

```yaml
plugins:
  entries:
    hermes-skill-router:
      settings:
        embedBaseUrl: https://embeddings.example.com
        embedPath: /v1/embeddings
        embedModelId: my-embedding-model
        rerankBaseUrl: https://rerank.example.com
        rerankPath: /v1/rerank
        rerankModel: my-reranker
        rerankProvider: cohere
        embedTimeoutMs: 10000
        rerankTimeoutMs: 30000
```

The model ids are free-form strings passed through to your servers. The vector width is your choice;
the only requirement is that every response for one index uses one consistent width.

## Failure behaviour

The plugin fails open and inert. Any of these conditions leaves the request exactly as it arrived, so
the catalog comes back and the model keeps working when it is the first failure in the compaction
window:

- An endpoint is unreachable, refuses the connection, or fails TLS.
- An endpoint times out.
- The routing budget runs out before the pass finishes, which leaves the bare catalog in place.
- An endpoint answers a redirect. The plugin does not follow it, and no second request is sent.
- A response is malformed, short, long or ragged, or carries a score that is not a finite number.
- The index cannot be built, is empty, or has no routed rows.
- The request has no user text.
- The rendered block would be empty, because nothing was selected and no row was kept.
- The last message already starts with the shared first line of either header, so a replayed request
  is not injected twice.
- The plugin is disabled, or the request carries no skills section.

The catalog is shown once per compaction window. The first failed pass in a window leaves the
request unchanged and sends the whole catalog, a dump, and the rewrite records that fact in the
payload it returns by appending one marker message, `Skill catalog dropped after a routing
failure`. Hermes rebuilds the system message each turn from its cached prompt and drops the prior
turn's appended system messages, so the marker reaches the payload the rewrite sees through the
history the host carries forward. The rewrite recognises the marker in a later request and, on any
further failed pass in that window, strips the catalog and appends the marker again instead of
dumping the catalog a second time. It also suggests nothing after a dump, because the dump showed
every skill and nothing is suggestible for the rest of that window. The marker names no skill and
does not read as a routed block. A compaction that removes the marker starts a new window and the
catalog may be dumped once again. When the first failure arrives in a window the request is left
unchanged, and when the marker is already present the rewrite is a clean catalog strip plus one
marker message, never a partial strip. `None` is still returned whenever there is no marker and no
prior block, so a strip without a replacement never loses the catalog for nothing.

Each distinct failure reason logs one warning per process, not one per request, and no warning
contains the user's text. The reasons are `embed unreachable`, `embed timeout`, `index build failed`,
`rerank unreachable`, `no user text`, and `unsupported api mode`. A routing pass that runs out of its
budget logs one line naming that budget, `routing budget <n> ms exhausted`, and then abandons the
pass. After a failure the endpoint is not probed again until `healthCheckMs` has passed, so a stopped
service costs one attempt per window instead of one timeout per request. A cache write failure is
logged once and the in-memory index is still used.

A catalog under half the saved size is not one of those failures. When the routed rows number fewer
than half the rows in the saved index at `cachePath`, the build is refused: nothing is embedded, the
cache file is left untouched, and the plugin routes from the saved index instead. Suggestions and
their read paths come from the saved rows, including a name the arrived catalog does not carry,
because the saved index stores no paths and the host lookup resolves them again. The refusal logs one
warning per refused catalog, naming the arrived count, the saved count and the cache path, and it
says to remove the cache file to force a rebuild. The trade is deliberate: a real catalog shrink of
more than half keeps routing against the saved list until someone removes the file. A catalog whose
routed rows number at least half the saved rows is adopted as before, including growth, a handful of
lost rows and an unchanged catalog. A missing, corrupt, wrong-version or wrong-shaped cache is not a
saved list, so the build proceeds and no refusal warning is logged. A turn whose routed catalog is
empty returns the request unchanged before the index is touched, so the guard's zero case is reached
only by a direct library call.

The term index and the association store are not preconditions. A term index that is missing,
foreign or unbuildable disables only the BM25 channel for that pass, with one warning, and a store
that is missing or unreadable disables only the association channel and the bump, with one warning.
Both leave the semantic channel and the rerank pass working, so the turn still suggests what the
remaining channels found. The warnings name the file and never the query.

`skill_view` and `skills_list` are never touched. The model can open any name the block lists, and
the background review can still list and open every skill.

## Spawn coverage

The plugin runs on every request that reaches the middleware, using that request's last user text as
the query. There is no spawn list. A `/btw` side question, the after-turn background review, a cron
job and a kanban worker get exactly the same treatment as an interactive turn. A forked agent replays
its own recorded request text, and that text is what ships, so the fork routes on its own turn rather
than on its parent's wording. `skills_list` and `skill_view` stay available on every spawn, which is
what the background review uses.

## Sharing the endpoints

Both endpoints are plain HTTP, so another agent or tool can use them. The embeddings side takes
`POST /v1/embeddings` with a `model` and an `input` list. The rerank side takes the Cohere shape at
`/v1/rerank` with a `model`, a `query` and a `documents` list, each document in the `documentPrefix`
format above. A client that sends the same model ids and the same `documentPrefix` ranks against the
same vectors and scores, so it suggests the same skills for the same query.

## Measured effect

Measured on the author's machine against the live endpoints on `127.0.0.1:8012` and
`127.0.0.1:8010`, during `hermes -z` turns recorded with `HERMES_DUMP_REQUESTS=1`, which writes the
request to `~/.hermes/sessions/request_dump_*.json` under `request.body`.

| Measure | Result |
| --- | --- |
| System message before the rewrite | 70,743 characters carrying 424 catalog rows |
| Rewritten system message | 22,657 characters |
| Injected block on that turn | 1,233 characters, 11 lines, 8 suggestions |
| Net change | about 47,000 fewer characters, roughly 12,000 tokens |
| Warm rewrite | 0.75 to 0.79 seconds |
| Cold index build | 27 chunked embeddings calls, 37.8 seconds total for 424 rows |

The values come from live request dumps, not from one byte-identical payload: the 70,743-character
message and its 424 rows are from a dump that still carried the catalog, while the 22,657-character
message and the 1,233-character block are from a rewritten dump later in the run. A strip of the
70,743-character message measured 22,310 characters, so the net figure is approximate. The token
figure is an estimate at roughly four characters per token; the character counts are exact. The warm
rewrite figure is the plugin's `rewrite` over the dumped payload with the index cache warm. The first
call after a process starts measured 1.6 to 2.0 seconds; steady-state calls measured 0.75 to 0.79
seconds.

The cold build embeds the 424 routed rows in 27 requests of at most 16 texts, about 38 seconds in
total. A 16-row chunk measured 1.2 to 2.0 seconds in the live run, and 2.4 seconds at the peak of a
repeat under load; the 6000 ms default leaves room for one slow chunk. Later turns reuse the cache
until the fingerprint changes.

## Verification status

Verified live on this machine on 2026-11-10, with Hermes Agent v0.22.2+1671, Python 3.12.11 and
pytest 9.1.1:

- Both endpoints answered their health routes, and the embeddings unit was linked from the checkout
  into `~/.config/systemd/user/` and running.
- Live `hermes -z` turns with `HERMES_DUMP_REQUESTS=1` produced dumps whose payloads show the catalog
  stripped, the last message starting with `Skills routed for this request`, and the `model` and
  `tools` fields intact. The dumps carry the 70,743, 424, 22,657 and 1,233 numbers above.
- The cache at the default path matched the live catalog fingerprint with 424 vectors of width 1024,
  and a warm rewrite over the same payload measured 0.75 to 0.79 seconds.
- The cold build of those 424 rows took 27 chunked calls and 37.8 seconds against the live service,
  with 16-row chunks measured at 1.2 to 2.0 seconds in the live run (2.4 seconds at the peak of a
  repeat).

Covered by tests with fake endpoints and temporary files, not by a live turn: both HTTP clients on
HTTP errors, timeouts, 302 and 307 redirects without a follow-up request, and malformed or partial
bodies; the catalog strip and partition on captured live bytes, including `[names only]` rows and
shadowed-copy notes; index cache hits and fingerprint invalidation; selection parity; and the
injection shape. None of those tests needs a GPU, a model file or a network.

Not verified here: a live Codex Responses or Anthropic Messages turn, which the tests cover only as
an unchanged payload; the desktop pane and TUI widget rendered in a live desktop or TUI session;
macOS, Windows, and non-CPU embedding backends, which are out of scope.

## Tests and validation

```bash
python3 -m pytest -q
hermes plugins validate .
```

The plugin is stdlib-only Python; `urllib` is its only HTTP client. The embeddings service needs
`onnxruntime` and `tokenizers`, but the plugin itself does not import either.

## Troubleshooting

### No block appears in the dump

A correct dump carries no `<available_skills>` block and no `## Skills` heading, its last message
starts with `Skills routed for this request`, and `request.body` still carries `model` and `tools`.
When the block is missing, read the plugin log: a failed pass leaves one warning that names the
reason, which you can compare with the reason list under [Failure behaviour](#failure-behaviour), and
a completed pass leaves the decision line under [Data disclosure](#data-disclosure). Enable the
plugin in the profile first: with it disabled, every request keeps its catalog.

### The first turn is slow

The first turn after a cold cache builds the index once, about 38 seconds on this host in 27 chunked
calls. Each 16-row chunk measured 1.2 to 2.0 seconds in the live run, and 2.4 seconds at the peak of
a repeat, so the default `embedTimeoutMs` of 6000 covers it; raise that value on a slower host. A
failed build keeps the catalog, and the endpoint is not probed again until `healthCheckMs` has
passed. Later turns load the cached index.

### The rerank endpoint is missing

This repository ships no reranker. Until `rerankBaseUrl` answers the protocol in
[Endpoint protocols](#endpoint-protocols), the rerank call fails, the request keeps its catalog and
the plugin suggests nothing.

### The pane or the widget shows idle

Check the two enable steps under [Enabling the display](#enabling-the-display). With
`showSuggestions: false` the plugin removes the record and writes nothing, so both surfaces stay
idle. The TUI widget reads the file directly: it shows idle once the record is gone, and the inert
replacement when the removal was refused.

### A setting was ignored

The log names the key once, at the first load that rejects it, and the value falls back to its
default. Check the bound in [Every key](#every-key).

### A remote endpoint is rejected

A plain `http` URL works only for `127.0.0.1`, `localhost` or `::1`. Any other host needs `https`, and
a URL with credentials, a query or a fragment is refused. The rule and its fallback are in
[Endpoint URL rule](#endpoint-url-rule).
