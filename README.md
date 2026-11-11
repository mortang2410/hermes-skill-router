# hermes-skill-router

A Hermes Agent plugin that removes the always-on skill catalog from every Chat Completions request
and appends a short, reranked list of skill names and one-line descriptions for the turn. The model
keeps `skills_list` and `skill_view`, so it can still list and open any skill; the injected block
carries the guidance the removed prose provided. The plugin shares one embeddings endpoint and one
rerank endpoint with DSH's skill router, so both harnesses rank against the same vectors.

License: MIT. See [LICENSE](LICENSE).

## What the plugin removes and adds

Hermes builds a `## Skills` section into the system prompt on every turn. The plugin removes that
whole section from the first system message: the `## Skills` heading, the prose before and after
the `<available_skills>` block, the block itself, the trailing sentence, and any single-line
parenthetical notes the renderer appends after the block.

The catalog carries two row shapes:

- `- name: description` rows.
- `category [names only]: a, b` lines, which a coding posture demotes to a list of names with no
  descriptions.

Every row is partitioned once per request into kept rows and routed rows:

- Kept rows stay listed in the injected block and are never suggested: every name in
  `eagerSkills`; every row a non-empty `deferredSkills` list does not name; and every name on a
  demoted `[names only]` line, even when `deferredSkills` names it. A demoted name is always listed
  by name only and never routed, because the renderer never emitted its description.
- Routed rows are everything else. Only routed rows enter the embeddings index and can be
  suggested.

Hermes also appends a parenthetical note for shadowed skills when two tiers carry a skill with the
same name and relative path. That note names copies that `skill_view` cannot load, so the plugin
removes the note with the section and drops those names from the block. They were not loadable
before the plugin ran either.

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
guidance the removed prose provided, including the call to `skills_list` when none of the
suggestions fits.

Below the header the block carries the selected suggestions, best first, then the kept rows under an
`Also listed:` marker:

```
- <suggested name>: <description>
- <suggested name with no description>

Also listed:
- <kept name>: <description>
```

Each row is one line. Descriptions are collapsed to one line and clipped to `maxDescriptionChars`.
A row that has no description is listed as `- name`, which is how demoted `[names only]` rows and a
suggested name the index no longer carries appear. A name that is both kept and suggested appears
only once, as a suggestion. `maxSuggestions` caps suggested rows alone, so a value of `0` suggests
nothing while every kept row is still listed.

The strip and the injection are atomic. The plugin removes the section only when it can replace it
with a non-empty block, so a request with no suggestion and no kept row keeps its original section
byte for byte. The first line of the block is a marker, not the removed `## Skills` heading, so a
request dump shows whether the plugin ran.

The selected suggestion names are also written to a names-only display record after the payload is
built. The record is never part of a request; see [Display](#display).

## Measured effect

Measured on the author's machine in the live run behind this README, with the live endpoints on
`127.0.0.1:8012` and `127.0.0.1:8010`. The payloads come from live `hermes -z` turns recorded with
`HERMES_DUMP_REQUESTS=1`, which writes the request to `~/.hermes/sessions/request_dump_*.json`
under `request.body`.

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
rewrite figure is the plugin's `rewrite` over the dumped payload with the index cache warm. The
first call after a process starts measured 1.6 to 2.0 seconds; steady-state calls measured 0.75 to
0.79 seconds.
The cold build embeds the 424 routed rows in 27 requests of at most 16 texts, about 38 seconds in
total. A 16-row chunk measured 1.2 to 2.0 seconds in the live run, and 2.4 seconds at the peak of a
repeat under load; the 6000 ms default leaves room for one slow chunk. Later turns reuse the cache
until the fingerprint changes.

Operational note: enable the plugin in the profile before expecting any routing. The first turn
after a cold cache builds the index once, about 38 seconds on this host in 27 chunked calls. Each
16-row chunk measured 1.2 to 2.0 seconds in the live run (2.4 seconds at the peak of a repeat), and
the default `embedTimeoutMs` of 6000 covers it; raise that value on a slower host. A failed build
keeps the catalog, and the endpoint is not probed again until `healthCheckMs` has passed. Later
turns load the cached index.

## Install

### 1. Install the plugin

From a published repository, replacing `<owner>` with the GitHub account that publishes the plugin:

```bash
hermes plugins install <owner>/hermes-skill-router
```

For development, link the checkout into the plugins directory instead:

```bash
ln -sfn /path/to/hermes-skill-router ~/.hermes/plugins/hermes-skill-router
```

### 2. Enable it in the profile

Add the plugin id to the profile's `plugins.enabled` list, or run:

```bash
hermes plugins enable hermes-skill-router
```

Settings live under `plugins.entries.hermes-skill-router.settings` in the same profile config:

```yaml
plugins:
  enabled:
    - hermes-skill-router
  entries:
    hermes-skill-router:
      settings:
        embedBaseUrl: http://127.0.0.1:8012
        rerankBaseUrl: http://127.0.0.1:8010
```

### 3. Start the two endpoints

The rerank endpoint is a separate service, and this repository does not ship one. Point
`rerankBaseUrl` at any service that implements the rerank protocol below. The author's host runs
one on `127.0.0.1:8010` with the `tasksource-decider-nano` model. Both base URLs must be bare
origins: a URL with credentials, a query or a fragment is rejected and falls back to the default
with one warning.

You need an embeddings endpoint too, but any server that answers the embeddings protocol works. This
repository ships one for BGE-M3 ONNX on CPU. Its model directory must contain `onnx/model.onnx` and
`tokenizer.json`, and the Python environment needs `onnxruntime`, `tokenizers`, `fastapi`, `uvicorn`
and `numpy`. From the checkout root:

```bash
HSR_MODEL_DIR=/path/to/bge-m3 HSR_PORT=8012 python3 -m service.embed_server
```

The service listens on `127.0.0.1:8012` by default and serves `POST /v1/embeddings` and
`GET /health`. Other settings are `HSR_MODEL_ID` (default `bge-m3`) and `HSR_MAX_LENGTH` (default
512). The tracked systemd user unit runs the same command, but it carries the author's model path in
`HSR_MODEL_DIR`, so override that value for your host, for example with
`systemctl --user edit hermes-skill-router-embed.service`. Then link and start the unit:

```bash
ln -sfn /path/to/hermes-skill-router/service/hermes-skill-router-embed.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now hermes-skill-router-embed.service
```

### 4. Check it

```bash
curl -s 127.0.0.1:8012/health
curl -s 127.0.0.1:8010/health
```

The first answers `{"status": "ok", "model": "bge-m3", "dim": 1024}` once the model is loaded, and
`{"status": "loading"}` while it starts. Then run one turn and read the newest dump:

```bash
HERMES_DUMP_REQUESTS=1 hermes -z "draft a git commit message for a bug fix"
ls -t ~/.hermes/sessions/request_dump_*.json | head -1
```

In that dump, the system message must carry no `<available_skills>` block and no `## Skills`
heading, the last message must start with `Skills routed for this request`, and `model` and `tools`
must still be present in `request.body`. If the block is missing, read the plugin log for the one
warning that names the failure reason.

### 5. Optional: install the TUI widget

See [Display](#display) for the desktop pane and the Ink TUI widget.

## How routing works

1. Strip the skills section and partition its rows with `eagerSkills` and `deferredSkills`.
2. Build or load the index over the routed rows only. The index is the routed rows' names and
   descriptions, embedded through the embeddings endpoint.
3. Embed the text of the request's last user message, then keep the `shortlistSize` routed rows
   nearest to it by cosine similarity.
4. Send those candidates to the rerank endpoint, one document per candidate, in the DSH format
   `documentPrefix + name + " - " + description`.
5. Keep the candidates whose rerank score is at or above `cutoff`, sort them by score, and cap them
   at `maxSuggestions`.
6. Render the block with those suggestions and the kept rows, and append it as one system message.

The persisted index at `cachePath` is keyed by a fingerprint of every routed row's text and full
description, `embedModelId`, the document format version, and `embedBaseUrl + embedPath`. A change
to a routed skill, the model, the endpoint, or the description clip rebuilds it. A change to a kept
row alone leaves it intact, and the cache is reused across sessions and across turns. The catalog
is frozen per Hermes session, so a session start normally reuses the cache.

## Configuration

Every value in this section is read from `plugins.entries.hermes-skill-router.settings` in the
profile config. An out-of-bound or malformed value falls back to its default and appends a note
naming the key. Each rejected setting is logged once per process, on the first load that rejects
it, and an endpoint URL in that message is reduced to its scheme and host, so credentials never
reach the log. An unknown key appends a note and is otherwise ignored. A malformed value never
aborts a request.

### Eager and deferred lists

`eagerSkills` and `deferredSkills` mirror DSH's eager and deferred split, applied to skills. The
partition is computed per request from the catalog rows.

- `eagerSkills` names skills that stay listed on every request and are never routed. Example:
  `eagerSkills: [writing-for-agents]` keeps that skill in the block under `Also listed:` and out of
  the index, the rerank call and the display record.
- `deferredSkills` names the routable pool. An empty list, the default, means every skill that is
  not eager is routable. A non-empty list restricts routing to the named skills and leaves every
  other skill listed. Example: `deferredSkills: [github-pr-workflow, code-review]` suggests only
  those two and lists every other non-eager skill under `Also listed:`.
- A name in both lists is eager, and the plugin warns once per process for it. A configured name
  that the catalog does not carry is ignored, and an unknown eager or deferred name warns once per
  process. A name on a demoted `[names only]` line is always listed and never routed, even when
  `deferredSkills` names it.

### Every key

<!-- config-defaults:start -->
| Key | Default | Bound | Effect |
| --- | --- | --- | --- |
| `enabled` | `true` | boolean | Master switch. False leaves every request untouched. |
| `eagerSkills` | empty | list of skill names | Skills always listed and never routed. A name in both lists is eager. |
| `embedBaseUrl` | `http://127.0.0.1:8012` | http for a loopback host only, https elsewhere, no credentials, query or fragment | Embeddings endpoint root. A rejected value falls back to the default with one warning. |
| `embedPath` | `/v1/embeddings` | path starting with `/` | Path appended to `embedBaseUrl`. |
| `embedModelId` | `bge-m3` | non-empty string | `model` field sent to the embeddings endpoint. Part of the index fingerprint, so a change rebuilds the cache. |
| `embedTimeoutMs` | `6000` | 100..60000 | Per-request embeddings timeout. Each index-build chunk of at most 16 texts gets its own budget; 16-row chunks measured 1.2 to 2.0 s in the live run. |
| `rerankBaseUrl` | `http://127.0.0.1:8010` | http for a loopback host only, https elsewhere, no credentials, query or fragment | Rerank endpoint root. A rejected value falls back to the default with one warning. |
| `rerankPath` | `/v1/rerank` | path starting with `/` | Path appended to `rerankBaseUrl`. |
| `rerankModel` | `tasksource-decider-nano` | non-empty string | `model` field sent to the rerank endpoint. |
| `rerankProvider` | `cohere` | `cohere` | Request and response shape. Cohere is the only implemented provider. |
| `documentPrefix` | `This skill supplies a procedure, tool, or domain-specific instructions that the agent would actually need to carry out the current task correctly. Skill: ` | non-empty string | Exact text before `<name> - <description>` in a rerank document. The default matches DSH. |
| `deferredSkills` | empty | list of skill names | Routable skills. Empty means every non-eager skill is routable; a non-empty list restricts routing to the named skills and lists the rest. |
| `rerankTimeoutMs` | `20000` | 100..60000 | Per-request rerank timeout. One rerank call carries the whole shortlist. |
| `routingTimeoutMs` | `30000` | 1000..60000 | Whole routing pass budget, from the query embed through the rerank reply. It starts after the index is loaded or built, so the cold build is never cut, and each per-request timeout is capped at the time left. On expiry the request keeps the bare catalog and gets no suggestions. |
| `shortlistSize` | `50` | 1..383 | Candidates carried from embeddings into rerank. |
| `cutoff` | `0.01` | 0..1 | Rerank `relevance_score` a candidate needs, compared inclusively. |
| `maxSuggestions` | `8` | 0..50 | Hard cap on suggested rows. Zero selects none while kept rows are still listed. |
| `maxDisplayNames` | `12` | 1..50 | Names the desktop pane and the TUI widget list. The count stays visible when collapsed. |
| `maxDescriptionChars` | `400` | 3..4000 | Description characters kept per row, in the block and the embedding text. Part of the index fingerprint. |
| `cachePath` | `~/.cache/hermes-skill-router/index.json` | path, empty disables persistence | Persisted index. Empty means each process builds its own index and keeps it in memory. |
| `showSuggestions` | `true` | boolean | Let the display read the last suggestion names. False stops the record and removes an existing one. |
| `suggestionsPath` | `~/.cache/hermes-skill-router/suggestions.json` | path, empty disables the record | Display record both surfaces read. Names only, never request text. |
| `healthCheckMs` | `60000` | 0..3600000 | Minimum gap between probes of an endpoint that failed. Zero probes every request. |
<!-- config-defaults:end -->

Data disclosure: on every routed request the plugin sends the text of the request's last user
message to the embeddings endpoint. On a cold index build it sends the name and description of
every routed skill there too. It sends the shortlisted skill names and descriptions to the rerank
endpoint. Nothing else leaves the process: the display record stores names only, request text is
never logged, and no request text is written to the index cache.

Endpoint URL rule: a plain `http` URL is accepted only for a loopback host, which means
`127.0.0.1`, `localhost` or `::1`. Every other host must use `https`. A URL carrying credentials, a
query or a fragment is rejected and falls back to its default with one warning, because the request
path is appended to the root. Both HTTP clients refuse to follow a redirect, so a 3xx fails the call
instead of sending the request text to another host. The defaults point at loopback:
`http://127.0.0.1:8012` for embeddings and `http://127.0.0.1:8010` for rerank.

## Endpoint protocols

Both endpoints are plain HTTP or HTTPS with JSON bodies. The plugin sends no API key and no
`Authorization` header, so an endpoint that needs credentials must sit behind a proxy that adds
them. Any server that answers these shapes works, including the bundled embeddings service and the
author's rerank service.

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
one response must share one width. A short, long, ragged or non-numeric response fails the call,
and no partial result is used. The bundled service returns 1024-wide L2-normalised vectors, but the
plugin only needs a consistent width because it computes cosine similarity itself.

Health check:

```http
GET /health HTTP/1.1
Host: 127.0.0.1:8012
```

```json
{"status": "ok", "model": "bge-m3", "dim": 1024}
```

The plugin does not call `/health`; it treats a failed request as the health signal and remembers
the failure for `healthCheckMs`. The route exists for operators and for the bundled unit's own
readiness. While the bundled service loads its model it answers `{"status": "loading"}` and
returns 503 for embeddings requests; the plugin treats that as a failure and leaves the request
unchanged.

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

The response must carry exactly one result per document. Results may arrive in any order; `index`
is the zero-based position of the document in the request, and the plugin restores input order from
it. `relevance_score` is a finite probability. A missing, duplicated or out-of-range index, a
non-numeric score, or a result count that differs from the document count fails the call, and no
partial ranking is used. The selector compares the score directly and inclusively with `cutoff`;
it applies no second softmax.

### Health and timeouts

| Setting | Applies to | Default | Bound |
| --- | --- | --- | --- |
| `routingTimeoutMs` | One whole routing pass: the query embed, the shortlist, and the rerank reply | 30000 ms | 1000..60000 |
| `embedTimeoutMs` | One embeddings request, including each index-build chunk | 6000 ms | 100..60000 |
| `rerankTimeoutMs` | One rerank request carrying the whole shortlist | 20000 ms | 100..60000 |
| `healthCheckMs` | How long a failed endpoint is not probed again | 60000 ms | 0..3600000 |

`routingTimeoutMs` bounds one whole routing pass. The budget starts after the skill index is
loaded or built, so it never covers that build: the build issues 27 chunked embeddings requests and
takes about 38 seconds here, and it already fails through its own path by returning no index
instead of being cut off. Each per-request timeout is capped at the milliseconds the budget has
left, so a slow embed or rerank is cut when the budget ends. The cap bounds the wait for an answer,
not the transfer of one: a server that trickles bytes within every socket timeout can still run
past the budget. When the budget runs out the plugin abandons the pass: the request keeps the bare
catalog and gets no suggestion block, the same inert fallback every other failure uses, and one
warning names the budget.

The 30000 ms default rests on two facts. The two per-request timeouts allow a 26 second pass at
worst, 6000 plus 20000, so 30000 sits above that ceiling and a working but slow path is never cut,
and it matches the DSH router's 30 second catalog read deadline. It also has headroom over the
measurements recorded on this machine against its live endpoints: a steady-state single-text
embed took 236 to 306 ms, a 50-document rerank took 2158 to 2255 ms, and the first embed after an
endpoint restart took 2700 ms, so a cold but healthy pass cost about 5 seconds, 2700 plus 2255.
Those are the slow case rather than a constant: an earlier live run on the same host measured a
full warm rewrite at 0.75 to 0.79 seconds, and a probe on 2026-11-10 measured a one-text embed at
84 ms and a 50-document rerank at 287 ms. On a host whose endpoints are slower than the recorded
figures, raise the budget above your own measured cold pass before lowering anything else.

## Point it at your own models

The plugin needs no OpenAI key and ships no model. To use your own models:

1. Run an embeddings server that answers the embeddings protocol and a reranker that answers the
   rerank protocol.
2. Set `embedBaseUrl`, `embedPath` and `embedModelId` for the first, and `rerankBaseUrl`,
   `rerankPath` and `rerankModel` for the second.
3. Use `https` for any host that is not loopback. The plugin rejects a plain `http` URL for a
   remote host, a URL with credentials, and any redirect.
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

The model ids are free-form strings passed through to your servers. The vector width is your
choice; the only requirement is that every response for one index uses one consistent width. Both
endpoints can be shared with DSH, which is described below.

## Failure behaviour

The plugin fails open and inert. Any of these conditions leaves the request exactly as it arrived,
so the catalog comes back and the model keeps working:

- An endpoint is unreachable, refuses the connection, or fails TLS.
- An endpoint times out.
- The routing budget runs out before the pass finishes, which leaves the bare catalog in place.
- An endpoint answers a redirect. The plugin does not follow it, and no second request is sent.
- A response is malformed, short, long or ragged, or carries a score that is not a finite number.
- The index cannot be built, is empty, or has no routed rows.
- The request has no user text.
- The rendered block would be empty, because nothing was selected and no row was kept.
- The last message already starts with the shared first line of either header, so a replayed
  request is not injected twice.
- The plugin is disabled, or the request carries no skills section.

Each distinct failure reason logs one warning per process, not one per request, and no warning
contains the user's text. The reasons are `embed unreachable`, `embed timeout`, `index build
failed`, `rerank unreachable`, `no user text`, and `unsupported api mode`. A routing pass that runs
out of its budget logs one line naming that budget, `routing budget <n> ms exhausted`, and then
abandons the pass. After a failure the endpoint is not probed again until `healthCheckMs` has
passed, so a stopped service costs one attempt per window instead of one timeout per request. A
cache write failure is logged once and the in-memory index is still used.

Only OpenAI-compatible Chat Completions requests, with `api_mode: "chat_completions"`, are
rewritten. Codex Responses (`api_mode: "codex_responses"`) carries the prompt in `instructions` and
`input`, and Anthropic Messages (`api_mode: "anthropic_messages"`) carries it in `system`, so a
`messages` rewrite would do nothing or corrupt the call. Those two modes pass through untouched and
keep the catalog; the unsupported mode is logged once.

`skill_view` and `skills_list` are never touched. The model can open any name the block lists, and
the background review can still list and open every skill.

## Spawn coverage

The plugin runs on every request that reaches the middleware, using that request's last user text as
the query. There is no spawn list. A `/btw` side question, the after-turn background review, a cron
job and a kanban worker get exactly the same treatment as an interactive turn. A forked agent
replays its own recorded request text, and that text is what ships, so the fork routes on its own
turn rather than on its parent's wording. `skills_list` and `skill_view` stay available on every
spawn, which is what the background review uses.

## Display

The last routed decision is readable in two places, and neither place costs a token. The plugin
writes the record after it has built the payload it returns, and nothing from the record is ever
added to a request.

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

`names` holds the selected suggestion names only, capped at `maxDisplayNames`. A kept row is never
recorded, so the display cannot present a pinned or deferred-excluded skill as a suggestion. The
record never carries request text. `status` is `routed` when the block was injected and `inert`
when the catalog stayed in place. `origin` is `turn` for the automatic pass and `search` for a model
call to the search tool, and a schema 1 file still reads with origin `turn`. The newest record wins, and it carries the session and turn ids
the dispatcher supplies so a second session or a fork replacing it is visible on screen when the
ids are present. `showSuggestions: false` stops the writes and removes an existing record, which
turns both surfaces idle.

Desktop pane: `Skills routed` is a collapsed-by-default pane on the right side, 260 px wide. Its
heading carries the count, its list shows the names, and for a routed record one muted caption line
reads `turn <turn_id> · <local HH:MM>` from the record's own `turn_id` and `at`, or
`model search · <local HH:MM>` when a model search produced the names. The caption is
shown only for a routed record, so a stale turn cannot sit under an inert or idle state. It
refreshes on mount and when the plugin broadcasts
`plugin.hermes-skill-router.suggestions.updated`. It reads the plugin's own route,
`GET /api/plugins/hermes-skill-router/suggestions`, which returns the record or
`{"status": "idle", "names": []}` when the file is absent or unreadable. The route never returns
the user's text. The pane is copied into `$HERMES_HOME/desktop-plugins/hermes-skill-router/` when
the plugin folder is in a plugins root.

TUI widget: Hermes exposes no plugin display hook in the Ink TUI, so the plugin ships an ambient
user widget instead. Install it with a symlink:

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
`turn <turn_id> · <local HH:MM>` for a turn record, and `model search · <local HH:MM>` for a record
a model search wrote, which has no turn id. The caption is shown only for a routed record. A
missing, corrupt or `inert` record shows one idle line instead. It uses theme tones only and never
reads plugin config, which is why the path is an argument.

The display has two independent enable steps:

1. The profile's `plugins.enabled` entry enables the Python backend, including the route.
2. The plugin's own toggle in the desktop app's Capabilities panel enables the desktop half. Hermes
   never calls the desktop half's `register()` when that toggle is off, so the pane does not exist.
   With the backend disabled instead, the pane shows the enable line for the missing route.

The three display keys are `showSuggestions`, `suggestionsPath` and `maxDisplayNames`, documented in
the config table above.

The classic prompt_toolkit TUI exposes no plugin display surface, so it is not covered.

## Sharing endpoints with DSH

DSH's skill router reaches the embeddings endpoint through its HTTP embedder, and the rerank
endpoint through its gateway URL. The `skill-router` row in the active DSH profile's
`cordis.patch.yml` carries these keys:

| Key | Default | Bound | Effect |
| --- | --- | --- | --- |
| `embedBaseUrl` | empty | string | Non-empty selects the HTTP embedder. Empty keeps the in-process ONNX embedder. |
| `embedPath` | `/v1/embeddings` | string | Path appended to `embedBaseUrl`. |
| `embedModelId` | `bge-m3` | string | `model` field sent to the endpoint. |
| `embedTimeoutMs` | `2000` | 100..60000 | Per-request timeout. |

For example:

```yaml
- id: skill-router
  name: '@local/dsh-skill-router'        # the installed package name
  config:
    modelPath: '/path/to/bge-m3'          # still required: suppression is gated on it
    embedBaseUrl: 'http://127.0.0.1:8012'
    embedPath: '/v1/embeddings'
    embedModelId: 'bge-m3'
    embedTimeoutMs: 2000
    gatewayUrl: 'http://127.0.0.1:8010'
    judgmentModelId: 'tasksource-decider-nano'
    judgmentProvider: 'cohere-compatible'
```

DSH still requires a non-empty `modelPath` even when `embedBaseUrl` is set, because its catalog
suppression is gated on it; the ONNX path stays as the fallback. The DSH index fingerprint includes
the embedder identity, meaning the endpoint kind, base URL, path and model id, so repointing the
endpoint rebuilds its cache instead of reusing vectors from another model. The rerank side is
unchanged: DSH calls the same `gatewayUrl` on 8010 with the Cohere shape, which is why the two
harnesses suggest from the same scores.

## Verification status

Verified live on this machine on 2026-11-10, with Hermes Agent v0.22.2+1671, Python 3.12.11 and
pytest 9.1.1:

- Both endpoints answered their health routes, and the embeddings unit was linked from the
  checkout into `~/.config/systemd/user/` and running.
- Live `hermes -z` turns with `HERMES_DUMP_REQUESTS=1` produced dumps whose payloads show the
  catalog stripped, the last message starting with `Skills routed for this request`, and the
  `model` and `tools` fields intact. The dumps carry the 70,743, 424, 22,657 and 1,233 numbers
  above.
- The cache at the default path matched the live catalog fingerprint with 424 vectors of width
  1024, and a warm rewrite over the same payload measured 0.75 to 0.79 seconds.
- The cold build of those 424 rows took 27 chunked calls and 37.8 seconds against the live
  service, with 16-row chunks measured at 1.2 to 2.0 seconds in the live run (2.4 seconds at the peak
  of a repeat).

Covered by tests with fake endpoints and temporary files, not by a live turn: both HTTP clients on
HTTP errors, timeouts, 302 and 307 redirects without a follow-up request, and malformed or partial
bodies; the catalog strip and partition on captured live bytes, including `[names only]` rows and
shadowed-copy notes; index cache hits and fingerprint invalidation; selection parity; and the
injection shape. None of those tests needs a GPU, a model file or a network.

Not verified here: a live Codex Responses or Anthropic Messages turn, which the tests cover only as
an unchanged payload; the desktop pane and TUI widget rendered in a live desktop or TUI session;
macOS, Windows, and non-CPU embedding backends, which are out of scope.

## Development

```bash
python3 -m pytest -q
hermes plugins validate .
```

The plugin is stdlib-only Python; `urllib` is its only HTTP client. The embeddings service needs
`onnxruntime` and `tokenizers`, but the plugin itself does not import either.

## License

MIT. See [LICENSE](LICENSE).
