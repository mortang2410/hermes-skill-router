# hermes-skill-router

A Hermes Agent plugin that takes the always-on skill catalog out of every Chat Completions request
and appends a short list of the skills that fit the turn. The model keeps `skills_list` and
`skill_view`, so it can still list or open any skill.

This fits people who keep one super-agent profile that does everything but do not want the context
polluted by so many skills that the model cannot tell which one to pick. It also pairs well with a
memory provider like [Hindsight](https://github.com/vectorize-io/hindsight), which takes care of
memory routing.

The skill list is embedded once and cached, so a warm turn routes in about a second.

Every setting, both endpoint protocols, the display record and the measurements are in
[technical-documentation.md](technical-documentation.md).

![](docs/assets/performance.png)

## How it works

Hermes dumps the whole skill catalog into the model context. This plugin hides the catalog and only
surfaces skill suggestions when they fit the task. You can also pin a list of eager skills that gets
listed at the start of each compaction window.

Two endpoints rank them. The embeddings endpoint compares your message with the skill rows and keeps
the nearest ones. The rerank endpoint scores that shortlist against the same message, and the plugin
suggests what scores at or above `cutoff`.

If a service is down or too slow, the plugin falls back to the ordinary skill catalog, so a bad
endpoint costs latency, not a broken turn.

## Install

1. Install the plugin.

   ```bash
   hermes plugins install <owner>/hermes-skill-router
   ```

   For development, link the checkout instead:

   ```bash
   ln -sfn /path/to/hermes-skill-router ~/.hermes/plugins/hermes-skill-router
   ```

2. Enable it.

   ```bash
   hermes plugins enable hermes-skill-router
   ```

3. Point the plugin at an embedder and a reranker in the profile config:

   ```yaml
   plugins:
     entries:
       hermes-skill-router:
         settings:
           embedBaseUrl: http://127.0.0.1:8012
           rerankBaseUrl: http://127.0.0.1:8010
   ```

4. Run both services. This repository ships an embeddings server for BGE-M3 on CPU:

   ```bash
   HSR_MODEL_DIR=/path/to/bge-m3 HSR_PORT=8012 python3 -m service.embed_server
   ```

   It serves `POST /v1/embeddings` and `GET /health` on `127.0.0.1:8012`. No reranker is bundled, so
   point `rerankBaseUrl` at any service that answers the rerank protocol in the technical
   documentation. Both base URLs must be bare origins.

5. Check both health routes, then run one turn and read its dump:

   ```bash
   curl -s 127.0.0.1:8012/health
   curl -s 127.0.0.1:8010/health
   HERMES_DUMP_REQUESTS=1 hermes -z "draft a git commit message for a bug fix"
   ```

   The newest dump under `~/.hermes/sessions/` carries no `## Skills` heading, and its last message
   starts with `Skills routed for this request`. When no block appears, the plugin log carries one
   warning naming the reason, or, for a completed pass that selected nothing, the decision line.

## Recommended setup

The defaults are what we run, so you only have to set the two base URLs:

- Embeddings: BGE-M3 on CPU, served by the bundled `service.embed_server` on 8012, with
  `embedModelId: bge-m3`.
- Rerank: `tasksource-decider-nano` through a Cohere-compatible rerank service on 8010, with
  `rerankProvider: cohere`.
- Routing: `cutoff: 0.01`, `shortlistSize: 50`, `maxSuggestions: 10`, and the 6000, 20000 and
  30000 ms timeouts. Keep them unless your endpoints need different budgets.

## Use

A routed turn ends with one extra system message. The sample below is the opening block of a
compaction window: suggestion rows come first, then the pinned rows:

```
Suggestions:
- <suggested name>: <description>
  path: <absolute path to the skill's SKILL.md>
- <suggested name with no description>
  path: <absolute path>

Pinned skills:
- <eager name>: <description>
  path: <absolute path>
```

Every suggestion and pinned row carries its name, its description and the path the host's skill lookup
resolved, which is the file `skill_view` opens. Suggestions are best first, capped at
`maxSuggestions`, where `0` means no cap. Pinned rows appear only in the block that opens a compaction
window; a later turn in the same window carries the suggestions only, and a turn whose messages no
longer carry an earlier block opens the next window with the pinned rows again. Rows that
`deferredSkills` leaves listed stay under an `Also listed:` marker on every turn and are never
suggested. The model can open any row with `skill_view`, and `skills_list` still lists the whole
catalog. When the shortlist misses, it can search by hand: `skill_router_search` takes any `query` in
the model's own words and returns ranked names, descriptions and skill paths.

The last decision also appears on screen, and neither place costs a token. The desktop app gets a
`Skills routed` pane on the right rail, collapsed by default, with the names and one caption line. In
`hermes --tui`, run `/skills-routed` to dock the widget, which reads the same record and shows the
same names. Both show their caption only for a routed record. The record holds names only and never
request text.

## Configuration

Settings live under `plugins.entries.hermes-skill-router.settings` in the profile config. These are
the ones most setups touch:

| Option | Default | Meaning |
| --- | --- | --- |
| `enabled` | `true` | Master switch. False leaves every request untouched. |
| `embedBaseUrl` | `http://127.0.0.1:8012` | Embeddings endpoint root. |
| `embedModelId` | `bge-m3` | Model name sent to the embeddings endpoint. |
| `rerankBaseUrl` | `http://127.0.0.1:8010` | Rerank endpoint root. |
| `rerankModel` | `tasksource-decider-nano` | Model name sent to the rerank endpoint. |
| `eagerSkills` | empty | Skills listed once in each compaction window's opening block, under `Pinned skills:`, with their read paths, and never suggested. |
| `maxSuggestions` | `10` | Cap on suggested rows, from 0 to 50. `0` means unlimited. |

Every other key, with its bound and its effect, is in the
[configuration reference](technical-documentation.md#configuration).

## Documentation

- [technical-documentation.md](technical-documentation.md) covers routing end to end, the request
  rewrite, the suggestion record schema, both display protocols, every setting, both endpoint
  protocols, failure behaviour, the measured effect and the tests.
- [LICENSE](LICENSE) is the MIT license.
