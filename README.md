# hermes-skill-router

A Hermes Agent plugin that takes the always-on skill catalog out of every Chat Completions request
and appends a short list of the skills that fit the turn. The model keeps `skills_list` and
`skill_view`, so it can still list or open any skill.

Every setting, both endpoint protocols, the display record and the measurements are in
[technical-documentation.md](technical-documentation.md).

![Hermes Skill Router artwork](docs/assets/performance.png)

Plugin artwork: a task panel, four checks, the router, and a row of skill cards with Performance
highlighted.

## How it works

Hermes builds a `## Skills` section into the system prompt on every turn. The plugin removes that
section and appends one extra system message with the skills that match your last user message, best
first, plus the skills you pinned under `Also listed:`.

Two endpoints rank them. The embeddings endpoint compares your message with the routed skill rows
and keeps the nearest ones. The rerank endpoint scores that shortlist against the same message, and
the plugin suggests what scores at or above `cutoff`. Both can be the endpoints DSH's skill router
uses, so the two harnesses rank with the same vectors.

The plugin strips and injects in one step, so when it cannot build a block the request arrives
unchanged with its catalog. A stopped endpoint then costs latency, not a broken turn.

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
   warning naming the reason.

## Use

A routed turn ends with one extra system message. Suggestion rows come first, then the kept rows
under an `Also listed:` marker:

```
- <suggested name>: <description>
- <suggested name with no description>

Also listed:
- <kept name>: <description>
```

Rows above `Also listed:` are this turn's suggestions, best first, capped at `maxSuggestions`. Rows
below it stay visible and are never suggested. The model can open any of them with `skill_view`, and
`skills_list` still lists the whole catalog. When the shortlist misses, the model can call
`skill_router_search` with wording of its own.

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
| `eagerSkills` | empty | Skills always listed and never suggested. |
| `maxSuggestions` | `8` | Cap on suggested rows. |

Every other key, with its bound and its effect, is in the
[configuration reference](technical-documentation.md#configuration).

## Documentation

- [technical-documentation.md](technical-documentation.md) covers routing end to end, the request
  rewrite, the suggestion record schema, both display protocols, every setting, both endpoint
  protocols, failure behaviour, the measured effect and the tests.
- [LICENSE](LICENSE) is the MIT license.
