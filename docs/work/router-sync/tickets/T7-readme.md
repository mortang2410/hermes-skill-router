# T7 README with every knob and both protocols

Status: ready for agent. Repo: `hermes-skill-router`. Work root: `docs/work/router-sync/`.

## Delivers

The public `README.md`: what the plugin removes and adds, install steps, the measured cost it
removes, every config key with its default and bound, both endpoint protocols with request and
response examples, the failure behaviour, the spawn coverage, and how a third party points it at
their own models.

## Covers

Stories S8 (and the README clauses of S5, S7, S9, S10). Requirements R7, R10. Module none. Contracts
C1, C2.

## Exclusive files

- `README.md`
- `LICENSE`
- `tests/test_readme.py`

## Blockers

T5, and T6 for the shared-endpoint section.

## What it must do

1. Opening section: the measured baseline on this machine (427 rows, 47,981 characters in
   `<available_skills>`, about 12,400 tokens per request) and what replaces it (a block of names and
   one-line descriptions for the turn's shortlist).
2. Install: `hermes plugins install <owner>/hermes-skill-router`, or a symlink of the checkout into
   `~/.hermes/plugins/hermes-skill-router` for development; then start the two services, and install
   the embeddings unit by linking it into `~/.config/systemd/user/` and running `daemon-reload`.
3. Config reference: one row per key from `hermes_skill_router/config.py`, with default, bound, and
   what changing it does. Keys live under `plugins.entries.hermes-skill-router.settings` in the
   profile config. Document `eagerSkills` and `deferredSkills` as the mirror of DSH's eager/deferred
   split: eager stays listed and is never routed; an empty deferred list routes every non-eager skill;
   a non-empty one restricts routing to the named skills and leaves the rest listed; a name in both is
   eager; a name on a demoted `[names only]` line is always listed and never routed. Give one worked
   example of each list.
4. Endpoint protocols, so a third party can bring their own models:
   - embeddings: `POST <embedBaseUrl><embedPath>`, body `{"model", "input"}`, response
     `{"data": [{"index", "embedding"}]}`, one entry per input, in order;
   - rerank: `POST <rerankBaseUrl><rerankPath>`, body `{"model", "query", "documents", "top_n"}`,
     response `{"results": [{"index", "relevance_score"}]}` in any order;
   - a `health` example and the timeouts.
5. Failure behaviour: any endpoint failure, timeout, malformed response, empty index, a request with
   no user text, or a provider mode other than Chat Completions leaves the request exactly as it
   arrived, so the catalog comes back and the model keeps working. Each reason logs once. Say which
   modes are supported: Chat Completions is rewritten; Codex Responses and Anthropic Messages pass
   through untouched and keep the catalog.
6. Data disclosure: the plugin sends the request's last user text and the shortlisted skill names and
   descriptions to the two configured endpoints. A plain `http` URL is accepted only for a loopback
   host; any other host must be `https`; URL credentials are rejected; redirects are not followed.
   Say this where the keys are documented, and name the default loopback URLs.
7. Both injected headers, quoted verbatim with their shared first line named, so a reader can recognise the block in a request dump and tell a one-shot turn from a normal one,
   plus the note that it carries the skill-use guidance the removed catalog prose used to provide.
8. Spawn coverage, documented as a fact: the plugin runs on every request that reaches the
   middleware, so `/btw`, the after-turn background review, cron and kanban workers get the same
   treatment; there is no spawn list. `skills_list` and `skill_view` stay available, which is what the
   background review uses.
9. The DSH section: the same 8012 and 8010 endpoints serve DSH's router through its HTTP embedder,
   with the profile keys named, plus the fact that the DSH fingerprint includes the endpoint URL so a
   repointed endpoint rebuilds the cache.
10. Honest verification labels: say which parts were verified live on this machine and which are
    mock-host tests.
11. The display section: the `Skills routed` pane in Hermes Desktop and the `/skills-routed` TUI
    widget, the symlink install step into `$HERMES_HOME/tui-widgets/`, the path argument the widget
    accepts and its default, the two independent enable steps (the profile's `plugins.enabled` entry
    for the Python backend and the plugin's own toggle in the desktop app's Capabilities panel), the
    fact that the display shows skill names and the record's turn id and time as a caption, adds
    nothing to a request, and the `showSuggestions`, `suggestionsPath` and `maxDisplayNames` keys.

## Verification

```bash
python3 -m pytest tests/test_readme.py -q
hermes plugins validate /home/wilder/git-repos/hermes-skill-router
```

`tests/test_readme.py` imports `DEFAULTS` from `hermes_skill_router/config.py`, parses the README
config table, and asserts the two key sets are identical, that each default matches, and that each row
carries a non-empty bound and effect column. A new key without a README row fails the suite, and so
does a row whose bound or effect is blank, so the table cannot drift into a bare key list.

## Acceptance criteria

- [ ] Every config key in `DEFAULTS` appears in the README table with its default, its bound and what
      changing it does, and `tests/test_readme.py` fails when a bound or effect cell is empty.
- [ ] Both endpoint protocols have a request and a response example that a reader could implement.
- [ ] The measured baseline numbers appear with the source of the measurement.
- [ ] The data disclosure and the TLS rule appear next to the endpoint keys.
- [ ] The supported provider modes and the pass-through behaviour are stated.
- [ ] The display section names both surfaces, the widget install step and path argument, all three
      display keys, and both enable steps.
- [ ] The failure behaviour and the spawn coverage are stated as facts, not as promises.
- [ ] No claim is made that was not verified; a mock-host test is labelled as such.
- [ ] `LICENSE` (MIT) is present and named in the README.

## Notes

- No em dashes anywhere in the text.
- Do not paste internal absolute paths of this machine except where a default value needs it.
