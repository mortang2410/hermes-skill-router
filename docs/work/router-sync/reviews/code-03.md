# code-03: verification of the credential-redaction fix

Round 3. The round-2 finding (a rejected URL note logged its userinfo) is fixed in
`hermes_skill_router/config.py`. This verdict checks the sanitizer, its call sites, and one probe with
a credential-bearing URL; it also audits every log line in the package for unredacted user values.
Read-only: no source file was changed by this review.

## Reviewed revision

| file | sha256 (first 16) | mtime |
| --- | --- | --- |
| hermes_skill_router/config.py | 2717cbda73c7eaf9 | 09:05:17 |
| tests/test_config.py | fa26d39b4b874ed5 | 09:05:37 |
| README.md | cd17892afbb19360 | 09:05 |

`catalog.py`, `inject.py`, `router.py`, `desktop/plugin.js`, `tui/skill-router.mjs` keep their
code-02 hashes; spec.md and T8 were also updated (see the docs check below).

## Tests and scans

- `python3 -m pytest -q tests/test_config.py`, exit 0: `67 passed in 0.06s`.
- `python3 -m pytest -q`, exit 0: `420 passed in 4.44s` (was 408; the ten new redaction tests are
  among the additions).
- `cccc` over the two changed Python files, exit 0: 2 files, 31 functions, 0 parse errors, cognitive
  sum 115, max 49 (`config.py:_coerce` at 205), `_redact_value` 14, `load` 13.
- `lizard` over the same files, exit 1 by design: 1 warning, `_coerce` CCN 29, NLOC 59, at
  `config.py:205-264`. No new warning from the sanitizer.

## The fix

Read `config.py:148-181` (`_redact_url`, `_redact_value`, `_shown`) and every call site:

- `_redact_url` returns `scheme://netloc` where `netloc` is `parts.netloc.rsplit("@", 1)[-1]`, so
  userinfo, path, query and fragment are dropped and the port survives; a value with no netloc or no
  scheme becomes `<redacted URL>`; a string without `://` passes through.
- `_invalid` (`:184-190`), the bounds note (`:222-226`) and the URL note (`:255-263`) now format the
  value and the default with `_shown`, which applies `_redact_value` (recursing into dicts, lists and
  tuples).
- The unknown-key note (`:294-297`) redacts a URL-shaped key and never logs the key's value at all.
- `report_notes` (`:271-277`) logs the note text unchanged, so everything it writes has already been
  through `_shown`.

Probe (`python3 /home/wilder/Trash/router-review/round3/probe_redact.py`), captured log records:

```
basic-auth         embedBaseUrl 'https://alice:hunter2@example.com/v1'
                   -> "embedBaseUrl: 'https://example.com' is not a safe endpoint URL ..."
port-and-path      'http://alice:hunter2@127.0.0.1:8012/v1/embed'
                   -> "embedBaseUrl: 'http://127.0.0.1:8012' is not a safe endpoint URL ..."
userinfo-no-host   'https://alice:hunter2@'
                   -> "embedBaseUrl: '<redacted URL>' is not a safe endpoint URL ..."
unknown-url-key    {'https://bob:swordfish@example.com/v1': 1}
                   -> "https://example.com: unknown config key ignored"
```

Neither username nor password appears in any record; the port is kept; the path, query and fragment
are dropped. The reported case is closed. The new tests
(`test_url_notes_log_only_the_scheme_and_host` over both endpoint keys, `test_a_url_shaped_value_in_another_setting_is_redacted_in_its_note`
for `embedPath`, `test_a_url_shaped_value_inside_a_list_is_redacted_in_its_note`,
`test_an_unknown_key_that_looks_like_a_url_is_redacted`) pin these paths.

## Findings

### 1. Userinfo without `://` still reaches the log, and the README says it cannot

Where: `hermes_skill_router/config.py:154-155` (`if "://" not in value: return value`) with
`config.py:258-262` (the URL note) and `README.md:240-241` ("an endpoint URL in that message is
reduced to its scheme and host, so credentials never reach the log").

What breaks: the sanitizer decides by the presence of `://`, not by the presence of userinfo, so a
credential-bearing value that lost or never had its scheme is logged verbatim. Reproduced with
`load({'embedBaseUrl': 'alice:hunter2@example.com/v1'})`:

```
hermes-skill-router config: embedBaseUrl: 'alice:hunter2@example.com/v1' is not a safe endpoint URL
(http is allowed only for a loopback host, other hosts must use https, and credentials are rejected);
using default 'http://127.0.0.1:8012'
```

The same happens for `//alice:hunter2@example.com/v1`, `https:/alice:hunter2@example.com/v1` and a
bare `user:pw@host` (`/home/wilder/Trash/router-review/round3/probe_shapes.py`). `is_safe_base_url`
rejects all four, so all four produce the note. This is the exact class round 2 reported, narrowed to
values without a scheme; the README claim is currently false for them.

Smallest fix: detect userinfo independently of `://`. If `@` occurs before the first `/`, `?` or `#`
and a `:` occurs before that `@`, treat the segment as userinfo and replace it with `<redacted>@`
keeping the rest, then keep the existing `://` branch for scheme-qualified values. Add tests for
`alice:hunter2@example.com/v1` and `//alice:hunter2@example.com/v1`, asserting neither name appears in
`cfg.notes` or in the captured log record.

### 2. A query in a base URL is accepted, mangles the endpoint, and is echoed at debug level

Where: `hermes_skill_router/config.py:137-145` (`is_safe_base_url` checks scheme, host and userinfo,
never the query or fragment), `hermes_skill_router/embed.py:119` and `rerank.py:60` (endpoint is
`base_url + path`, so a query swallows the path), and the debug lines `embed.py:121,136,140,145` and
`rerank.py:62,84,88,93` (they print that endpoint).

What breaks: `embedBaseUrl: http://127.0.0.1:9/x?token=SECRET-VALUE` passes validation with no note,
the endpoint becomes `http://127.0.0.1:9/x?token=SECRET-VALUE/v1/embeddings`, and the first failure
logs the token. Reproduced (`/home/wilder/Trash/router-review/round3/probe_endpoint.py`):

```
accepted base: http://127.0.0.1:9/x?token=SECRET-VALUE
notes: ()
result: None
LOG: DEBUG embeddings endpoint http://127.0.0.1:9/x?token=SECRET-VALUE/v1/embeddings answered HTTP 502
```

The mangled path also means a query in a base URL cannot work: the request goes to the wrong URL. The
token never appears at warning level, so this only leaks where debug logging is enabled. It is
pre-existing (the round-3 diff did not touch `is_safe_base_url`), and the audit that the brief asked
for is what surfaced it.

Smallest fix: reject a query or fragment in `is_safe_base_url` (a base URL is an origin plus an
optional path prefix) and add a test for `https://example.com/v1?token=x`; that removes both the
mangled request and the echo. If a query must stay supported, sanitize the endpoint in the log calls
instead and split the path correctly when building it.

## Log-line audit

Every `logger.*` call in the package was read for a user-supplied argument:

| line | value in the line | verdict |
| --- | --- | --- |
| `config.py:277` | the note text | safe where `_shown` is used; finding 1 covers the gap |
| `config.py:288` | fixed text | safe |
| `catalog.py:247` | a configured skill name | diagnostic content, not a secret in normal use; would echo a URL pasted into `eagerSkills` |
| `router.py:362` | a fixed reason | safe |
| `router.py:118,170,192,231,332,339`, `__init__.py:46`, `plugin_api.py:145` | fixed text plus a traceback | tracebacks carry no locals, so no request or config value beyond an exception message |
| `display.py:106` | fixed text | safe |
| `display.py:99` | record `status` | plugin-written, not user text |
| `display.py:140-201`, `library.py:286` | the configured path plus an OS error | path only; a URL-shaped `suggestionsPath` would echo, the same class as finding 1 but a far-fetched value |
| `embed.py:121-145`, `rerank.py:62-93` | the configured endpoint | finding 2 |
| `embed.py:53` | a redirect target (`req.full_url`) | server-supplied, refused before use, debug level |
| `library.py:165,172,176,218` | counts and fixed text | safe |

No request text is logged anywhere: the lines above carry config values, record fields or fixed text.
The two residuals that can carry a credential are findings 1 and 2.

## Docs check for the round-2 finding

The round-2 docs gap is closed: `README.md:240-241` now says rejected settings are logged once and
URLs are reduced to scheme and host (finding 1 falsifies "never", not the mechanism),
`README.md:514-515` and `README.md:537-538` describe the caption, `spec.md:154-156` adds the caption
to D12, and `tickets/T8-display-names.md:133-135` replaces "says which turn it came from" with the
caption format and its suppression rules. Those match the code as verified in code-02.

## Per-file verdicts

- `hermes_skill_router/config.py` - FINDINGS 1, 2 (sanitizer scheme-dependent; base URL query accepted and echoed)
- `tests/test_config.py` - PASS (no test for the no-scheme userinfo or a query-bearing base URL)
- `README.md` - FINDINGS 1 (line 240 claim is stronger than the code)
- `docs/work/router-sync/spec.md` - PASS (D12 caption text matches the code)
- `docs/work/router-sync/tickets/T8-display-names.md` - PASS
- `hermes_skill_router/catalog.py`, `inject.py`, `router.py`, `desktop/plugin.js`, `tui/skill-router.mjs` - unchanged since code-02, verdicts stand
- `hermes_skill_router/embed.py`, `rerank.py`, `display.py`, `library.py`, `__init__.py`, `dashboard/plugin_api.py` - audited only, not changed this round; finding 2 concerns their log lines

## The three earlier findings

- code-01 finding 1 (one-shot header): closed in code-02, unchanged.
- code-01 finding 2 (display caption): closed in code-02, docs closed in round 3.
- code-01 finding 3 (warnings and notes): closed in code-02; its disclosure follow-up (code-02 finding 1)
  is closed for scheme-qualified URLs and open for userinfo without `://` (finding 1 here).

## Limits

- Static and module-level: no live restart or provider traffic. The leak probes write no credential
  into the live profile; the evidence is the captured log records above.
- Finding 2 used a closed loopback port and debug logging; it did not send data off the host.
- The audit covers `logger.*` calls, not exception messages raised by third-party libraries.
