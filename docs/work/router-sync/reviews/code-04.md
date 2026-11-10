# code-04: verification of the two round-3 findings

Round 4. Both round-3 findings are fixed in `hermes_skill_router/config.py`: credential shapes without
`://` are redacted, and a base URL carrying a query or fragment is rejected. This verdict checks the
two functions with their call sites, the tests, and a probe over the six credential shapes plus the
query case. Read-only: no source file was changed by this review.

## Reviewed revision

| file | sha256 (first 16) | mtime |
| --- | --- | --- |
| hermes_skill_router/config.py | 078ffdc73b8ed3ec | 09:12:55 |
| tests/test_config.py | e9c3e7b0557cdb88 | 09:12:55 |

`README.md` (`cd17892afbb19360`, 09:05) and everything else are unchanged since code-03.

## Tests

- `python3 -m pytest tests/test_config.py -q`, exit 0: `88 passed in 0.08s` (was 67; the new cases
  include `test_a_base_url_with_a_query_or_fragment_is_rejected_without_echoing_it` and
  `test_credential_shapes_without_a_url_scheme_are_redacted`, which covers `alice:hunter2@...`,
  `//alice:hunter2@...`, `https:/alice:hunter2@...` and `///alice:hunter2@...`).
- `python3 -m pytest -q`, exit 0: `441 passed in 4.97s` (was 420).

## The two functions

`is_safe_base_url` (`config.py:127-150`) now returns false when `parts.query or parts.fragment`
(`:148`), after the scheme, host and userinfo checks and before the loopback rule. Its docstring
states the reason: the request path is appended to the root, so a query or fragment would be
swallowed and then echoed on failure.

`_redact_url` (`config.py:181-201`) keeps the old `://` branch and adds
`_carries_credentials` (`:162-178`): strip a scheme prefix (`_SCHEME_PREFIX`, `:156`) or a leading
`//`, cut the remainder at the first `/`, `?` or `#` (`_AUTHORITY_END`, `:159`), and treat a
remaining `@` as userinfo. Such a value becomes `<redacted URL>`; a value with neither shape passes
through unchanged. `_redact_value` and `_shown` were not changed, so every note and the
unknown-key path inherit the two rules.

## Probe

`python3 /home/wilder/Trash/router-review/round4/probe.py`, captured log records for both endpoint
keys and every shape:

```
'https://alice:hunter2@example.com/v1'             -> 'https://example.com'
'http://alice:hunter2@127.0.0.1:8012/v1/embed'     -> 'http://127.0.0.1:8012'
'//alice:hunter2@example.com/v1'                   -> '<redacted URL>'
'https:/alice:hunter2@example.com/v1'              -> '<redacted URL>'
'alice:hunter2@example.com/v1'                     -> '<redacted URL>'
'https://alice:hunter2@'                           -> '<redacted URL>'
'smtp://alice:hunter2@example.com:587'             -> 'smtp://example.com:587'
'user:pw@host'                                     -> '<redacted URL>'
'https://example.com/v1?api_key=sk-live-123'       -> 'https://example.com'
'http://127.0.0.1:9/x?token=SECRET-VALUE'          -> 'http://127.0.0.1:9'
```

Every note and every log record for those values on `embedBaseUrl` and `rerankBaseUrl` was checked
for `alice`, `hunter2`, `pw`, `sk-live-123` and `SECRET-VALUE`: no occurrence (the probe's `LEAKS`
list is empty). The query case in detail:

```
effective base: http://127.0.0.1:8012   (falls back to the default)
note: embedBaseUrl: 'http://127.0.0.1:9' is not a safe endpoint URL (... and a query or fragment
      is not allowed); using default 'http://127.0.0.1:8012'
token present: False
```

Validator behaviour: `http://127.0.0.1:8012` and `https://example.com/v1` are accepted;
`http://127.0.0.1:8012?x=1`, `https://example.com/v1#frag`, `https://u:p@example.com/v1` and
`http://example.com/v1` are rejected. Non-credentialed strings are unchanged
(`/home/u/.cache/x.json`, `plain text`, `a/b@c`).

The live profile sets `embedBaseUrl: 'http://127.0.0.1:8012'` (`profiles/web/cordis.patch.yml:462`),
which carries no query, so the new rejection does not change the running setup.

## Verdict on the round-3 findings

- Finding 1 (userinfo without `://` reached the log): closed. All six credential shapes are reduced
  to their origin or `<redacted URL>` on both endpoint keys, and `README.md:241` now holds.
- Finding 2 (a query in a base URL was accepted, mangled the endpoint and was echoed at debug level):
  closed. The query and fragment are rejected at validation, the value falls back to the default, and
  the note shows only `scheme://host:port`, so the debug endpoint echoes in `embed.py` and `rerank.py`
  can no longer see a token. Both findings are closed.

## Finding

### 1. The README's endpoint rule and the two URL rows do not state the query and fragment rejection

Where: `README.md:297-301` ("Endpoint URL rule ... A URL carrying credentials is rejected ..."),
`README.md:269` and `README.md:273` (bound column: "no URL credentials"), `README.md:419-420` (setup
step 3: "rejects a plain http URL for a remote host, a URL with credentials, and any redirect").

What breaks: the code now also rejects a base URL that carries a query or a fragment
(`config.py:148`), and spec D4 (`spec.md:96-98`) states that rule, but the operator-facing README
never mentions it. An operator who writes `embedBaseUrl: https://host/v1?token=x` gets a silent
fallback to the loopback default plus one warning line, with no README sentence to explain why the
value was rejected. Nothing in the tests reads the bound column's text, so this cannot fail visibly.

Smallest fix: add "no query or fragment" beside "no URL credentials" at README:269 and README:273,
the same clause to the rule at README:297-301 and to step 3 at README:419-420. One clause each.

## Observations, not findings

- The new rule redacts any value whose authority holds an `@`, including non-URL values such as a
  malformed provider string `cohere@v2` or a path `@v1/embeddings`; their notes then read
  `<redacted URL>` instead of showing the shape. That is a deliberate trade (a note is diagnostic
  text, not a value), and ordinary malformed values keep their notes per
  `test_notes_about_ordinary_values_are_unchanged`.
- `_carries_credentials` is private; if other modules ever log configured values (the audit in
  code-03 lists the path echoes in `display.py` and `library.py`), they should reuse `_shown` rather
  than re-derive the rule.

## Limits

- Static and module-level: no live restart and no provider traffic. The probe wrote no credential
  into the live profile; the evidence is the captured log records above.
- Only `config.py` and its tests changed this round; the audit in code-03 still stands for the other
  log lines.
