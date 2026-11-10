# T3 Skill index, embeddings client, cache

Status: ready for agent. Repo: `hermes-skill-router`. Work root: `docs/work/router-sync/`.

## Delivers

The index built from `Partition.routed`, the rows the plugin removes and may suggest, the embeddings
HTTP client, the persisted cache, and the in-process memo that keeps a warm request off the network.

## Covers

Stories S2, S6, S9. Requirements R2, R5, R6, R8, R11. Modules M4, M5. Contracts C4, C5.

## Exclusive files

- `hermes_skill_router/library.py`
- `hermes_skill_router/embed.py`
- `tests/test_library.py`
- `tests/test_embed_client.py`
- `tests/fake_endpoint.py` (shared fake server helper, created here and reused by T4)

## Blockers

T2, for `Config`, `strip_catalog` and the package layout.

## What it must do

1. `load_or_build(cfg, rows, embedder)`: take the `routed` half of the `Partition` that
   `strip_catalog` returned from the live system message and build the index from those rows only.
   Do not walk the filesystem for `SKILL.md` files: the catalog is the set of skills Hermes will
   actually let the model open, including plugin skills and the gated rows, under the names
   `skill_view` accepts. An eager skill never enters the index, so it can never be routed.
2. `fingerprint(rows, cfg)`: sha256 over every row's document text and full description in catalog
   order, plus `cfg.embed_model_id`, a document format version constant, and
   `cfg.embed_base_url + cfg.embed_path`. Every routed-row change invalidates, including a description
   edit past `max_description_chars`, because the cached descriptions are the rerank document text and
   the block text. A changed model id or a repointed endpoint invalidates too.
3. `load_or_build` returns the cache when `cfg.cache_path` names a readable JSON file whose `version`
   and `fingerprint` match, whose `dim` is a positive integer equal to the width of every stored
   vector, and whose entry count equals the live deduped routed row count; a cache that fails any of
   those is rebuilt. Otherwise it embeds every row in chunks of at most `BATCH_TEXTS` (16) texts per
   request, so `embedTimeoutMs` bounds one request instead of the whole build and a failed chunk fails
   the build, writes the cache atomically (temp file in the same directory, then `os.replace`), and
   returns the index. A cache write failure logs once and still returns the in-memory index. Empty rows return `None`. The endpoint's real dimension cannot be
   checked without an embedding call, and the cache hit must make none, so `dim` is validated against
   the stored vectors and the live row count only.
4. Memoise the index per fingerprint behind a `threading.Lock`, so two concurrent requests embed the
   library once.
5. `embed_texts(cfg, texts)`: `POST` JSON `{"model": cfg.embed_model_id, "input": [texts]}` to
   `cfg.embed_base_url + cfg.embed_path` with `urllib.request`, a timeout from `cfg.embed_timeout_ms`,
   and `Content-Type: application/json`. Build the request through an opener that refuses redirects:
   `urllib.request.build_opener` with a `HTTPRedirectHandler` subclass whose `redirect_request`
   returns `None` and raises, because the default opener follows a 301, 302 or 303 on a POST to a new
   host and would send the user's text there (D4). Expose that opener from the module so T4's rerank
   client reuses it instead of building a second policy. Return the vectors in input order, or `None`
   on
   connection error, HTTP error including any 3xx, timeout, a body without `data`, or a length
   mismatch. No retry, no partial result.
6. `embed_texts` keeps one health flag per endpoint with a `time.monotonic()` stamp: after a failure,
   a later call within `cfg.health_check_ms` returns `None` without opening a socket; the first call
   after the window probes again.

## Verification

```bash
python3 -m pytest tests/test_library.py tests/test_embed_client.py -q
```

The fake endpoint in `tests/fake_endpoint.py` is a `http.server` thread with switchable behaviour
(200, 500, slow response, wrong vector count). Every client test uses it, so the suite needs no
model, no GPU and no network.

## Acceptance criteria

- [ ] A cache hit performs zero embedding calls (asserted by a counting fake embedder).
- [ ] Changing one routed row's description (including only past `maxDescriptionChars`) or name,
      `embedModelId`, `embedPath` or `embedBaseUrl` each changes the fingerprint and forces a rebuild,
      while changing a kept row alone leaves it unchanged.
- [ ] `rows` with a duplicate name are de-duplicated before embedding, keeping the first occurrence.
- [ ] An eager skill is absent from the index, proved by a partition where it is the only kept row.
- [ ] A name from a demoted `[names only]` line is absent from the index and from every rerank document.
- [ ] `embed_texts` returns `None` for 500, timeout, and a short `data` list, and never raises.
- [ ] A fake endpoint that answers 302 or 307 to the POST makes `embed_texts` return `None`, and the
      fake records no follow-up request, including when the `Location` points at another host.
- [ ] Two threads calling `load_or_build` on a cold cache embed once, proved by the counter.
- [ ] The cache file is valid JSON and survives being read by a second process.
- [ ] `dim` is checked as far as a cache can prove it: a cache whose stored vectors contradict its
      `dim`, whose row count disagrees with the live routed rows, or whose `version` differs, is
      rebuilt, not trusted.
- [ ] A description longer than `maxDescriptionChars` is clipped in the embedding document text,
      exactly as the DSH `documentText` clips it.
- [ ] A cold build of more rows than one chunk makes ceil(rows / 16) embedder calls; each 16-row
      request fits within the 6000 ms default (measured 1.2 to 2.0 s, with one 2.4 s repeat peak), and
      the full 27-request build of the live 424-row catalog takes about 38 seconds.

## Notes

- The document text sent to the embedder is `name + "\n" + description` clipped to
  `cfg.max_description_chars`, the shape and clip DSH's `documentText` uses. The texts are identical
  only while both sides carry the same value; both default to 400, and each side's own setting wins.
  The shortlist is ranked by cosine over this text before reranking, so a different clip would change
  the shortlist; match DSH instead of inventing a second budget. Keep the helper in one place so T5 and
  the DSH side stay comparable.
- Descriptions come from the parsed catalog rows, so the index never disagrees with what the model
  was offered.
