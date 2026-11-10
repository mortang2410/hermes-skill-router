# T6 DSH router uses the same embeddings endpoint

Status: ready for agent. Repo: `skill-router-and-jev` (a different repository from the plugin).
Work root there: `packages/dsh-skill-router/`.

## Delivers

An HTTP `TextEmbedder` for the DSH skill router and the config keys that select it, so DSH and
Hermes ask one endpoint for the same vectors. The ONNX embedder stays as the fallback.

## Covers

Stories S6. Requirements R6, R9. Module M10. Contract C10.

## Exclusive files

- `packages/dsh-skill-router/src/index/http-embedder.ts` (new)
- `packages/dsh-skill-router/src/index/embedding-index.ts` (the `EmbedIndexOptions` identity and
  `catalogFingerprint` input)
- `packages/dsh-skill-router/src/injection/router.ts` (config keys and wiring)
- `packages/dsh-skill-router/test/http-embedder.test.mjs` (new)
- `packages/dsh-skill-router/README.md` and `packages/dsh-skill-router/cordis.patch.yml` (the four new
  keys beside the existing `modelPath`)

The package has no `src/index/index.ts`; its barrel is `src/index.ts`, which does not re-export
`createBgeEmbedder`, so no re-export edit is expected. If a later build adds one, exporting
`createHttpEmbedder` there is in scope. `src/index.ts` itself stays out of scope: it still gates
suppression on a non-empty `modelPath` (line 834), which the switch depends on.

## Blockers

T1, for a live endpoint to check against.

## What it must do

1. Export `createHttpEmbedder(opts: HttpEmbedderOptions): TextEmbedder` satisfying
   `embedOne(text: string): Promise<readonly number[]>` from `src/index/embedding-index.ts:99`.
   One text per HTTP request, no batching, no retry, and reject rather than return a wrong vector.
2. Request body `{"model": modelId, "input": [text]}` to `baseUrl + (path ?? "/v1/embeddings")`,
   `Content-Type: application/json`, timeout `timeoutMs`. Read `data[0].embedding` and reject a body
   whose first entry has no numeric vector.
3. Add zod config keys beside `modelPath`: `embedBaseUrl` (default `""`), `embedPath` (default
   `/v1/embeddings`), `embedModelId` (default `bge-m3`), `embedTimeoutMs` (default `2000`, range
   100..60000).
4. Wire the embedder choice where `createBgeEmbedder` is constructed today
   (`src/injection/router.ts:450`): a non-empty `embedBaseUrl` selects the HTTP embedder; otherwise
   the existing ONNX path is unchanged, including its warning when `modelPath` is empty.
5. Keep the index fingerprint honest. `catalogFingerprint` today hashes `modelId` plus the catalog
   text (`src/index/embedding-index.ts:161-169`), so a repointed endpoint alone would silently reuse
   vectors from a different model. Add the selected embedder identity to the fingerprint input: the
   `embedModelId` and the HTTP `baseUrl` plus `path` when the HTTP embedder is used, the `modelPath`
   when the ONNX embedder is used. Extend `EmbedIndexOptions` with that identity rather than widening
   `modelId` silently, and keep the existing cache files valid for the ONNX path.
6. Document the four new keys beside `modelPath` in `packages/dsh-skill-router/README.md` and, where the
   package's config sample shows `modelPath`, in `packages/dsh-skill-router/cordis.patch.yml`. State
   that `modelPath` must stay set even when `embedBaseUrl` is used, because suppression is gated on it.

## Verification

In `packages/dsh-skill-router`:

```bash
npm run check
npm test
```

The new test uses `node:http` to stand up a local server and asserts: one request per `embedOne`,
the request body shape, that a 500 rejects, that a slow response rejects at the timeout, and that a
malformed body rejects.

Live, once the plugin's service is running:

```bash
curl -s 127.0.0.1:8012/health
node --test test/http-embedder.test.mjs
```

Then switch the profile row to the URL and restart DSH once (the host restart limit applies), and
compare the shortlist for one fixed query against the pre-change shortlist. The live switch is the
`skill-router` row of `/home/wilder/.dsh/profiles/web/cordis.patch.yml` (`id: skill-router`, `name:
'@local/dsh-skill-router'`): add `embedBaseUrl: http://127.0.0.1:8012` beside the existing `modelPath`
and leave `modelPath` in place as the fallback. That file is outside this repository and is edited by
the orchestrator after the code lands, not by the implementer; name the exact row and key in the
report.

## Acceptance criteria

- [ ] `npm run check` and `npm test` pass in the package.
- [ ] With `embedBaseUrl` empty, the ONNX path is byte-identical in behaviour to before.
- [ ] With `embedBaseUrl` set, a router boot returns a shortlist for a fixed query whose top 10
      matches the ONNX shortlist for the same query, or the difference is explained in the report.
- [ ] The fingerprint changes when `embedModelId` changes, when `embedBaseUrl` changes and when
      `embedPath` changes, each asserted separately.
- [ ] Switching from the ONNX embedder to the HTTP embedder with the same model id rebuilds the index
      instead of reusing the ONNX vectors.
- [ ] No new runtime dependency; `node:http`/`fetch` only.
- [ ] The package README and config sample document `embedBaseUrl`, `embedPath`, `embedModelId` and
      `embedTimeoutMs`, and still require a non-empty `modelPath`.

## Notes

- This repository is separate from the plugin. Commit there on its own branch state and do not
  copy the plugin's files into it.
- Report the profile key you switch and the exact restart you performed.
