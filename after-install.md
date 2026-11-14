# hermes-skill-router: after install

The router needs two local model servers: one embeddings server on `127.0.0.1:8012` and one
rerank server on `127.0.0.1:8010`. One command downloads, installs and starts both of them.
Run it from the checkout root after `hermes plugins install`:

```bash
python3 tools/setup_models.py
```

The script asks where the model files should go, and offers
`~/.cache/hermes-skill-router/models` as the default. It downloads only what the two bundled
servers load, checks the two weight files against the pinned sha256 digests in
`service/model-digests.txt`, and asks again when a file does not match. It then starts both
servers and prints the commands to manage, start and stop them.

It never runs automatically: `hermes plugins install` does not start it, and nothing in the
plugin calls it. Run it yourself once, or again after a model update. It does not change any
plugin setting.

On Linux the script installs two systemd user units,
`hermes-skill-router-embed.service` and `hermes-skill-router-decider.service`, enables them
and prints their `systemctl --user` start, stop and status commands. It needs a running
systemd user session. On macOS and Windows it starts both bundled server scripts with
`python3` and prints the commands for that platform.

If you bring your own endpoints, set `embedBaseUrl` and `rerankBaseUrl` in the plugin settings
and skip this step. The plugin accepts any server that answers its embeddings or rerank
request, with or without a `/health` route.

[README.md](README.md) carries the install steps, [default-models.md](default-models.md)
covers the two models, and [technical-documentation.md](technical-documentation.md) carries
both endpoint protocols and every setting.
