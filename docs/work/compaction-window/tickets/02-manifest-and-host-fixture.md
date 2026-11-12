# 02: The manifest declares the hook, the floor and the release, and the host tests read the running install

**What to build:** the plugin's manifest tells the truth about what it consumes and what it needs. It
declares the `post_compaction` hook it registers, requires a host release that has the event, and moves
the plugin version to a breaking release because a floor rose and a detector was removed. The two test
modules that import the host stop naming a hard-coded stale source tree and resolve the install the
running `hermes` launcher reports, so the hook contract is checked against a host that can deliver the
event; when no host resolves, the test skips and the skip reason names the path it looked for.

**Blocked by:** 01

**Verify:**

```sh
cd /home/wilder/git-repos/hermes-skill-router
python3 -m pytest tests/test_plugin_registration.py tests/test_middleware_contract.py -q
```

**Status:** ready-for-agent

- [ ] The manifest declares `provides_hooks` as exactly `["post_compaction"]`, `requires_hermes` as
      `">=0.22.3"`, and `version` as the quoted literal `"0.2.0"`, and `provides_middleware` and
      `provides_tools` keep their values.
- [ ] The registration test asserts the hook the plugin registers, by name, and no test asserts an empty
      hook list any more.
- [ ] The registration test asserts the four manifest fields above against a strict parse that still
      refuses duplicate keys.
- [ ] Both host-importing test modules resolve the running install from the `hermes` launcher's own
      report, and neither names a fixed stale directory.
- [ ] A missing host produces a skip whose reason names the path that was looked for, and the skip does
      not hide a failing assertion.
- [ ] One test forces that path: it makes the resolution find no host and asserts the skip reason, so the
      missing-host branch runs in the suite instead of being assumed from reading the code.
- [ ] The middleware contract test still passes against the resolved host, including the var-keyword
      requirement on the callback.

**Deliverables**

- `plugin.yaml`
- `tests/test_plugin_registration.py`, `tests/test_middleware_contract.py`
