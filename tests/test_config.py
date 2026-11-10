"""Contract C2: config defaults, bounds, notes and endpoint URL safety."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes_skill_router import config as config_module  # noqa: E402
from hermes_skill_router.config import (  # noqa: E402
    BOUNDS,
    DEFAULTS,
    Config,
    is_safe_base_url,
    load,
)

# The spec's config table, transcribed here on purpose: the test must not read its expectations
# from the module it checks.
SPEC_DEFAULTS = {
    "enabled": True,
    "eagerSkills": (),
    "embedBaseUrl": "http://127.0.0.1:8012",
    "embedPath": "/v1/embeddings",
    "embedModelId": "bge-m3",
    "embedTimeoutMs": 6000,
    "rerankBaseUrl": "http://127.0.0.1:8010",
    "rerankPath": "/v1/rerank",
    "rerankModel": "tasksource-decider-nano",
    "rerankProvider": "cohere",
    "documentPrefix": (
        "This skill supplies a procedure, tool, or domain-specific instructions that the agent would "
        "actually need to carry out the current task correctly. Skill: "
    ),
    "deferredSkills": (),
    "rerankTimeoutMs": 20000,
    "routingTimeoutMs": 30000,
    "shortlistSize": 50,
    "cutoff": 0.01,
    "maxSuggestions": 8,
    "maxDisplayNames": 12,
    "maxDescriptionChars": 400,
    "cachePath": "~/.cache/hermes-skill-router/index.json",
    "showSuggestions": True,
    "suggestionsPath": "~/.cache/hermes-skill-router/suggestions.json",
    "healthCheckMs": 60000,
}

SPEC_BOUNDS = {
    "embedTimeoutMs": (100, 60000),
    "rerankTimeoutMs": (100, 60000),
    "routingTimeoutMs": (1000, 60000),
    "shortlistSize": (1, 383),
    "cutoff": (0.0, 1.0),
    "maxSuggestions": (0, 50),
    "maxDisplayNames": (1, 50),
    "maxDescriptionChars": (3, 4000),
    "healthCheckMs": (0, 3600000),
}

FIELD_BY_KEY = {
    "embedTimeoutMs": "embed_timeout_ms",
    "rerankTimeoutMs": "rerank_timeout_ms",
    "routingTimeoutMs": "routing_timeout_ms",
    "shortlistSize": "shortlist_size",
    "cutoff": "cutoff",
    "maxSuggestions": "max_suggestions",
    "maxDisplayNames": "max_display_names",
    "maxDescriptionChars": "max_description_chars",
    "healthCheckMs": "health_check_ms",
    "embedBaseUrl": "embed_base_url",
    "rerankBaseUrl": "rerank_base_url",
    "rerankProvider": "rerank_provider",
}

CONFIG_FIELDS = {
    "enabled",
    "embed_base_url",
    "embed_path",
    "embed_model_id",
    "embed_timeout_ms",
    "rerank_base_url",
    "rerank_path",
    "rerank_model",
    "rerank_provider",
    "rerank_timeout_ms",
    "routing_timeout_ms",
    "shortlist_size",
    "cutoff",
    "max_suggestions",
    "max_description_chars",
    "cache_path",
    "health_check_ms",
    "show_suggestions",
    "suggestions_path",
    "max_display_names",
    "document_prefix",
    "eager_skills",
    "deferred_skills",
    "notes",
}

OUT_OF_BOUNDS = [
    ("embedTimeoutMs", 99),
    ("embedTimeoutMs", 60001),
    ("rerankTimeoutMs", 0),
    ("rerankTimeoutMs", 60001),
    ("routingTimeoutMs", 999),
    ("routingTimeoutMs", 60001),
    ("shortlistSize", 0),
    ("shortlistSize", 384),
    ("cutoff", -0.01),
    ("cutoff", 1.01),
    ("maxSuggestions", -1),
    ("maxSuggestions", 51),
    ("maxDisplayNames", 0),
    ("maxDisplayNames", 51),
    ("maxDescriptionChars", 2),
    ("maxDescriptionChars", 4001),
    ("healthCheckMs", -1),
    ("healthCheckMs", 3600001),
]

MALFORMED = [
    {"embedTimeoutMs": "2000"},
    {"shortlistSize": "fifty"},
    {"cutoff": {"value": 0.5}},
    {"cutoff": True},
    {"maxSuggestions": 1.5},
    {"enabled": 5},
    {"eagerSkills": "alpha"},
    {"eagerSkills": ["alpha", 3]},
    {"deferredSkills": {"alpha": True}},
    {"rerankProvider": "jina"},
    {"embedBaseUrl": None},
    {"cachePath": 7},
    {"documentPrefix": ""},
]


def test_defaults_cover_the_spec_table():
    assert set(DEFAULTS) == set(SPEC_DEFAULTS)
    for key, default in SPEC_DEFAULTS.items():
        assert DEFAULTS[key] == default, key


def test_bounds_cover_the_spec_table():
    assert BOUNDS == SPEC_BOUNDS


def test_config_fields_cover_the_contract():
    assert set(Config.__dataclass_fields__) == CONFIG_FIELDS


def test_load_none_returns_every_default():
    cfg = load(None)
    assert isinstance(cfg, Config)
    assert cfg.notes == ()
    assert cfg.enabled is True
    assert cfg.embed_base_url == "http://127.0.0.1:8012"
    assert cfg.rerank_base_url == "http://127.0.0.1:8010"
    assert cfg.embed_timeout_ms == 6000
    assert cfg.rerank_timeout_ms == 20000
    assert cfg.routing_timeout_ms == 30000
    assert cfg.shortlist_size == 50
    assert cfg.cutoff == 0.01
    assert cfg.max_suggestions == 8
    assert cfg.max_description_chars == 400
    assert cfg.max_display_names == 12
    assert cfg.health_check_ms == 60000
    assert cfg.show_suggestions is True
    assert cfg.eager_skills == ()
    assert cfg.deferred_skills == ()
    assert cfg.cache_path == "~/.cache/hermes-skill-router/index.json"
    assert cfg.suggestions_path == "~/.cache/hermes-skill-router/suggestions.json"
    assert cfg.rerank_provider == "cohere"


@pytest.mark.parametrize("key,bad", OUT_OF_BOUNDS)
def test_out_of_bounds_falls_back_to_the_default_with_a_note(key, bad):
    cfg = load({key: bad})
    assert getattr(cfg, FIELD_BY_KEY[key]) == SPEC_DEFAULTS[key]
    assert len(cfg.notes) == 1
    assert key in cfg.notes[0]


def test_in_bounds_values_are_kept():
    cfg = load(
        {
            "embedTimeoutMs": 100,
            "rerankTimeoutMs": 60000,
            "shortlistSize": 383,
            "cutoff": 1.0,
            "maxSuggestions": 0,
            "maxDisplayNames": 50,
            "maxDescriptionChars": 3,
            "healthCheckMs": 0,
        }
    )
    assert cfg.notes == ()
    assert (cfg.embed_timeout_ms, cfg.rerank_timeout_ms) == (100, 60000)
    assert cfg.shortlist_size == 383
    assert cfg.cutoff == 1.0
    assert cfg.max_suggestions == 0
    assert cfg.max_display_names == 50
    assert cfg.max_description_chars == 3
    assert cfg.health_check_ms == 0


@pytest.mark.parametrize("raw", MALFORMED, ids=lambda raw: ",".join(raw))
def test_malformed_values_never_raise(raw):
    cfg = load(raw)
    assert isinstance(cfg, Config)
    assert cfg.notes, raw
    for key in raw:
        assert any(key in note for note in cfg.notes), (key, cfg.notes)


def test_unknown_key_is_ignored_with_a_note_naming_it():
    cfg = load({"notAKey": 1})
    assert len(cfg.notes) == 1
    assert "notAKey" in cfg.notes[0]
    assert cfg.enabled is True
    assert cfg.shortlist_size == 50


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com:8012",
        "http://user:pw@127.0.0.1:8012",
        "ftp://127.0.0.1:8012",
        "not-a-url",
        "",
        "http://127.0.0.1:8012?key=value",
        "http://127.0.0.1:8012#fragment",
        "https://embed.example.com/?key=value",
    ],
)
def test_is_safe_base_url_rejects_unsafe_roots(url):
    assert is_safe_base_url(url) is False


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8012",
        "https://embed.example.com",
        "http://localhost:8012",
        "http://[::1]:8012",
    ],
)
def test_is_safe_base_url_accepts_loopback_http_and_any_https(url):
    assert is_safe_base_url(url) is True


@pytest.mark.parametrize("key", ["embedBaseUrl", "rerankBaseUrl"])
@pytest.mark.parametrize("bad", ["http://example.com:8012", "http://user:pw@127.0.0.1:8012"])
def test_unsafe_endpoint_url_falls_back_to_its_default_with_a_note(key, bad):
    cfg = load({key: bad})
    assert getattr(cfg, FIELD_BY_KEY[key]) == SPEC_DEFAULTS[key]
    assert len(cfg.notes) == 1
    assert key in cfg.notes[0]


@pytest.mark.parametrize("key", ["embedBaseUrl", "rerankBaseUrl"])
@pytest.mark.parametrize("good", ["http://127.0.0.1:8012", "https://embed.example.com"])
def test_safe_endpoint_url_is_kept(key, good):
    cfg = load({key: good})
    assert getattr(cfg, FIELD_BY_KEY[key]) == good
    assert cfg.notes == ()


def test_rerank_provider_only_accepts_cohere():
    assert load({"rerankProvider": "cohere"}).rerank_provider == "cohere"
    cfg = load({"rerankProvider": "jina"})
    assert cfg.rerank_provider == "cohere"
    assert any("rerankProvider" in note for note in cfg.notes)


def test_name_lists_are_stripped_deduplicated_and_empty_by_default():
    cfg = load({"eagerSkills": ["alpha", " beta ", "Alpha", ""], "deferredSkills": []})
    assert cfg.eager_skills == ("alpha", "beta", "Alpha")
    assert cfg.deferred_skills == ()
    assert cfg.notes == ()


def test_document_prefix_defaults_to_the_dsh_skill_prefix():
    cfg = load()
    assert cfg.document_prefix.startswith("This skill supplies a procedure")
    assert cfg.document_prefix.endswith("Skill: ")


def test_a_rejected_bound_logs_its_note_once(caplog):
    config_module._LOGGED_NOTES.clear()

    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.config"):
        first = load({"shortlistSize": 0})
        second = load({"shortlistSize": 0})
        third = load({"shortlistSize": 0})

    assert first.notes == second.notes == third.notes
    assert any("shortlistSize" in note for note in first.notes)
    warnings = [record for record in caplog.records if "shortlistSize" in record.getMessage()]
    assert len(warnings) == 1, [record.getMessage() for record in warnings]


@pytest.mark.parametrize("key", ["embedBaseUrl", "rerankBaseUrl"])
@pytest.mark.parametrize(
    "value,origin",
    [
        ("https://alice:hunter2@example.com/v1", "https://example.com"),
        ("http://alice:hunter2@example.com:8012/v1?token=secret#frag", "http://example.com:8012"),
    ],
)
def test_url_notes_log_only_the_scheme_and_host(key, value, origin, caplog):
    config_module._LOGGED_NOTES.clear()

    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.config"):
        cfg = load({key: value})

    assert len(cfg.notes) == 1
    note = cfg.notes[0]
    assert note.startswith(f"{key}: ")
    assert f"'{origin}'" in note
    assert "alice" not in note and "hunter2" not in note and "secret" not in note

    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1, messages
    assert origin in messages[0]
    assert "alice" not in messages[0] and "hunter2" not in messages[0]
    assert "secret" not in messages[0]


def test_a_url_shaped_value_in_another_setting_is_redacted_in_its_note(caplog):
    config_module._LOGGED_NOTES.clear()

    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.config"):
        cfg = load({"embedPath": "https://alice:hunter2@example.com/v1"})

    assert len(cfg.notes) == 1
    note = cfg.notes[0]
    assert note.startswith("embedPath: ")
    assert "'https://example.com'" in note
    assert "alice" not in note and "hunter2" not in note
    assert not any("hunter2" in record.getMessage() for record in caplog.records)


def test_a_url_shaped_value_inside_a_list_is_redacted_in_its_note():
    cfg = load({"eagerSkills": ["https://alice:hunter2@example.com/v1", 3]})
    assert len(cfg.notes) == 1
    note = cfg.notes[0]
    assert "'https://example.com'" in note
    assert "alice" not in note and "hunter2" not in note


def test_an_unknown_key_that_looks_like_a_url_is_redacted():
    cfg = load({"https://alice:hunter2@example.com/v1": 1})
    assert len(cfg.notes) == 1
    assert cfg.notes[0] == "https://example.com: unknown config key ignored"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ({"shortlistSize": 0}, "shortlistSize: 0 is outside 1..383; using default 50"),
        ({"cutoff": 2.5}, "cutoff: 2.5 is outside 0.0..1.0; using default 0.01"),
        ({"enabled": "maybe"}, "enabled: 'maybe' is not a valid boolean; using default True"),
    ],
)
def test_notes_about_ordinary_values_are_unchanged(raw, expected):
    cfg = load(raw)
    assert cfg.notes == (expected,)


@pytest.mark.parametrize("key", ["embedBaseUrl", "rerankBaseUrl"])
@pytest.mark.parametrize(
    "value,origin",
    [
        ("http://127.0.0.1:9/x?token=SECRET", "http://127.0.0.1:9"),
        ("http://127.0.0.1:9/x#SECRET", "http://127.0.0.1:9"),
        ("https://embed.example.com/v1?token=SECRET", "https://embed.example.com"),
    ],
)
def test_a_base_url_with_a_query_or_fragment_is_rejected_without_echoing_it(
    key, value, origin, caplog
):
    config_module._LOGGED_NOTES.clear()

    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.config"):
        cfg = load({key: value})

    assert getattr(cfg, FIELD_BY_KEY[key]) == SPEC_DEFAULTS[key]
    assert len(cfg.notes) == 1
    note = cfg.notes[0]
    assert note.startswith(f"{key}: ")
    assert f"'{origin}'" in note
    assert "SECRET" not in note and "token" not in note
    assert any(origin in record.getMessage() for record in caplog.records)
    assert not any("SECRET" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize(
    "value,secret",
    [
        ("alice:hunter2@example.com/v1", "alice"),
        ("alice:hunter2@example.com/v1", "hunter2"),
        ("//alice:hunter2@example.com", "alice"),
        ("//alice:hunter2@example.com", "hunter2"),
        ("https:/alice:hunter2@example.com", "alice"),
        ("https:/alice:hunter2@example.com", "hunter2"),
        ("user:pw@host", "user:pw"),
        ("user:pw@host", "pw@"),
        ("///alice:hunter2@example.com", "alice"),
        ("///alice:hunter2@example.com", "hunter2"),
        ("user:pw@host:8012", "user:pw"),
        ("user:pw@host:8012", "pw@"),
    ],
)
def test_credential_shapes_without_a_url_scheme_are_redacted(value, secret, caplog):
    config_module._LOGGED_NOTES.clear()

    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.config"):
        cfg = load({"embedBaseUrl": value})

    assert getattr(cfg, "embed_base_url") == SPEC_DEFAULTS["embedBaseUrl"]
    assert len(cfg.notes) == 1
    note = cfg.notes[0]
    assert note.startswith("embedBaseUrl: ")
    assert "not a safe endpoint URL" in note
    assert secret not in note
    assert not any(secret in record.getMessage() for record in caplog.records)


def test_a_credential_shaped_value_in_another_setting_is_redacted_in_its_note(caplog):
    config_module._LOGGED_NOTES.clear()

    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.config"):
        cfg = load({"embedPath": "alice:hunter2@example.com/v1"})

    assert len(cfg.notes) == 1
    note = cfg.notes[0]
    assert note.startswith("embedPath: ")
    assert "<redacted URL>" in note
    assert "alice" not in note and "hunter2" not in note
    assert not any("hunter2" in record.getMessage() for record in caplog.records)
