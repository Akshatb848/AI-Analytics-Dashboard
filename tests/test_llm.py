"""GLM query planning for Ask Data, tested against a fake Z.ai API."""
import json
from pathlib import Path

import pytest
import requests
from streamlit.testing.v1 import AppTest

import app as dashboard
from analytics import llm
from analytics.data_utils import generate_sample_data
from analytics.llm import GLMClient, LLMConfig, LLMError, describe_schema, extract_json

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")
FAKE_KEY = "test-key-not-real"


class FakeResponse:
    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self._body = body

    def json(self):
        if self._body is None:
            raise ValueError("no JSON")
        return self._body


class FakeZai:
    """Records requests and replies with a canned model message or error."""

    def __init__(self, reply=None, status_code=200, error=None):
        self.reply, self.status_code, self.error = reply, status_code, error
        self.calls = []

    def __call__(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        if self.error:
            raise self.error
        if self.status_code != 200:
            return FakeResponse(self.status_code, {"error": {"message": "rate limit reached"}})
        return FakeResponse(200, {"choices": [{"message": {"content": self.reply}}]})


@pytest.fixture
def fake_api(monkeypatch):
    def install(**kwargs):
        fake = FakeZai(**kwargs)
        monkeypatch.setattr(llm.requests, "post", fake)
        return fake
    monkeypatch.setenv("ZAI_API_KEY", FAKE_KEY)
    monkeypatch.delenv("ZAI_BASE_URL", raising=False)
    monkeypatch.delenv("ZAI_MODEL", raising=False)
    dashboard.plan_with_llm.clear()
    yield install
    dashboard.plan_with_llm.clear()


@pytest.fixture(scope="module")
def sample():
    df = generate_sample_data(1000)
    return df, dashboard.data_fingerprint(df), dashboard.detect_numeric_columns(df), \
        dashboard.detect_categorical_columns(df), "date"


def ask(sample, query, use_llm=True):
    df, fingerprint, numeric, categorical, date_col = sample
    return dashboard.answer_question(df, fingerprint, query, numeric, categorical, date_col, use_llm)


# -----------------------------------------------------------------------------
# Configuration and client
# -----------------------------------------------------------------------------

def test_no_key_means_no_llm(monkeypatch):
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    assert LLMConfig.from_env() is None
    assert LLMConfig.from_env(secrets=lambda name: None) is None


def test_config_reads_env_then_secrets_and_hides_the_key(monkeypatch):
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    config = LLMConfig.from_env(secrets=lambda name: {"ZAI_API_KEY": "from-secrets"}.get(name))
    assert config.api_key == "from-secrets"
    assert config.model == "glm-4.5-flash"
    assert config.base_url == "https://api.z.ai/api/paas/v4"
    assert "from-secrets" not in repr(config)

    def broken_secrets(name):
        raise FileNotFoundError("no secrets.toml")
    assert LLMConfig.from_env(secrets=broken_secrets) is None


def test_client_sends_an_openai_compatible_request(fake_api):
    fake = fake_api(reply="hi")
    config = LLMConfig.from_env()
    assert GLMClient(config).chat([{"role": "user", "content": "hello"}]) == "hi"
    call = fake.calls[0]
    assert call["url"] == "https://api.z.ai/api/paas/v4/chat/completions"
    assert call["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"
    assert call["json"]["model"] == "glm-4.5-flash"
    assert call["json"]["thinking"] == {"type": "disabled"}
    assert call["timeout"]


@pytest.mark.parametrize("kwargs, message", [
    (dict(status_code=429), "HTTP 429: rate limit reached"),
    (dict(error=requests.ConnectionError("boom")), "could not reach the model API"),
])
def test_client_errors_are_clear_and_never_include_the_key(fake_api, kwargs, message):
    fake_api(**kwargs)
    with pytest.raises(LLMError) as excinfo:
        GLMClient(LLMConfig.from_env()).chat([{"role": "user", "content": "x"}])
    assert message in str(excinfo.value)
    assert FAKE_KEY not in str(excinfo.value)


@pytest.mark.parametrize("reply", [
    '{"type": "aggregate"}',
    'Here is the plan:\n```json\n{"type": "aggregate"}\n```',
    'Sure! {"type": "aggregate"} Hope that helps.',
])
def test_extract_json_tolerates_fences_and_chatter(reply):
    assert extract_json(reply) == {"type": "aggregate"}


@pytest.mark.parametrize("reply", ["not json", "[1, 2]", ""])
def test_extract_json_rejects_non_objects(reply):
    with pytest.raises(LLMError):
        extract_json(reply)


def test_schema_sent_to_the_model_has_no_raw_rows(sample):
    df, _, numeric, categorical, date_col = sample
    schema = describe_schema(df, numeric, categorical, date_col)
    text = json.dumps(schema)
    assert str(round(df["sales"].iloc[0], 4)) not in text
    region = next(c for c in schema["columns"] if c["name"] == "region")
    assert region["kind"] == "category" and "North" in region["values"]
    sales = next(c for c in schema["columns"] if c["name"] == "sales")
    assert set(sales) == {"name", "kind"}


# -----------------------------------------------------------------------------
# Answering questions with the model's plan, and falling back
# -----------------------------------------------------------------------------

def test_model_plan_is_validated_and_run(fake_api, sample):
    fake = fake_api(reply=json.dumps({
        "type": "aggregate", "aggregation": "average", "metric": "profit", "groupby": "region",
        "filters": [{"column": "channel", "operator": "==", "value": "Online"}],
        "limit": 3, "sort_order": "desc",
    }))
    result = ask(sample, "which 3 regions earn the most profit per order online?")
    df = sample[0]
    expected = df[df["channel"] == "Online"].groupby("region")["profit"].mean().nlargest(3)
    assert result["source"] == "glm-4.5-flash"
    assert result["llm_note"] is None
    assert list(result["data"]["region"]) == list(expected.index)
    assert "Channel = Online" in result["text"]

    # The question and the schema go to the model; the answer is cached per question
    sent = fake.calls[0]["json"]["messages"][-1]["content"]
    assert "which 3 regions" in sent and '"region"' in sent
    ask(sample, "which 3 regions earn the most profit per order online?")
    assert len(fake.calls) == 1


def test_model_plan_with_unknown_columns_is_cleaned(fake_api, sample):
    fake_api(reply=json.dumps({"type": "aggregate", "aggregation": "sum", "metric": "salary",
                               "groupby": "region", "filters": [{"column": "city", "operator": "==",
                                                                 "value": "Pune"}]}))
    result = ask(sample, "total by region")
    assert result["source"] == "glm-4.5-flash"
    assert result["intent"]["metric"] is None and result["intent"]["filters"] == []
    assert result["success"]


@pytest.mark.parametrize("kwargs, note", [
    (dict(status_code=429), "was unavailable (model API returned HTTP 429"),
    (dict(reply="I cannot help with that."), "was unavailable (the model's reply was not valid JSON)"),
    (dict(reply='{"type": null}'), "couldn't map this question"),
])
def test_falls_back_to_the_built_in_parser(fake_api, sample, kwargs, note):
    fake_api(**kwargs)
    result = ask(sample, "Top 3 regions by profit")
    assert result["source"] == "built-in parser"
    assert note in result["llm_note"]
    assert len(result["data"]) == 3


def test_failed_calls_are_retried_not_cached(fake_api, sample):
    fake = fake_api(status_code=500)
    ask(sample, "total sales by region")
    ask(sample, "total sales by region")
    assert len(fake.calls) == 2


def test_llm_toggle_off_skips_the_model(fake_api, sample):
    fake = fake_api(reply='{"type": "aggregate"}')
    result = ask(sample, "total sales by region", use_llm=False)
    assert fake.calls == [] and result["source"] == "built-in parser"


# -----------------------------------------------------------------------------
# In the app
# -----------------------------------------------------------------------------

def test_app_uses_the_model_when_a_key_is_set(fake_api):
    fake_api(reply=json.dumps({"type": "aggregate", "aggregation": "sum", "metric": "sales",
                               "groupby": "channel"}))
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert at.toggle(key="use_llm").value is True
    at.text_input(key="nl_query").input("how is each channel doing?").run()
    assert not at.exception
    assert any("Sum of Sales by Channel" in m.value for m in at.markdown)
    assert any(e.label == "How this was answered: glm-4.5-flash" for e in at.expander)


def test_app_without_a_key_uses_the_built_in_parser(monkeypatch):
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not at.exception
    assert any("Set `ZAI_API_KEY`" in c.value for c in at.caption)
    at.text_input(key="nl_query").input("Total sales by region").run()
    assert any(e.label == "How this was answered: built-in parser" for e in at.expander)
