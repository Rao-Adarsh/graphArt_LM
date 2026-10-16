"""Model backends for the prompting experiments.

Every backend is a callable taking (system, user) and returning the reply text.
Adding one means adding a class here and a line in `make_backend` -- nothing in
the runner or the scorer changes.

Note on reproducibility: the current hosted Claude models no longer expose
`temperature` / `top_p` (they are rejected), so a run is not bit-deterministic.
What we record instead, with every result, is the model id and the effort level.
Local models via Ollama do still take temperature, and we pin it to 0.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

ANTHROPIC_DEFAULT = "claude-opus-5"
OLLAMA_DEFAULT = "qwen2.5-coder:1.5b"
OLLAMA_URL = "http://localhost:11434/api/generate"
OPENROUTER_DEFAULT = "qwen/qwen-2.5-coder-32b-instruct"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class BackendError(RuntimeError):
    pass


def resolve_key(name: str) -> str | None:
    """Find an API key without requiring a shell restart.

    `setx` on Windows writes the variable to the registry, but processes that
    are already running -- and anything they spawn -- keep the environment they
    started with. So a key set after this session began is invisible to
    os.environ until everything is restarted. Falling back to the registry
    avoids that, and avoids keeping a second copy of the secret in a file.
    """
    value = os.environ.get(name)
    if value:
        return value
    if sys.platform != "win32":
        return None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            return winreg.QueryValueEx(key, name)[0] or None
    except (ImportError, OSError):
        return None


class MockBackend:
    """Returns a canned reply. Lets the whole pipeline be tested with no credentials."""

    name = "mock"

    # The canned reply is in the active domain's format on purpose: a mock that
    # emits the wrong domain fails to parse, which makes every smoke test look
    # like a failure and hides real plumbing breakage.
    DEFAULT_REPLY = ("INPUTS 1\n"
                     "NODES\n"
                     "0 placeholder x [1,4]\n"
                     "1 call_module fc Linear(in=4,out=2,bias=True) [1,2]\n"
                     "2 output output [1,2]\n"
                     "EDGES\n"
                     "0 -> 1 : arg0\n"
                     "1 -> 2 : arg0")

    def __init__(self, model: str = "echo", reply: str = "", **_):
        self.model = model
        self._reply = reply or self.DEFAULT_REPLY

    def __call__(self, system: str, user: str) -> str:
        return self._reply


class AnthropicBackend:
    """Hosted Claude models through the official SDK (`pip install anthropic`)."""

    name = "anthropic"

    def __init__(self, model: str = ANTHROPIC_DEFAULT, *, effort: str = "high",
                 max_tokens: int = 8000, **_):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise BackendError(
                "the anthropic package is not installed -- "
                "run: .venv/Scripts/pip install -r requirements-pipeline.txt"
            ) from exc
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic()

    def __call__(self, system: str, user: str) -> str:
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "output_config": {"effort": self.effort},
            "messages": [{"role": "user", "content": user}],
        }
        if system:
            # One cache breakpoint on the instruction prefix: it is identical for
            # every sample in a run, so the few-shot prompts get it for ~10% cost.
            kwargs["system"] = [{"type": "text", "text": system,
                                 "cache_control": {"type": "ephemeral"}}]
        response = self._client.messages.create(**kwargs)
        if response.stop_reason == "refusal":
            raise BackendError(f"model refused: {response.stop_details}")
        return "".join(b.text for b in response.content if b.type == "text")


class OllamaBackend:
    """A local open model served by Ollama. No API key, no rate limit, free."""

    name = "ollama"

    # Small local models ramble, and on a CPU every wasted token is ~0.3s. Our
    # targets are ~200 tokens, so 1500 is generous and caps a runaway reply at
    # a few minutes instead of forty.
    def __init__(self, model: str = OLLAMA_DEFAULT, *, max_tokens: int = 1500,
                 url: str = OLLAMA_URL, **_):
        self.model = model
        self.max_tokens = max_tokens
        self.url = url

    def __call__(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "prompt": user,
            "stream": False,
            "options": {"temperature": 0, "num_predict": self.max_tokens},
        }
        if system:
            payload["system"] = system
        request = urllib.request.Request(
            self.url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                return json.loads(response.read())["response"]
        except urllib.error.URLError as exc:
            raise BackendError(
                f"could not reach Ollama at {self.url} ({exc}) -- "
                f"is it running, and has `ollama pull {self.model}` been done?"
            ) from exc


class OpenRouterBackend:
    """Everything else, through one key: open-weight models, GPT, Gemini.

    Claude deliberately does NOT go through here -- it has its own adapter above,
    which gets prompt caching, effort control and refusal handling that this
    OpenAI-shaped endpoint cannot express.

    Find model ids with:  python -m pipeline.list_models qwen coder
    """

    name = "openrouter"

    def __init__(self, model: str = OPENROUTER_DEFAULT, *, max_tokens: int = 8000,
                 api_key: str | None = None, max_retries: int = 4, **_):
        self.api_key = api_key or resolve_key("OPENROUTER_API_KEY")
        if not self.api_key:
            raise BackendError(
                "OPENROUTER_API_KEY is not set -- create a key at "
                "https://openrouter.ai/keys, then:  setx OPENROUTER_API_KEY sk-or-..."
            )
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def __call__(self, system: str, user: str) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": user}]
        body = {"model": self.model, "messages": messages,
                "temperature": 0, "max_tokens": self.max_tokens}
        request = urllib.request.Request(
            OPENROUTER_URL, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}",
                     "X-Title": "graphArt-LM (IE 643 course project)"})

        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    payload = json.loads(response.read())
                break
            except urllib.error.HTTPError as exc:
                # 429 is expected on the free tier; 5xx is the provider wobbling.
                if exc.code not in (429, 500, 502, 503, 504) or attempt == self.max_retries:
                    detail = exc.read().decode("utf-8", "replace")[:300]
                    raise BackendError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
                wait = float(exc.headers.get("Retry-After") or 2 ** attempt * 3)
                print(f"      HTTP {exc.code}, retrying in {wait:.0f}s "
                      f"({attempt + 1}/{self.max_retries})", flush=True)
                time.sleep(wait)
            except urllib.error.URLError as exc:
                if attempt == self.max_retries:
                    raise BackendError(f"could not reach OpenRouter: {exc}") from exc
                time.sleep(2 ** attempt * 3)

        if "error" in payload and not payload.get("choices"):
            raise BackendError(f"OpenRouter error: {payload['error']}")
        message = payload["choices"][0]["message"]
        # Some reasoning models put the answer in `content` and the working in
        # `reasoning`; we only ever want the answer.
        content = (message.get("content") or "").strip()
        if not content:
            # Free tiers sometimes return an empty completion (truncated, or the
            # whole budget went to reasoning tokens). Treat that as a failure so
            # it is recorded as one and never cached as a valid empty answer.
            finish = payload["choices"][0].get("finish_reason")
            raise BackendError(f"empty completion (finish_reason={finish!r})")
        return content


BACKENDS = {"anthropic": AnthropicBackend, "ollama": OllamaBackend,
            "openrouter": OpenRouterBackend, "mock": MockBackend}
DEFAULT_MODELS = {"anthropic": ANTHROPIC_DEFAULT, "ollama": OLLAMA_DEFAULT,
                  "openrouter": OPENROUTER_DEFAULT, "mock": "echo"}


def make_backend(name: str, model: str | None = None, **kwargs):
    if name not in BACKENDS:
        raise BackendError(f"unknown backend {name!r}; choose from {sorted(BACKENDS)}")
    return BACKENDS[name](model or DEFAULT_MODELS[name], **kwargs)
