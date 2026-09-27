"""Small, bounded model client. Credentials come only from the environment."""
from __future__ import annotations

from dataclasses import dataclass, field
import ipaddress
import json
import math
import os
import socket
import time
from urllib import error, parse, request


class ModelError(ValueError):
    """A safe error code; never contains provider response bodies or credentials."""


def http_error_code(exc):
    """Classify a bounded error body into fixed labels, without logging its text."""
    base = 'model_http_' + str(exc.code)
    try:
        raw = exc.read(16385)
        if len(raw) > 16384:
            return base + ':body_too_large'
        value = json.loads(raw)
        detail = value.get('error', value) if isinstance(value, dict) else {}
        if isinstance(detail, str):
            detail = {'message': detail}
        if not isinstance(detail, dict):
            return base
        message = str(detail.get('message', detail.get('detail', ''))).lower()
        code = str(detail.get('code', '')).lower()
        fields = ('thinking', 'reasoning_effort', 'max_tokens', 'max_completion_tokens',
                  'response_format', 'messages', 'model', 'temperature', 'stream',
                  'endpoint', 'path', 'method', 'content-type', 'content_type',
                  'role', 'system', 'url', 'tool', 'json', 'encoding', 'proxy',
                  'header', 'query', 'request', 'scheme', 'client', 'session',
                  'version', 'store', 'extra_body')
        parameter = detail.get('param')
        labels = []
        for field in fields:
            if parameter == field or field in message:
                labels.append(field)
        for name, patterns in (
            ('unsupported', ('unsupported', 'not supported', 'not allowed', 'unknown parameter', 'unrecognized')),
            ('invalid_model', ('model_not_found', 'model does not exist', 'invalid model')),
            ('credential', ('invalid_api_key', 'authentication', 'api key')),
            ('missing_configuration', ('not configured', 'not connected', 'missing model', 'no model')),
            ('quota', ('insufficient_balance', 'insufficient_quota', 'insufficient balance')),
            ('context_limit', ('context length', 'context_length_exceeded')),
        ):
            if any(pattern in message or pattern in code for pattern in patterns):
                labels.append(name)
        return base + (':' + ','.join(labels) if labels else ':unclassified')
    except (ValueError, TypeError, AttributeError, OSError):
        return base


@dataclass(frozen=True)
class ObserverSettings:
    mode: str = "offline"
    model: str = ""
    base_url: str = ""
    api_key: str = field(default="", repr=False)
    api_mode: str = "chat"
    timeout: float = 8.0
    max_calls: int = 8
    max_output_tokens: int = 800
    reasoning_effort: str = "default"

    @classmethod
    def from_environment(cls, mode=None):
        key_name = os.environ.get("MODEL_API_KEY_ENV", "MODEL_API_KEY")
        result = cls(
            mode=mode or os.environ.get("OBSERVER_MODE", "offline").strip(),
            model=os.environ.get("MODEL_NAME", "").strip(),
            base_url=os.environ.get("MODEL_BASE_URL", "").strip().rstrip("/"),
            api_key=os.environ.get(key_name, "").strip(),
            api_mode=os.environ.get("MODEL_API_MODE", "chat").strip(),
            timeout=float(os.environ.get("OBSERVER_MODEL_TIMEOUT", "8")),
            max_calls=int(os.environ.get("OBSERVER_MAX_MODEL_CALLS", "8")),
            max_output_tokens=int(os.environ.get("OBSERVER_MAX_OUTPUT_TOKENS", "800")),
            reasoning_effort=os.environ.get("MODEL_REASONING_EFFORT", "default").strip(),
        )
        result.validate()
        return result

    def validate(self):
        if self.mode not in {"offline", "model"}:
            raise ModelError("invalid_observer_mode")
        if not math.isfinite(self.timeout) or not 0 < self.timeout <= 60:
            raise ModelError("invalid_model_timeout")
        if not 2 <= self.max_calls <= 32 or not 128 <= self.max_output_tokens <= 4096:
            raise ModelError("invalid_model_budget")
        if self.api_mode not in {"chat", "responses"}:
            raise ModelError("invalid_model_api_mode")
        if self.reasoning_effort not in {"default", "none", "low", "high", "max"}:
            raise ModelError("invalid_model_reasoning_effort")
        if self.mode == "offline":
            return
        if not self.model or not self.api_key or not self.base_url:
            raise ModelError("model_mode_requires_name_base_url_and_key")
        url = parse.urlsplit(self.base_url)
        if (url.scheme != "https" or not url.hostname or url.username or url.password
                or url.query or url.fragment or url.port not in (None, 443)):
            raise ModelError("model_endpoint_requires_public_https")
        try:
            ipaddress.ip_address(url.hostname)
        except ValueError:
            pass
        else:
            raise ModelError("model_endpoint_requires_hostname")
        if url.hostname in {"localhost", "localhost.localdomain"} or url.hostname.endswith(".local"):
            raise ModelError("model_endpoint_requires_public_hostname")

    def public_dict(self):
        return {"mode": self.mode, "model": self.model if self.mode == "model" else "",
                "base_url": self.base_url if self.mode == "model" else "",
                "api_mode": self.api_mode, "timeout": self.timeout,
                "max_calls": self.max_calls, "max_output_tokens": self.max_output_tokens,
                "reasoning_effort": self.reasoning_effort}


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ModelError("model_redirect_rejected")


def parse_object(text):
    if not isinstance(text, str) or len(text) > 16000:
        raise ModelError("invalid_model_text")
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ModelError("invalid_model_json")
        value = "\n".join(lines[1:-1])
    try:
        result = json.loads(value)
    except (ValueError, TypeError) as exc:
        raise ModelError("invalid_model_json") from None
    if not isinstance(result, dict):
        raise ModelError("model_output_must_be_object")
    return result


class JsonModelClient:
    """One HTTP attempt per call, without SDK dependencies or implicit retries."""
    def __init__(self, settings, opener=None):
        settings.validate()
        if settings.mode != "model":
            raise ModelError("model_client_disabled")
        self.settings = settings
        self.opener = opener or request.build_opener(_NoRedirect())

    def complete(self, system, context):
        config = self.settings
        prompt = json.dumps(context, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        deepseek = parse.urlsplit(config.base_url).hostname == "api.deepseek.com"
        if config.api_mode == "responses":
            suffix = "/responses"
            payload = {"model": config.model, "input": messages, "store": False,
                       "max_output_tokens": config.max_output_tokens,
                       "text": {"format": {"type": "json_object"}}}
            if config.reasoning_effort != "default":
                payload["reasoning"] = {"effort": config.reasoning_effort}
        else:
            suffix = "/chat/completions"
            payload = {"model": config.model, "messages": messages,
                       "max_completion_tokens": config.max_output_tokens,
                       "response_format": {"type": "json_object"}, "stream": False}
            if deepseek:
                # DeepSeek's published limit covers thinking and final output.
                payload["max_tokens"] = payload.pop("max_completion_tokens")
                if config.reasoning_effort != "default":
                    enabled = config.reasoning_effort != "none"
                    payload["thinking"] = {"type": "enabled" if enabled else "disabled"}
                    if enabled:
                        payload["reasoning_effort"] = config.reasoning_effort
            elif config.reasoning_effort != "default":
                payload["reasoning_effort"] = config.reasoning_effort
        req = request.Request(config.base_url.rstrip("/") + suffix,
                              data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                              headers={"Authorization": "Bearer " + config.api_key,
                                       "Content-Type": "application/json"}, method="POST")
        started = time.monotonic()
        try:
            with self.opener.open(req, timeout=config.timeout) as response:
                raw = response.read(262145)
            if len(raw) > 262144:
                raise ModelError("model_response_too_large")
            body = json.loads(raw)
            if config.api_mode == "responses":
                if body.get("status") != "completed":
                    raise ModelError("model_response_incomplete")
                parts = [item.get("text", "") for block in body.get("output", [])
                         if block.get("type") == "message" for item in block.get("content", [])
                         if item.get("type") == "output_text"]
                content = "\n".join(parts)
            else:
                choice = body["choices"][0]
                if choice.get("finish_reason") != "stop":
                    raise ModelError("model_response_incomplete")
                content = choice["message"]["content"]
            result = parse_object(content)
            usage = body.get("usage") or {}
            metrics = {key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens",
                       "input_tokens", "output_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens")
                       if type(usage.get(key)) is int and usage[key] >= 0}
            details = usage.get("completion_tokens_details") or {}
            if isinstance(details, dict) and type(details.get("reasoning_tokens")) is int and details["reasoning_tokens"] >= 0:
                metrics["reasoning_tokens"] = details["reasoning_tokens"]
            meta = {"elapsed_seconds": time.monotonic() - started, "usage": metrics}
            response_model = body.get("model")
            if isinstance(response_model, str) and len(response_model) <= 200:
                meta["response_model"] = response_model
            # Keep only the final JSON and aggregate usage, never reasoning text.
            return result, meta
        except ModelError:
            raise
        except error.HTTPError as exc:
            raise ModelError(http_error_code(exc)) from None
        except (TimeoutError, socket.timeout):
            raise ModelError("model_timeout") from None
        except (error.URLError, OSError):
            raise ModelError("model_transport_error") from None
        except (ValueError, TypeError, KeyError, IndexError, AttributeError):
            raise ModelError("model_response_invalid") from None
