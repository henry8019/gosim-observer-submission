"""Forward unchanged provider requests through the platform's per-run proxy."""
import io
import json
import sys
import time
from urllib import error, parse, request

from observer_model import ModelError, _NoRedirect


class PlatformRelay:
    def __init__(self, base_url, credential, provider_base_url, opener=None):
        endpoint = parse.urlsplit(base_url)
        # The evaluator may inject an HTTP endpoint inside its isolated network.
        # This address comes only from process environment, never from snapshots
        # or model output. The provider's permanent credential is not used here.
        if (endpoint.scheme not in {"http", "https"} or not endpoint.hostname
                or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment
                or any(ord(c) < 33 for c in base_url) or not credential):
            raise ModelError("invalid_platform_proxy_configuration")
        base = base_url.rstrip('/')
        self.endpoint = base if endpoint.path.rstrip('/').endswith('/chat/completions') else base+'/chat/completions'
        # Some OpenAI-compatible gateways publish a root base but mount v1;
        # others include v1 in the base. Only try the corresponding standard
        # route on the SAME injected origin after an explicit generic rejection.
        if endpoint.path.rstrip('/').endswith('/chat/completions'):
            self.alternate = None
        elif endpoint.path.rstrip('/').endswith('/v1'):
            self.alternate = base[:-3]+'/chat/completions'
        else:
            self.alternate = base+'/v1/chat/completions'
        self.provider_endpoint = provider_base_url.rstrip("/") + "/chat/completions"
        self.credential = credential
        self.opener = opener or request.build_opener(_NoRedirect())

    def open(self, original, timeout):
        if original.full_url != self.provider_endpoint or original.get_method() != "POST":
            raise ModelError("unexpected_platform_proxy_request")
        def send(endpoint, remaining):
            forwarded = request.Request(endpoint, data=original.data, method='POST',
                headers={'Authorization': 'Bearer '+self.credential, 'Content-Type': 'application/json'})
            return self.opener.open(forwarded, timeout=remaining)
        started = time.monotonic()
        try:
            return send(self.endpoint, timeout)
        except error.HTTPError as exc:
            if exc.code != 400 or self.alternate is None:
                raise
            raw = exc.read(16385)
            try:
                body = json.loads(raw) if len(raw) <= 16384 else None
                detail = body.get('error', body) if isinstance(body, dict) else None
                label = detail.get('message', detail.get('code', '')) if isinstance(detail, dict) else detail
                unsupported = isinstance(label, str) and label.strip().lower() == 'unsupported'
            except (ValueError, TypeError):
                unsupported = False
            restored = error.HTTPError(exc.url, exc.code, exc.reason, exc.headers, io.BytesIO(raw))
            remaining = timeout-(time.monotonic()-started)
            if not unsupported or remaining <= 0:
                raise restored from None
            print('observer-transport-route standard_v1_compatibility_attempt', file=sys.stderr, flush=True)
            response = send(self.alternate, remaining)
            self.endpoint, self.alternate = self.alternate, None
            print('observer-transport-route standard_v1_compatibility_accepted', file=sys.stderr, flush=True)
            return response
