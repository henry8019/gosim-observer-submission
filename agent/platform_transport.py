"""Forward unchanged provider requests through the platform's per-run proxy."""
from urllib import parse, request

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
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.provider_endpoint = provider_base_url.rstrip("/") + "/chat/completions"
        self.credential = credential
        self.opener = opener or request.build_opener(_NoRedirect())

    def open(self, original, timeout):
        if original.full_url != self.provider_endpoint or original.get_method() != "POST":
            raise ModelError("unexpected_platform_proxy_request")
        forwarded = request.Request(
            self.endpoint, data=original.data, method="POST",
            headers={"Authorization": "Bearer " + self.credential,
                     "Content-Type": "application/json"},
        )
        return self.opener.open(forwarded, timeout=timeout)
