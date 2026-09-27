"""Packaged as run_agent.py beside settings.json and agent/."""
import json
import os
from pathlib import Path
import sys


def main():
    base = Path(__file__).resolve().parent
    proxy_base = os.environ.get("OPENAI_BASE_URL", "").strip()
    proxy_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if bool(proxy_base) != bool(proxy_key):
        raise SystemExit("Platform model proxy configuration is incomplete.")
    credential = proxy_key if proxy_base else os.environ.get("MODEL_API_KEY", "").strip()
    if not credential:
        raise SystemExit("Configure the platform model API or set MODEL_API_KEY for a local run.")
    parameters = json.loads((base / "settings.json").read_text(encoding="utf-8"))
    for name in list(os.environ):
        if name.startswith(("SAC_", "OBSERVER_", "MODEL_")):
            del os.environ[name]
    os.environ.update(parameters)
    os.environ["MODEL_API_KEY"] = credential
    sys.path.insert(0, str(base / "agent"))
    from observer_agent import run
    from observer_model import JsonModelClient, ModelError, ObserverSettings
    from platform_transport import PlatformRelay

    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        settings = ObserverSettings.from_environment()
        model = None
        if proxy_base:
            if settings.api_mode != "chat":
                raise ModelError("platform_proxy_requires_chat_completions")
            relay = PlatformRelay(proxy_base, proxy_key, settings.base_url)
            model = JsonModelClient(settings, opener=relay)
            print("observer-transport platform-model-proxy", file=sys.stderr, flush=True)
        run(settings=settings, model=model)
    except ModelError as exc:
        print("observer-configuration-error " + str(exc), file=sys.stderr, flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
