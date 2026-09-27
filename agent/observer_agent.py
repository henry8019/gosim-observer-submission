"""Unified observer entry for practice rehearsal and the formal v2 protocol."""
from __future__ import annotations

import json
from pathlib import Path
import sys

# Keep `python -I` usable in the agent-only deployment image.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from observer_controller import ObserverController
from observer_model import JsonModelClient, ModelError, ObserverSettings


def run(stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr, settings=None, model=None):
    settings = settings or ObserverSettings.from_environment()
    settings.validate()
    if model is None and settings.mode == "model":
        model = JsonModelClient(settings)

    def emit(kind, record):
        print("observer-"+kind+" "+json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
              file=stderr, flush=True)

    emit("start", {"entry": "observer_agent.py", "settings": settings.public_dict(),
                   "offline_is_model_evidence": False})
    agent = None
    version = None
    snapshot_schema = None
    try:
        for line in stdin:
            if not line.strip():
                continue
            message = json.loads(line)
            current = message.get("protocol_version")
            if current not in {"participant-agent-protocol-v1", "participant-agent-protocol-v2"}:
                raise ValueError("Unsupported participant protocol")
            payload = message.get("payload")
            if not isinstance(payload, dict):
                raise ValueError("Protocol payload must be an object")
            if message.get("message_type") == "initialize":
                if agent is not None or payload.get("schema_version") != "initial-publication-v2":
                    raise ValueError("Invalid or repeated initialization")
                version = current
                agent = ObserverController(payload, settings, model=model, emit=emit)
                continue
            if agent is None or current != version or message.get("message_type") != "decision_request":
                raise ValueError("Expected a decision request after initialization")
            schema = payload.get("schema_version")
            if schema not in {"decision-snapshot-v2", "decision-snapshot-v3"}:
                raise ValueError("Unsupported snapshot schema")
            if schema == "decision-snapshot-v3" and version != "participant-agent-protocol-v2":
                raise ValueError("Formal snapshots require participant protocol v2")
            if snapshot_schema is not None and schema != snapshot_schema:
                raise ValueError("Snapshot mechanics cannot change during a run")
            snapshot_schema = schema
            if type(message.get("decision_sequence")) is not int or message["decision_sequence"] != payload.get("decision_sequence"):
                raise ValueError("Decision sequence mismatch")
            decision = agent.decide(payload)
            response = {"protocol_version": version, "message_type": "decision_response",
                        "decision_sequence": message["decision_sequence"], **decision}
            print(json.dumps(response, ensure_ascii=False, separators=(",", ":")), file=stdout, flush=True)
    finally:
        if agent is not None:
            emit("summary", agent.summary())


if __name__ == "__main__":
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        run()
    except ModelError as exc:
        print("observer-configuration-error "+str(exc), file=sys.stderr, flush=True)
        raise SystemExit(2)
