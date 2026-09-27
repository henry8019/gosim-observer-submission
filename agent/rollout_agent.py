"""Explicit new practice candidate; same entry and settings for every duration."""
import json
from pathlib import Path
import sys

from model_factory import ModelSettings
from rollout_planner import RolloutPlanner
from scoring_preview import preview_actions


def run(stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr):
    settings = ModelSettings.from_environment(Path(__file__).resolve().parent / '.env')
    if not settings.deterministic:
        raise ValueError('Research entry only supports MODEL_PROVIDER=none')
    planner = None
    version = None
    try:
        for line in stdin:
            if not line.strip():
                continue
            message = json.loads(line)
            current = message.get('protocol_version')
            if current not in ('participant-agent-protocol-v1', 'participant-agent-protocol-v2'):
                raise ValueError('Unsupported practice protocol')
            payload = message['payload']
            if message['message_type'] == 'initialize':
                if planner is not None or payload['schema_version'] != 'initial-publication-v2':
                    raise ValueError('Invalid initialization')
                planner, version = RolloutPlanner(payload), current
                print('rollout-start '+json.dumps({'variant':planner.variant, 'entry':'rollout_agent.py'}), file=stderr, flush=True)
                continue
            if planner is None or current != version or message['message_type'] != 'decision_request':
                raise ValueError('Expected a decision request')
            if message['decision_sequence'] != payload['decision_sequence']:
                raise ValueError('Sequence mismatch')
            planner.update(payload)
            previews = preview_actions(payload, planner.contract)
            decision = planner.choose(payload, previews, {'action':'wait'})
            # Predictions are audit estimates, never credited scores. Keep the
            # received scorer feedback as a separate, unmodified log record.
            print('rollout-prediction '+json.dumps({'decision_sequence':message['decision_sequence'],
                  **planner.last_prediction}, separators=(',', ':')), file=stderr, flush=True)
            if payload.get('tile_last_finished') is not None:
                print('rollout-feedback '+json.dumps({'decision_sequence':message['decision_sequence'],
                      'received_tile_last_finished':payload['tile_last_finished']}, separators=(',', ':')),
                      file=stderr, flush=True)
            if planner.diagnostics['calls'] == 1 or planner.diagnostics['calls'] % 25 == 0:
                print('rollout-progress '+json.dumps({'variant':planner.variant, **planner.diagnostics}, sort_keys=True), file=stderr, flush=True)
            print(json.dumps({'protocol_version':version, 'message_type':'decision_response',
                              'decision_sequence':message['decision_sequence'], **decision}, separators=(',', ':')),
                  file=stdout, flush=True)
    finally:
        if planner:
            print('rollout-summary '+json.dumps({'variant':planner.variant, **planner.diagnostics}, sort_keys=True), file=stderr, flush=True)


if __name__ == '__main__':
    run()
