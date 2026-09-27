"""Explicit opt-in practice entry; does not change the formal v2 entry point."""
import json
import os
import sys

from practice_long_horizon import LongHorizonPlanner
from scoring_preview import preview_actions


def run(stdin=sys.stdin, stdout=sys.stdout):
    planner = None
    version = None
    for line in stdin:
        if not line.strip():
            continue
        message = json.loads(line)
        current = message.get('protocol_version')
        if current not in ('participant-agent-protocol-v1', 'participant-agent-protocol-v2'):
            raise ValueError('Unsupported practice protocol')
        payload = message['payload']
        if message['message_type'] == 'initialize':
            if payload['schema_version'] != 'initial-publication-v2':
                raise ValueError('Unsupported initialization')
            version = current
            if os.getenv('SAC_PRACTICE_POLICY') == 'first_credit':
                from practice_stopping import FirstCreditPlanner
                planner = FirstCreditPlanner(payload)
            else:
                planner = LongHorizonPlanner(payload)
            continue
        if planner is None or current != version or message['message_type'] != 'decision_request':
            raise ValueError('Expected a decision request after initialization')
        if message['decision_sequence'] != payload['decision_sequence']:
            raise ValueError('Sequence mismatch')
        planner.update(payload)
        repeat = payload['schema_version'] == 'decision-snapshot-v3'
        previews = preview_actions(payload, planner.contract, planner.bests if repeat else None)
        decision = planner.choose(payload, previews, {'action': 'wait'})
        response = {'protocol_version': version, 'message_type': 'decision_response',
                    'decision_sequence': message['decision_sequence'], **decision}
        print(json.dumps(response, ensure_ascii=False, separators=(',', ':')), file=stdout, flush=True)


if __name__ == '__main__':
    run()
