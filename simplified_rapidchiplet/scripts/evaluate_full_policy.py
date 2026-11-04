"""Prediction-only held-out control-arm evaluation; no PPA or policy updates."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_local_rl_diagnostics import read, write
from analyze_replay_mapping import ranking_metrics
from simple_rapidchiplet.hybrid.multi_step_q import MultiStepQ
from simple_rapidchiplet.hybrid.cost_value_q import CostValueQ
from simple_rapidchiplet.hybrid.full_space import DesignTrial


def evaluate(out):
    out = Path(out).resolve()
    def events(folder):
        path = folder / 'events.jsonl'
        return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    trained = {r['state_hash'] for r in events(out / 'rl') if r.get('feasible') and r.get('new_full_ppa')}
    trained.update(DesignTrial.from_dict(r['trial']).fingerprint
                   for r in read(out / 'screening.json', dict(rows=[]))['rows'] if r.get('cost') is not None)
    trained.update(r['trial']['state']['architecture_candidate_id'] + ':' + json.dumps(r['trial']['state']['mapping'], sort_keys=True)
                   for r in read(out / 'screening.json', dict(rows=[]))['rows'] if r.get('cost') is not None)
    controls, seen = [], set()
    for method in ('sa', 'random'):
        for row in events(out / method):
            signature = row['action']['state']['architecture_candidate_id'] + ':' + json.dumps(row['action']['state']['mapping'], sort_keys=True)
            if row.get('feasible') and row.get('reward') is not None and row['state_hash'] not in trained and signature not in trained and row['state_hash'] not in seen:
                controls.append(row)
                seen.add(row['state_hash'])
    result = dict(held_out_count=len(controls), new_ppa=0, policy_updates=0,
        scope='control-arm immediate physical rewards; candidate hashes trained by RL/shared screen excluded; auxiliary reward ranking does not prove future-Q or search superiority')
    checkpoint = read(out / 'rl/checkpoint.json', {})
    if len(controls) >= 3 and checkpoint.get('policy_file'):
        x = np.asarray([r['features'] for r in controls], dtype=np.float32)
        y = np.asarray([r['reward'] for r in controls])
        for label, name in (('initial', 'policy_initial.pt'), ('after_shared', 'policy_after_shared.pt'), ('final', checkpoint['policy_file'])):
            path = out / 'rl' / name
            if not path.exists():
                continue
            header = torch.load(path, map_location='cpu', weights_only=False)
            factored = header.get('representation') == CostValueQ.representation
            model = CostValueQ(x.shape[1], header['after_indices']) if factored else MultiStepQ(x.shape[1])
            model.load(path)
            prediction = model.predict_reward(x, np.asarray([r['before_cost'] for r in controls])) if factored else model.predict_reward(x)
            result[label] = ranking_metrics(prediction, y)
    write(out / 'analysis/held_out_policy.json', result)
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    evaluate(p.parse_args().out)
