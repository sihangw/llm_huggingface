#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json
from evaluator import Evaluator

def run(gold_path, pred_path, out_csv):
    ev = Evaluator(enable_behavioral=False)
    import csv
    rows = []
    n = 0; total = 0.0
    with open(gold_path, "r", encoding="utf-8") as fg, open(pred_path, "r", encoding="utf-8") as fp:
        for line_g, line_p in zip(fg, fp):
            g = json.loads(line_g)
            p = json.loads(line_p)
            score, det = ev.score_example(g['mission'], p, g)
            n += 1; total += score
            rows.append({'idx': n-1, 'score': score, 'S_obj': det.get('S_obj'), 'S_con': det.get('S_con'),
                         'S_area': det.get('S_area'), 'S_surf': det.get('S_surf'), 'validity': det.get('validity'),
                         'reasons': ','.join(det.get('reasons', [])) if 'reasons' in det else ''})
    import csv
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Mean score: {total/max(1,n):.4f} over {n} examples. Results written to {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate predictions JSONL against gold JSONL.")
    ap.add_argument("--gold", default="data/test_missions_200.jsonl", help="Path to gold JSONL (with fields: mission, objective_function, constraints, areas)")
    ap.add_argument("--pred", default="data/test_missions_pred.jsonl", help="Path to predictions JSONL (fields: objective_function, constraints, areas) aligned with gold order")
    ap.add_argument("--out", default="data/test_results.csv", help="Path to write CSV of per-example scores")
    a = ap.parse_args()
    run(a.gold, a.pred, a.out)
