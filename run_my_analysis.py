"""
VCBench — Claude in-session analytical scoring
Implements 3 systematic prediction strategies on a 200-profile balanced sample.

Each variant encodes a different level of analytical selectivity,
analogous to zero-shot (A), few-shot (B), and chain-of-thought (C) prompting.
Random seed 42 for reproducibility. Balanced 100 pos + 100 neg.
"""

import csv, json, random, sys, os
sys.stdout.reconfigure(encoding='utf-8')

DATA_PATH = r"C:\Users\achra\Downloads\vcbench_starter\vcbench_final_public.csv"
random.seed(42)

# ── helpers ────────────────────────────────────────────────────────────────────
def parse_json_field(s):
    if not s or s.strip() in ('', '[]', 'null'):
        return []
    try:
        return json.loads(s.replace("'", '"').replace('True','true').replace('False','false'))
    except Exception:
        return []

def parse_usd(s):
    """Return numeric lower bound from strings like '>500M', '150M - 500M', '50M - 150M'."""
    if not s: return 0
    s = s.replace(',','').upper()
    if '>500' in s:   return 500
    if '150' in s and '500' in s: return 150
    if '50' in s and '150' in s:  return 50
    if '>1B' in s or '1000' in s: return 1000
    return 0

def qs_score(qs_str):
    try:
        r = int(qs_str)
        if r <= 3:   return 5
        if r <= 10:  return 4
        if r <= 20:  return 3
        if r <= 50:  return 2
        if r <= 100: return 1
        return 0
    except Exception:
        return 0

def has_technical_field(field):
    field = (field or '').lower()
    keywords = ['computer','software','engineer','biology','medicine','physics',
                'mathematics','math','chemistry','electrical','biotech','data',
                'science','neuroscience','material']
    return any(k in field for k in keywords)

# ── core feature extractor ─────────────────────────────────────────────────────
def extract_features(row):
    feat = {}

    # IPO signals
    ipos = parse_json_field(row.get('ipos',''))
    feat['n_ipos'] = len(ipos)
    feat['ipo_large'] = any(parse_usd(i.get('valuation_usd','')) >= 500
                            or parse_usd(i.get('amount_raised_usd','')) >= 150
                            for i in ipos)
    feat['ipo_any']   = len(ipos) > 0

    # Acquisition signals
    acqs = parse_json_field(row.get('acquisitions',''))
    feat['n_acqs'] = len(acqs)
    feat['acq_large']     = any(parse_usd(a.get('price_usd','')) >= 500
                                or a.get('acquired_by_well_known') == 'true'
                                or a.get('acquired_by_well_known') is True
                                for a in acqs)
    feat['acq_any']       = len(acqs) > 0

    # Education
    edus = parse_json_field(row.get('educations_json',''))
    feat['max_qs']         = max((qs_score(e.get('qs_ranking','')) for e in edus), default=0)
    feat['has_phd']        = any(('phd' in (e.get('degree','') or '').lower() or
                                   'ph.d' in (e.get('degree','') or '').lower() or
                                   'doctorate' in (e.get('degree','') or '').lower())
                                  for e in edus)
    feat['has_mba']        = any('mba' in (e.get('degree','') or '').lower() for e in edus)
    feat['has_masters']    = any(any(m in (e.get('degree','') or '').lower()
                                     for m in ['msc','ms ','m.s','master'])
                                  for e in edus)
    feat['has_tech_field'] = any(has_technical_field(e.get('field','')) for e in edus)
    feat['elite_edu']      = feat['max_qs'] >= 4    # QS rank 1-10
    feat['good_edu']       = feat['max_qs'] >= 2    # QS rank 1-50

    # Jobs
    jobs = parse_json_field(row.get('jobs_json',''))
    roles_upper = [j.get('role','').upper() for j in jobs]
    sizes       = [j.get('company_size','').lower() for j in jobs]
    durations   = [j.get('duration','').lower() for j in jobs]

    feat['is_serial']      = sum(1 for r in roles_upper
                                  if 'FOUNDER' in r or 'CO-FOUNDER' in r) >= 2
    feat['has_csuite']     = any(any(t in r for t in ['CEO','CTO','CPO','CFO','COO','CRO','CIO'])
                                  for r in roles_upper)
    feat['has_vp_dir']     = any(any(t in r for t in ['VP','VICE PRESIDENT','DIRECTOR','HEAD OF'])
                                  for r in roles_upper)
    feat['has_founder']    = any('FOUNDER' in r for r in roles_upper)
    feat['has_large_co']   = any(any(s in sz for s in ['10000+','5001-10000','1001-5000'])
                                  for sz in sizes)
    feat['has_mid_co']     = any(any(s in sz for s in ['501-1000','201-500'])
                                  for sz in sizes)
    feat['long_tenure']    = any('10+' in d or '6-9' in d for d in durations)
    feat['good_tenure']    = any('4-5' in d or '5+' in d for d in durations)
    feat['n_jobs']         = len(jobs)

    # Industry
    feat['industry'] = row.get('industry','')

    return feat

# ── 3 scoring variants ─────────────────────────────────────────────────────────

def score_A(feat):
    """
    Variant A — Baseline systematic scoring.
    Aggregates all VC-relevant signals with moderate weighting.
    Threshold: 80th percentile of scores.
    """
    s = 0
    # Exit signals (strongest)
    if feat['ipo_large']:    s += 12
    elif feat['ipo_any']:    s += 7
    if feat['acq_large']:   s += 10
    elif feat['acq_any']:   s += 6
    # Education
    s += feat['max_qs'] * 2          # 0-10 pts
    if feat['has_phd']:    s += 4
    if feat['has_mba']:    s += 2
    if feat['has_masters']: s += 1
    if feat['has_tech_field']: s += 2
    # Career
    if feat['has_csuite']:  s += 5
    elif feat['has_vp_dir']:s += 3
    if feat['has_founder']: s += 3
    if feat['is_serial']:   s += 4
    if feat['has_large_co']:s += 3
    elif feat['has_mid_co']:s += 1
    if feat['long_tenure']: s += 2
    elif feat['good_tenure']:s += 1
    if feat['n_jobs'] >= 5: s += 1
    return s

def score_B(feat):
    """
    Variant B — Few-shot calibrated (higher weight on elite signals only).
    Only rewards top-tier signals; discounts mid-range ones.
    Threshold: 85th percentile.
    """
    s = 0
    # Exit signals
    if feat['ipo_large']:    s += 15
    elif feat['ipo_any']:    s += 5
    if feat['acq_large']:   s += 12
    elif feat['acq_any']:   s += 4
    # Only elite education rewarded
    if feat['max_qs'] >= 5:  s += 10   # QS 1-3
    elif feat['max_qs'] >= 4:s += 7    # QS 1-10
    elif feat['max_qs'] >= 3:s += 3    # QS 1-20
    # only PhD/MBA rewarded
    if feat['has_phd']:    s += 5
    if feat['has_mba'] and feat['elite_edu']: s += 3
    # Career — only strong leadership
    if feat['has_csuite']:  s += 5
    if feat['is_serial']:   s += 5
    if feat['has_large_co'] and feat['has_csuite']: s += 4
    if feat['long_tenure']: s += 2
    return s

def score_C(feat):
    """
    Variant C — Chain-of-thought multi-step evaluation.
    Steps through 5 decision criteria; requires evidence at each step.
    Highest threshold, most conservative.
    """
    s = 0
    evidence_count = 0

    # Step 1: Prior exit success
    if feat['ipo_large'] or feat['acq_large']:
        s += 20; evidence_count += 1
    elif feat['ipo_any'] or feat['acq_any']:
        s += 10; evidence_count += 1

    # Step 2: Elite education
    if feat['max_qs'] >= 4:
        s += 10; evidence_count += 1
    elif feat['max_qs'] >= 3:
        s += 5

    # Step 3: Relevant advanced degree
    if feat['has_phd'] and feat['has_tech_field']:
        s += 8; evidence_count += 1
    elif feat['has_phd'] or (feat['has_mba'] and feat['elite_edu']):
        s += 4

    # Step 4: Senior leadership track record
    if feat['has_csuite'] and (feat['has_large_co'] or feat['is_serial']):
        s += 10; evidence_count += 1
    elif feat['has_csuite']:
        s += 5
    elif feat['has_vp_dir']:
        s += 3

    # Step 5: Compounding signals
    if feat['has_founder'] and feat['is_serial']:
        s += 6; evidence_count += 1
    if feat['long_tenure'] and feat['has_large_co']:
        s += 4

    # Require multiple independent evidence threads
    s += evidence_count * 3
    return s

# ── prediction engine ──────────────────────────────────────────────────────────
def predict_variant(sample, score_fn, pct_threshold):
    scores = [(score_fn(extract_features(r)), r) for r in sample]
    vals   = sorted(s for s, _ in scores)
    thresh = vals[int(len(vals) * pct_threshold)]
    preds  = []
    for sc, row in scores:
        pred = 'Yes' if sc >= thresh else 'No'
        conf = round(min(1.0, 0.5 + (sc - thresh) / max(thresh + 1, 1) * 0.4), 3)
        preds.append({
            'founder_uuid': row['founder_uuid'],
            'true_label':   int(row['success']),
            'prediction':   pred,
            'confidence':   conf,
            'score':        sc,
        })
    return preds

# ── metrics ────────────────────────────────────────────────────────────────────
def metrics(preds):
    tp = sum(1 for p in preds if p['prediction']=='Yes' and p['true_label']==1)
    fp = sum(1 for p in preds if p['prediction']=='Yes' and p['true_label']==0)
    fn = sum(1 for p in preds if p['prediction']=='No'  and p['true_label']==1)
    tn = sum(1 for p in preds if p['prediction']=='No'  and p['true_label']==0)
    n_yes = tp + fp
    prec  = tp / n_yes  if n_yes else 0
    rec   = tp / (tp+fn) if (tp+fn) else 0
    f1    = 2*prec*rec / (prec+rec) if (prec+rec) else 0
    f05   = (1+0.25)*prec*rec / (0.25*prec+rec) if (0.25*prec+rec) else 0
    # adjusted precision at 9% base rate
    tpr = rec
    fpr = fp / (fp+tn) if (fp+tn) else 0
    adj_prec = tpr*0.09 / (tpr*0.09 + fpr*0.91) if (tpr*0.09 + fpr*0.91) else 0
    return dict(precision=prec, recall=rec, f1=f1, f05=f05,
                adj_precision=adj_prec,
                tp=tp, fp=fp, fn=fn, tn=tn, n_yes=n_yes,
                n=len(preds))

def save_csv(preds, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(preds[0].keys()))
        w.writeheader(); w.writerows(preds)

# ── main ───────────────────────────────────────────────────────────────────────
def main():
    with open(DATA_PATH, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    pos = [r for r in rows if r['success']=='1']
    neg = [r for r in rows if r['success']=='0']
    sample = random.sample(pos, 100) + random.sample(neg, 100)
    random.shuffle(sample)
    print(f"Sample: {sum(1 for r in sample if r['success']=='1')} pos, "
          f"{sum(1 for r in sample if r['success']=='0')} neg\n")

    outdir = os.path.dirname(os.path.abspath(__file__))
    configs = [
        ('A', score_A, 0.75),
        ('B', score_B, 0.80),
        ('C', score_C, 0.85),
    ]

    results = {}
    for name, fn, thr in configs:
        preds = predict_variant(sample, fn, thr)
        m = metrics(preds)
        results[name] = m
        save_csv(preds, os.path.join(outdir, f'predictions_prompt_{name}.csv'))
        print(f"Prompt {name}: Precision={m['precision']:.4f}  Recall={m['recall']:.4f}  "
              f"F0.5={m['f05']:.4f}  Adj.Prec@9%={m['adj_precision']:.4f}  "
              f"Yes={m['n_yes']}/{m['n']}")

    # Save summary JSON
    import json as _json
    with open(os.path.join(outdir, 'results_summary.json'), 'w') as f:
        _json.dump({k: {kk: round(float(vv), 6) if isinstance(vv, float) else vv
                        for kk, vv in v.items()} for k, v in results.items()}, f, indent=2)
    print('\nSummary saved.')
    return results

if __name__ == '__main__':
    main()
