"""
Ablation study: measures adjusted precision drop when each feature category is removed from Prompt A.
"""
import csv, json, random, sys
sys.stdout.reconfigure(encoding='utf-8')

DATA_PATH = r"C:\Users\achra\Downloads\vcbench_starter\vcbench_final_public.csv"
random.seed(42)

def parse_json_field(s):
    if not s or s.strip() in ('', '[]', 'null'): return []
    try:
        return json.loads(s.replace("'", '"').replace('True','true').replace('False','false'))
    except: return []

def parse_usd(s):
    if not s: return 0
    s = s.replace(',','').upper()
    if '>500' in s: return 500
    if '150' in s and '500' in s: return 150
    if '50' in s and '150' in s: return 50
    if '>1B' in s or '1000' in s: return 1000
    return 0

def qs_score(qs_str):
    try:
        r = int(qs_str)
        if r<=3: return 5
        if r<=10: return 4
        if r<=20: return 3
        if r<=50: return 2
        if r<=100: return 1
        return 0
    except: return 0

def has_tech(f): return any(k in (f or '').lower() for k in ['computer','software','engineer','biology','medicine','physics','mathematics','math','chemistry','electrical','biotech','data','science'])

def score_A(row, ablate=None):
    ablate = ablate or set()
    s = 0
    ipos = parse_json_field(row.get('ipos',''))
    acqs = parse_json_field(row.get('acquisitions',''))
    if 'exits' not in ablate:
        if any(parse_usd(i.get('valuation_usd',''))>=500 or parse_usd(i.get('amount_raised_usd',''))>=150 for i in ipos): s+=12
        elif ipos: s+=7
        if any(parse_usd(a.get('price_usd',''))>=500 for a in acqs): s+=10
        elif acqs: s+=6
    edus = parse_json_field(row.get('educations_json',''))
    if 'education' not in ablate:
        best_qs = max((qs_score(e.get('qs_ranking','')) for e in edus), default=0)
        s += best_qs * 2
        if any('phd' in (e.get('degree','') or '').lower() or 'ph.d' in (e.get('degree','') or '').lower() for e in edus): s+=4
        if any('mba' in (e.get('degree','') or '').lower() for e in edus): s+=2
        if any(has_tech(e.get('field','')) for e in edus): s+=2
    jobs = parse_json_field(row.get('jobs_json',''))
    roles = [j.get('role','').upper() for j in jobs]
    sizes = [j.get('company_size','').lower() for j in jobs]
    durs  = [j.get('duration','').lower() for j in jobs]
    if 'leadership' not in ablate:
        if any(any(t in r for t in ['CEO','CTO','CPO','CFO','COO']) for r in roles): s+=5
        elif any(any(t in r for t in ['VP','DIRECTOR','HEAD OF']) for r in roles): s+=3
        if any('FOUNDER' in r for r in roles): s+=3
    if 'serial' not in ablate:
        if sum(1 for r in roles if 'FOUNDER' in r) >= 2: s+=4
    if 'scale' not in ablate:
        if any(any(t in sz for t in ['10000+','5001-10000','1001-5000']) for sz in sizes): s+=3
        elif any(any(t in sz for t in ['501-1000','201-500']) for sz in sizes): s+=1
    if 'tenure' not in ablate:
        if any('10+' in d or '6-9' in d for d in durs): s+=2
        elif any('4-5' in d or '5+' in d for d in durs): s+=1
    return s

def run_ablation(sample, ablate=None):
    scores = [(score_A(r, ablate), r) for r in sample]
    vals = sorted(s for s,_ in scores)
    thresh = vals[int(len(vals)*0.75)]
    tp=fp=fn=tn=0
    for sc, row in scores:
        pred = sc >= thresh
        true = int(row['success'])
        if pred and true: tp+=1
        elif pred and not true: fp+=1
        elif not pred and true: fn+=1
        else: tn+=1
    tpr = tp/(tp+fn) if (tp+fn) else 0
    fpr = fp/(fp+tn) if (fp+tn) else 0
    adj = tpr*0.09/(tpr*0.09+fpr*0.91) if (tpr*0.09+fpr*0.91) else 0
    return adj

with open(DATA_PATH, encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
pos = [r for r in rows if r['success']=='1']
neg = [r for r in rows if r['success']=='0']
sample = random.sample(pos,100) + random.sample(neg,100)

base = run_ablation(sample)
print(f"Full model (Prompt A): adj_precision={base:.4f}")
print()

ablations = [
    ('exits',      'Prior exit signals (IPO / Acquisition)'),
    ('education',  'Education quality (QS ranking)'),
    ('leadership', 'Leadership seniority (C-suite, VP)'),
    ('scale',      'Company scale (>=1,000 employees)'),
    ('tenure',     'Tenure depth (>=6 years)'),
]
for ablate_key, label in ablations:
    ablated = run_ablation(sample, {ablate_key})
    drop = base - ablated
    print(f"  Without {label}: adj_prec={ablated:.4f}  drop={drop:+.4f}")
