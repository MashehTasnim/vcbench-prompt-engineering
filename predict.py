"""
VCBench Prediction Script
Runs 3 prompt variants (A: baseline, B: few-shot, C: chain-of-thought) on a
200-profile balanced sample using Claude claude-sonnet-4-20250514.
"""

import os
import csv
import json
import time
import random
import argparse
import anthropic

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH    = r"C:\Users\achra\Downloads\vcbench_starter\vcbench_final_public.csv"
OUT_DIR      = "."
SAMPLE_SIZE  = 200          # 100 pos + 100 neg
MAX_RETRIES  = 3
RETRY_DELAY  = 5            # seconds
MODEL        = "claude-sonnet-4-20250514"
random.seed(42)

# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_BASE = "You are an expert venture capital analyst with 20 years of experience evaluating early-stage founders."

PROMPT_A = """\
You are an expert venture capital analyst. Analyze this founder profile and predict whether their startup will achieve major success (IPO/acquisition above $500M or raise $500M+).

Founder Profile:
{profile}

Instructions:
- Analyze the founder's educational background, work experience, domain expertise, and prior startup outcomes
- Consider signals of entrepreneurial potential: elite institutions, relevant industry experience, prior successful exits, strong technical or domain expertise
- The base rate of success is approximately 9% — be selective and precise
- Respond with ONLY a JSON object: {{"prediction": "Yes" or "No", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}
"""

PROMPT_B = """\
You are an expert venture capital analyst. Analyze this founder profile and predict whether their startup will achieve major success (IPO/acquisition above $500M or raise $500M+).

Here are two examples of SUCCESSFUL founders:
---
Example 1 (Success):
Education: BS Computer Science (Institution QS rank 1), MBA (Institution QS rank 2)
Experience: Software Engineer 2 years at large tech company, CTO 3 years at Series B startup (acquired), CEO/founder current company
Outcome: YES — Prior successful exit, elite education, deep technical + business expertise.

Example 2 (Success):
Education: BS Electrical Engineering (Institution QS rank 3)
Experience: Research Scientist 4 years at major tech lab, founded 2 prior companies (1 acquired), domain expert in AI/ML
Outcome: YES — Serial entrepreneur with successful exit, rare technical depth in high-growth domain.
---

Here are two examples of UNSUCCESSFUL founders:
---
Example 3 (Failure):
Education: BA Business Administration (unranked institution)
Experience: Sales representative 2 years, Marketing manager 2 years, no prior founding experience
Outcome: NO — No technical depth, no prior startup experience, no domain differentiation.

Example 4 (Failure):
Education: BS Engineering (Institution QS rank 50)
Experience: Engineer 3 years at mid-size company, first-time founder, saturated market (e-commerce)
Outcome: NO — Undifferentiated background, competitive market, limited network signals.
---

Founder Profile to Evaluate:
{profile}

Instructions:
- The base rate of success is approximately 9% — be selective
- Respond with ONLY a JSON object: {{"prediction": "Yes" or "No", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}
"""

PROMPT_C = """\
You are an expert venture capital analyst. Analyze this founder profile step by step and predict whether their startup will achieve major success (IPO/acquisition above $500M or raise $500M+).

Founder Profile:
{profile}

Think through this step by step:
1. EDUCATION: What signals does the educational background provide? (institution prestige, field relevance, degree level)
2. EXPERIENCE: What signals does the work history provide? (seniority, industry relevance, company scale, prior founding)
3. OUTCOMES: Any evidence of prior exits, acquisitions, or notable achievements?
4. DOMAIN FIT: Does the founder's background align with the startup's industry?
5. RISK ASSESSMENT: What are the main risk factors? What are the main success signals?
6. FINAL VERDICT: Given a base rate of 9%, weigh the evidence and decide.

After your step-by-step reasoning, conclude with ONLY a JSON object on the last line:
{{"prediction": "Yes" or "No", "confidence": 0.0-1.0, "reasoning": "one-sentence summary"}}
"""

PROMPTS = {"A": PROMPT_A, "B": PROMPT_B, "C": PROMPT_C}

# ── Data loading ──────────────────────────────────────────────────────────────
def load_balanced_sample(path, n=200):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    pos = [r for r in rows if r["success"] == "1"]
    neg = [r for r in rows if r["success"] == "0"]
    half = n // 2
    sample = random.sample(pos, min(half, len(pos))) + random.sample(neg, min(half, len(neg)))
    random.shuffle(sample)
    return sample

# ── API call ──────────────────────────────────────────────────────────────────
def call_claude(client, prompt_text, variant):
    for attempt in range(MAX_RETRIES):
        try:
            msg = client.messages.create(
                model=MODEL,
                max_tokens=512 if variant != "C" else 1024,
                messages=[{"role": "user", "content": prompt_text}],
                system=SYSTEM_BASE,
            )
            return msg.content[0].text
        except Exception as e:
            print(f"  API error (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    return None

def parse_response(raw):
    if raw is None:
        return None, 0.5, "API error"
    # For chain-of-thought, JSON is on the last non-empty line
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    for line in reversed(lines):
        if line.startswith("{"):
            try:
                obj = json.loads(line)
                pred = obj.get("prediction", "No")
                conf = float(obj.get("confidence", 0.5))
                reas = obj.get("reasoning", "")
                return pred, conf, reas
            except Exception:
                pass
    # Fallback: try parsing the whole response
    try:
        obj = json.loads(raw)
        return obj.get("prediction","No"), float(obj.get("confidence",0.5)), obj.get("reasoning","")
    except Exception:
        return "No", 0.5, "parse_error"

# ── Main ──────────────────────────────────────────────────────────────────────
def run_variant(client, sample, variant_key, out_path):
    prompt_tpl = PROMPTS[variant_key]
    results = []
    for i, row in enumerate(sample):
        if (i + 1) % 50 == 0:
            print(f"  [{variant_key}] {i+1}/{len(sample)}")
        prompt_text = prompt_tpl.format(profile=row["anonymised_prose"])
        raw = call_claude(client, prompt_text, variant_key)
        pred, conf, reas = parse_response(raw)
        results.append({
            "founder_uuid":  row["founder_uuid"],
            "true_label":    int(row["success"]),
            "prediction":    pred,
            "confidence":    round(conf, 4),
            "reasoning":     reas[:300] if reas else "",
            "raw_response":  (raw or "")[:500],
        })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"  Saved: {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default="A,B,C", help="Comma-separated prompt variants to run")
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Set ANTHROPIC_API_KEY environment variable before running.")

    client = anthropic.Anthropic(api_key=api_key)
    sample = load_balanced_sample(DATA_PATH, args.sample_size)
    print(f"Loaded {len(sample)} profiles ({sum(1 for r in sample if r['success']=='1')} pos, "
          f"{sum(1 for r in sample if r['success']=='0')} neg)\n")

    variants = [v.strip() for v in args.variants.split(",")]
    for v in variants:
        print(f"Running Prompt {v}...")
        out = os.path.join(OUT_DIR, f"predictions_prompt_{v}.csv")
        run_variant(client, sample, v, out)
        print()

    print("Done. Run evaluate.py to see metrics.")


if __name__ == "__main__":
    main()
