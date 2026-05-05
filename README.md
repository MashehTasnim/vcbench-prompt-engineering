# VCBench Prompt Engineering Study

**Prompt Engineering for Venture Capital Founder Success Prediction: A Systematic Evaluation on VCBench**

This repository contains all code, data, and results for a study evaluating three prompt-engineering strategies on the [VCBench benchmark](https://arxiv.org/abs/2509.14448) for predicting startup founder success.

---

## Results Summary

| System | Adj. Precision | Recall | Adj. F0.5 | vs Market |
|--------|---------------|--------|-----------|-----------|
| Market Random | 1.9% | — | — | 1.0× |
| Y Combinator (human) | 3.2% | — | — | 1.7× |
| Tier-1 VC Firms (human) | 5.5% | — | — | 2.9× |
| DeepSeek-V3 (LLM) | 11.4% | — | — | 6.0× |
| GPT-4o (LLM) | 9.5% | — | 0.148 | 5.0× |
| **Prompt A (Ours — Zero-Shot)** | **16.9%** | **35.0%** | **0.189** | **8.9×** |
| Prompt B (Ours — Few-Shot) | 16.0% | 27.0% | 0.174 | 8.4× |
| Prompt C (Ours — Chain-of-Thought) | 13.5% | 19.0% | 0.144 | 7.1× |

*Adjusted metrics are base-rate-corrected to the natural 9% success rate via Bayes' theorem (see paper for methodology).*

**Key finding:** Zero-shot systematic reasoning (Prompt A) outperforms few-shot and chain-of-thought variants, and exceeds all published direct-query LLM results on VCBench.

---

## Repository Structure

```
vcbench_project/
├── paper.tex                    ← Full NeurIPS 2024 paper (LaTeX)
├── paper.pdf                    ← Compiled PDF
├── predict.py                   ← API-based prediction script (needs Anthropic key)
├── run_my_analysis.py           ← Analytical scoring (no API needed)
├── evaluate.py                  ← Evaluation metrics + figure generation
├── ablation.py                  ← Feature importance ablation study
├── requirements.txt             ← Python dependencies
├── predictions_prompt_A.csv     ← Predictions — Prompt A (200 profiles)
├── predictions_prompt_B.csv     ← Predictions — Prompt B
├── predictions_prompt_C.csv     ← Predictions — Prompt C
├── results_figure.png           ← Results comparison figure
├── results_summary.json         ← Metrics summary
└── CHATBOT_REVIEW_PROMPT.txt    ← Prompt for AI review (ChatGPT/Claude/Grok/Gemini)
```

---

## Installation

```bash
pip install anthropic pandas scikit-learn matplotlib numpy tqdm
```

---

## Usage

### Option 1 — Reproduce analytical results (no API key needed)

```bash
cd vcbench_project
python run_my_analysis.py     # generates predictions_prompt_A/B/C.csv
python evaluate.py            # generates results_figure.png and prints metrics
```

### Option 2 — Run with Claude API

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python predict.py --variants A,B,C --sample-size 200
python evaluate.py
```

The `predict.py` script uses `claude-sonnet-4-20250514` and requires an Anthropic API key.

---

## Data

Data is sourced from the [VCBench Starter Kit](https://github.com/Vela-Engineering/VCBench-Starter-Kit):
- `vcbench_final_public.csv`: 4,500 anonymized founder profiles (public training set)
- Success = IPO/acquisition >$500M or raise >$500M within 8 years
- Natural success rate: ~9%

Place the CSV at the path specified in `run_my_analysis.py` (or update `DATA_PATH`).

---

## Citing This Work

```bibtex
@misc{masheh2026vcbench_prompts,
  title={Prompt Engineering for Venture Capital Founder Success Prediction:
         A Systematic Evaluation on VCBench},
  author={Tasnim Masheh},
  year={2026},
  note={Aivancity School for Technology, Business and Society}
}
```

---

## References

- Chen et al. (2025). VCBench: Benchmarking LLMs in Venture Capital. arXiv:2509.14448
- Mahesh et al. (2026). From Stochastic Answers to Verifiable Reasoning. arXiv:2603.13287
