# Phase 0.5 Pilot Harness (Document B §4–§6)

Four gated pilots. Mock-validated end-to-end (deterministic, verification-
script-asserted). GPU runs are yours; every script takes `--backend vllm`.

## Install (Colab/T4)
```
pip install vllm transformers datasets
```

## Run order and gates

**1. P0 — capability go/no-go (Document B §4)**
```
python p0_pandora.py --backend vllm --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --n 100 --outdir out_p0_15
python p0_pandora.py --backend vllm --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B  --n 100 --outdir out_p0_7b
```
Gate: clear-instance optimal match ≥ 0.70 → GO at that scale. Both fail → K0a
halt per Document B. Near-boundary instances (20%, by construction) are
excluded from the gate and reported separately. Parse-rate assert at 95%.

**2. K0b — comprehension probe (blocks all main runs)**
```
python k0b_comprehension.py --backend vllm --model <P0 winner> --n-items 20 --outdir out_k0b
```
Gate: ≥ 0.90 correct restatement per condition. Restatement format has four
slots (CORRECT / THINK / ANSWER / WRONG) — the ANSWER slot exists so the
placebo condition is expressible; the smoke test caught its absence.

**3. P1 — difficulty stratification**
```
python p1_stratify.py --backend vllm --model <P0 winner> --hf-dataset gsm8k --k 16 --per-stratum 50 --outdir out_p1
```
Output: `p1_selection.json` (50 ids per stratum). Warnings if a stratum is
short — enlarge the candidate pool (add MATH-500) rather than loosening bands.
Pilot seeds are burned into `burned_seeds.json`; main runs must assert
disjointness.

**4. P2 — clarification base rate (Exp 2 system-prompt decision)**
```
python p2_clarification.py --backend vllm --model Qwen/Qwen2.5-7B-Instruct --k 12 --outdir out_p2
```
Decision printed: permission line iff base ambiguous clarify-rate < 5%.
Warn if unambiguous items draw > 10% clarification (item or classifier issue).

## What to send back for review
`burned_seeds.json`, all four `.jsonl` logs (headers carry config
fingerprints), and the printed gate lines. I re-run `verify_summary` on the
raw logs before anything enters Document B at freeze.

## Known limitations to fix ON THE RIG (pre-registered as such)
1. `thinking_tokens()` uses a whitespace proxy — swap in the model tokenizer
   (one-line change; the proxy is fine for pilots, not for main-run L).
2. P0 prompts are my faithful one-step Pandora variant; after your CTA full
   read, port their App. J prompts and re-run P0 as a robustness check.
3. P2 clarify-classifier is heuristic-v0; validate against 100 hand labels
   (≥95% agreement or upgrade to LLM-judge). Classifier version is stamped
   into every record.
4. P2 has 8 seed items; Exp 2 proper needs 60 with the VOI three-way design.
5. vLLM backend uses `enforce_eager=True` and engine-model assertion; add your
   StaticCore session-drift controls (in-session baseline) when main runs start.
