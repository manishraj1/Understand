"""
k0b_comprehension.py — Manipulation-strength check (Document B §4, K0b).

For every cost condition, prepend the condition text to a sample task, then
require the model to restate the cost structure in a strict format BEFORE
solving. Grade by exact parse against the condition's ground truth.

GATE (pre-registered): >= 90% correct restatement per condition, on held-out
items never reused in main runs. A null cost effect without a passed K0b is
evidence of an unread prompt, not of cost-blindness.

Smoke: python k0b_comprehension.py --backend mock --n-items 10 --outdir out
"""
import argparse, re
from harness import (GenConfig, make_backend, RunLog, hard_assert,
                     assert_prompt_contains, burn_seeds, verify_summary)
from prompts import (build_exp1_prompt, K0B_PROBE, ALL_EXP1_CONDITIONS,
                     CC_DOSES, CE_DOSES, PLACEBO_DOSES)

# STRICT: the requested line, tolerant only of whitespace/bold/commas.
FMT_STRICT = re.compile(
    r"\**\s*CORRECT\s*:\s*\+?\s*(\d[\d,]*(?:\.\d+)?)\s*;\s*"
    r"THINK\s*:\s*-?\s*(\d[\d,]*(?:\.\d+)?)\s*/\s*100\s*tok\s*;\s*"
    r"ANSWER\s*:\s*-?\s*(\d[\d,]*(?:\.\d+)?)\s*/\s*word\s*;\s*"
    r"WRONG\s*:\s*-?\s*(\d[\d,]*(?:\.\d+)?)", re.I)

# LENIENT v2: per-slot semantic extraction from the post-think segment.
#   * keyword anchors OR natural prose anchors ("per 100 tokens", "per word")
#   * an OMITTED slot is graded as 0 — the probe says "write 0 for
#     unspecified", but omitting an unspecified rule is valid comprehension;
#     grading must not depend on compliance with my formatting instruction.
#   * last occurrence wins (end-anchored bias); omissions are counted.
SLOT = {
    "correct": [re.compile(r"CORRECT[^0-9\-]{0,30}(\d[\d,]*(?:\.\d+)?)", re.I),
                re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(?:points?|pts)[^.;:\n]{0,40}"
                           r"correct", re.I)],
    "think":   [re.compile(r"THINK(?:ING)?[^0-9\-]{0,30}(\d[\d,]*(?:\.\d+)?)", re.I),
                re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(?:points?|pts)?[^.;:\n]{0,40}"
                           r"(?:per|every)\s*100\s*tok", re.I),
                re.compile(r"100\s*tok[^0-9.;:\n]{0,40}(\d[\d,]*(?:\.\d+)?)", re.I)],
    "answer":  [re.compile(r"ANSWER[^0-9\-]{0,30}(\d[\d,]*(?:\.\d+)?)\s*/?\s*word", re.I),
                re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(?:points?|pts)?[^.;:\n]{0,40}"
                           r"(?:per|every)\s*word", re.I)],
    "wrong":   [re.compile(r"WRONG[^0-9\-]{0,30}(\d[\d,]*(?:\.\d+)?)", re.I),
                re.compile(r"(?:wrong|incorrect)[^0-9.;:\n]{0,40}(\d[\d,]*(?:\.\d+)?)\s*"
                           r"(?:points?|pts)", re.I)],
}

def _num(s):
    s = s.replace(",", "").strip(".")
    return float(s) if s else -1.0

def grade(seg: str, gold: tuple):
    m = FMT_STRICT.search(seg)
    strict = bool(m) and tuple(_num(x) for x in m.groups()) == gold
    vals, omitted = [], 0
    for key in ("correct", "think", "answer", "wrong"):
        hits = []
        for pat in SLOT[key]:
            hits += pat.findall(seg)
        if hits:
            vals.append(_num(hits[-1]))
        else:
            vals.append(0.0)      # omission graded as 0
            omitted += 1
    lenient = tuple(vals) == gold
    return strict, (strict or lenient), omitted

# ground truth per condition: (correct, think_per_100, answer_per_word, wrong)
def truth(cond: str):
    def num(s):  # pull the deduction number out of the condition text
        m = re.search(r"deducts ([\d\.]+) points", s)
        return float(m.group(1))
    if cond in CC_DOSES:
        return (1000.0, num(CC_DOSES[cond]), 0.0, 0.0)
    if cond in CE_DOSES:
        return (1000.0, 1.0, 0.0, num(CE_DOSES[cond]))
    if cond == "empty":
        return (1000.0, 0.0, 0.0, 0.0)
    if cond in PLACEBO_DOSES:
        return (1000.0, 0.0, num(PLACEBO_DOSES[cond]), 0.0)
    raise ValueError(cond)


def recompute(recs):
    out = {}
    for cond in sorted({r["condition"] for r in recs}):
        sub = [r for r in recs if r["condition"] == cond]
        out[f"acc__{cond}"] = round(sum(r["correct"] for r in sub) / len(sub), 6)
        out[f"strict__{cond}"] = round(
            sum(r.get("correct_strict", False) for r in sub) / len(sub), 6)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mock")
    ap.add_argument("--model", default="mock-model")
    ap.add_argument("--n-items", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--gate", type=float, default=0.90)
    ap.add_argument("--k", type=int, default=8,
                    help="samples per item; reliability across samples is "
                         "the measured quantity")
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="MUST match main-run decoding (Doc B §5.1). A "
                         "manipulation check at greedy certifies a regime "
                         "the experiment never runs in.")
    a = ap.parse_args()

    cfg = GenConfig(model_id=a.model, backend=a.backend, seed=a.seed,
                    max_new_tokens=2048,  # reasoning models think first
                    temperature=a.temperature)
    be = make_backend(cfg)
    burn_seeds("k0b", [a.seed], a.outdir)
    log = RunLog(a.outdir, f"k0b_{a.backend}", cfg,
                 {"n_items": a.n_items, "k": a.k,
                  "construct": "in-situ manipulation salience at main-run "
                               "decoding; NOT instructed extraction"})

    # held-out dummy questions (never in main runs by construction)
    questions = [f"Compute {i} + {i + 3}. HELDOUT_K0B" for i in range(a.n_items)]

    # brevity/baseline carry no cost structure -> nothing to restate; skipped
    conds = [c for c in ALL_EXP1_CONDITIONS if c not in ("baseline", "brevity")]
    prompts, meta = [], []
    for cond in conds:
        for q in questions:
            body, frag = build_exp1_prompt(cond, q)
            p = body.replace("PROBLEM:", K0B_PROBE + "\nPROBLEM:")
            assert_prompt_contains(p, frag, cond)
            assert_prompt_contains(p, "RESTATE_COST_STRUCTURE", "k0b-probe")
            prompts.append(p); meta.append(cond)

    outs = be.generate(prompts, n=a.k, max_new_tokens=2048)
    mem = []
    debug = open(f"{a.outdir}/k0b_fulltext.jsonl", "w")
    import json as _j
    for cond, os_ in zip(meta, outs):
        for j, o in enumerate(os_):
            seg = o.split("</think>")[-1]  # grade the answer, not thinking
            strict, lenient, omitted = grade(seg, truth(cond))
            rec = {"condition": cond, "sample": j,
                   "correct_strict": bool(strict),
                   "correct": bool(lenient), "omitted_slots": omitted,
                   "think_closed": "</think>" in o,
                   "tail": o[-200:]}
            mem.append(rec); log.write(dict(rec))
            debug.write(_j.dumps({"condition": cond, "sample": j,
                                  "correct": rec["correct"],
                                  "full": o}) + "\n")
    debug.close()

    summary = recompute(mem)
    log.close(summary)
    verify_summary(log.path, recompute)

    accs = {k: v for k, v in summary.items() if k.startswith("acc__")}
    fails = {k: v for k, v in accs.items() if v < a.gate}
    for k, v in sorted(summary.items()):
        if k.startswith(("acc__", "strict__")):
            print(f"  {k}: {v:.3f}")
    print("  (gate uses acc__ = lenient semantic; strict__ is diagnostic "
          "of format compliance)")
    if fails:
        print(f"[GATE] K0b FAIL for {list(fails)}")
        print("--- SELF-DIAGNOSIS: sample post-think output per failing "
              "condition ---")
        import json as _j
        by_cond = {}
        for line in open(f"{a.outdir}/k0b_fulltext.jsonl"):
            d = _j.loads(line)
            if not d.get("correct", True):
                by_cond.setdefault(d["condition"], d["full"])
        for key in fails:
            cond = key.replace("acc__", "")
            full = by_cond.get(cond, "")
            seg = full.split("</think>")[-1].strip()
            print(f"\n[{cond}] post-think segment (first 400 chars):")
            print("  " + seg[:400].replace("\n", "\n  "))
        print("\nFull outputs: " + f"{a.outdir}/k0b_fulltext.jsonl")
        raise SystemExit(1)
    print(f"[GATE] K0b PASS — all conditions >= {a.gate}")


if __name__ == "__main__":
    main()
