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

FMT = re.compile(r"CORRECT:\+?([\d\.]+);\s*THINK:-([\d\.]+)/100tok;\s*"
                 r"ANSWER:-([\d\.]+)/word;\s*WRONG:-([\d\.]+)", re.I)

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
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mock")
    ap.add_argument("--model", default="mock-model")
    ap.add_argument("--n-items", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--gate", type=float, default=0.90)
    a = ap.parse_args()

    cfg = GenConfig(model_id=a.model, backend=a.backend, seed=a.seed,
                    max_new_tokens=2048)  # reasoning models think first
    be = make_backend(cfg)
    burn_seeds("k0b", [a.seed], a.outdir)
    log = RunLog(a.outdir, f"k0b_{a.backend}", cfg, {"n_items": a.n_items})

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

    outs = be.generate(prompts, n=1, max_new_tokens=2048)
    mem = []
    for cond, o in zip(meta, outs):
        seg = o[0].split("</think>")[-1]  # grade the answer, not the thinking
        m = FMT.search(seg) or FMT.search(o[0])
        ok = False
        if m:
            got = tuple(float(x) for x in m.groups())
            ok = got == truth(cond)
        rec = {"condition": cond, "parsed": bool(m), "correct": bool(ok),
               "raw": o[0][:160]}
        mem.append(rec); log.write(dict(rec))

    summary = recompute(mem)
    log.close(summary)
    verify_summary(log.path, recompute)

    accs = {k: v for k, v in summary.items() if k.startswith("acc__")}
    fails = {k: v for k, v in accs.items() if v < a.gate}
    for k, v in sorted(accs.items()):
        print(f"  {k}: {v:.3f}")
    if fails:
        print(f"[GATE] K0b FAIL for {list(fails)} — redesign those prompts "
              f"before any main run.")
        raise SystemExit(1)
    print(f"[GATE] K0b PASS — all conditions >= {a.gate}")


if __name__ == "__main__":
    main()
