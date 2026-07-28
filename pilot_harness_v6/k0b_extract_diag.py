"""
k0b_extract_diag.py — UNGATED capability diagnostic (companion to k0b v5).

Construct: can the model extract cost numbers when explicitly directed and
decoding is deterministic? This measures ACCESSIBILITY of the cost text.
The gated k0b (in-situ, T=0.8) measures SALIENCE under task conditions.
The (accessibility, salience) pair localizes failures:
  pass/fail  -> encoded but not attended (salience failure)
  fail/fail  -> capacity failure
  pass/pass  -> platform fully viable for the manipulation check

Differences from the retired frozen diagnostic: uses the clean v5 grader
(no substring `present()` fallback), no hand-rolled ChatML (the backend
applies the model's real chat template), and reports per-condition rates
that are trustworthy in BOTH directions.

Usage:
  python k0b_extract_diag.py --backend vllm --model <model> --n-items 20 --outdir out_diag
"""
import argparse
from harness import (GenConfig, make_backend, RunLog, burn_seeds,
                     assert_prompt_contains, verify_summary)
from prompts import build_exp1_prompt, ALL_EXP1_CONDITIONS
from k0b_comprehension import grade, truth, recompute

EXTRACT_INSTR = (
    "RESTATE_COST_STRUCTURE: Read the task description below. Do not solve "
    "the problem. State the scoring rules it specifies: the points for a "
    "correct answer, any deduction per 100 thinking tokens, any deduction "
    "per word of the final answer, and any deduction for a wrong answer. "
    "If a rule is not specified, you may omit it.\n\nTASK DESCRIPTION:\n"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mock")
    ap.add_argument("--model", default="mock-model")
    ap.add_argument("--n-items", type=int, default=20)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--outdir", default="out_diag")
    a = ap.parse_args()

    cfg = GenConfig(model_id=a.model, backend=a.backend, seed=a.seed,
                    max_new_tokens=2048, temperature=0.0)
    be = make_backend(cfg)
    burn_seeds("k0b_extract_diag", [a.seed], a.outdir)
    log = RunLog(a.outdir, f"k0b_diag_{a.backend}", cfg,
                 {"n_items": a.n_items,
                  "construct": "directed extraction at greedy — "
                               "accessibility, NOT salience; UNGATED"})

    questions = [f"Compute {i} + {i + 3}. HELDOUT_K0B" for i in range(a.n_items)]
    conds = [c for c in ALL_EXP1_CONDITIONS if c not in ("baseline", "brevity")]
    prompts, meta = [], []
    for cond in conds:
        for q in questions:
            body, frag = build_exp1_prompt(cond, q)
            p = EXTRACT_INSTR + body
            assert_prompt_contains(p, frag, cond)
            prompts.append(p); meta.append(cond)

    import json as _j
    outs = be.generate(prompts, n=1)
    mem = []
    debug = open(f"{a.outdir}/k0b_diag_fulltext.jsonl", "w")
    for cond, o in zip(meta, outs):
        seg = o[0].split("</think>")[-1]
        strict, lenient, omitted = grade(seg, truth(cond))
        rec = {"condition": cond, "correct_strict": bool(strict),
               "correct": bool(lenient), "omitted_slots": omitted,
               "tail": o[0][-200:]}
        mem.append(rec); log.write(dict(rec))
        debug.write(_j.dumps({"condition": cond, "correct": rec["correct"],
                              "full": o[0]}) + "\n")
    debug.close()

    summary = recompute(mem)
    log.close(summary)
    verify_summary(log.path, recompute)
    for k, v in sorted(summary.items()):
        if k.startswith("acc__"):
            print(f"  {k}: {v:.3f}")
    print("[DIAGNOSTIC] ungated — interpret jointly with in-situ k0b per the "
          "accessibility/salience table in this file's docstring.")


if __name__ == "__main__":
    main()
