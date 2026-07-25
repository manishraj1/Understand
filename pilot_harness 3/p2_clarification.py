"""
p2_clarification.py — Clarification base-rate pilot (Document B §6.1, P2).

Measures P(clarify) on matched ambiguous/unambiguous probes under two system
prompts (base vs. explicit-permission). Decision output: which system prompt
Exp 2 uses (permission line added iff base rate < 5% on ambiguous items), held
CONSTANT across all Exp-2 conditions thereafter.

Clarify classifier: heuristic v0 (question mark directed at user + no final
answer). Pre-registered plan: heuristic validated against 100 hand labels on
the rig; if agreement < 95%, upgrade to LLM-judge with the heuristic as tie
report. The classifier version used is recorded in every record.

Smoke: python p2_clarification.py --backend mock --k 8 --outdir out
"""
import argparse, re
from harness import (GenConfig, make_backend, RunLog, burn_seeds,
                     assert_prompt_contains, verify_summary)
from prompts import P2_ITEMS, P2_SYSTEM_BASE, P2_SYSTEM_PERMISSION

CLASSIFIER_VERSION = "heuristic-v0"
QWORDS = re.compile(r"\b(which|what|could you|can you|do you mean|clarify|"
                    r"specify|whom|for how many|what time)\b", re.I)


def is_clarification(text: str) -> bool:
    t = text.strip()
    has_q = "?" in t and QWORDS.search(t) is not None
    gives_answer = bool(re.search(r"(answer|here is|def |\\boxed)", t, re.I))
    return has_q and not gives_answer


def recompute(recs):
    out = {}
    for sysname in ("base", "permission"):
        for variant in ("ambiguous", "unambiguous"):
            sub = [r for r in recs if r["system"] == sysname
                   and r["variant"] == variant]
            if sub:
                rate = sum(r["clarified"] for r in sub) / len(sub)
                out[f"rate__{sysname}__{variant}"] = round(rate, 6)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mock")
    ap.add_argument("--model", default="mock-model")   # Qwen2.5-7B-Instruct on rig
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--threshold", type=float, default=0.05)
    a = ap.parse_args()

    cfg = GenConfig(model_id=a.model, backend=a.backend, seed=a.seed,
                    max_new_tokens=512)
    be = make_backend(cfg)
    burn_seeds("p2", [a.seed], a.outdir)
    log = RunLog(a.outdir, f"p2_clar_{a.backend}", cfg,
                 {"k": a.k, "classifier": CLASSIFIER_VERSION})

    prompts, meta = [], []
    for sysname, sysprompt in (("base", P2_SYSTEM_BASE),
                               ("permission", P2_SYSTEM_PERMISSION)):
        for iid, variant, text in P2_ITEMS:
            p = f"[SYSTEM] {sysprompt}\n[USER] {text}\n[ASSISTANT]"
            assert_prompt_contains(p, "AMBIG_PROBE", "p2")
            prompts.append(p)
            meta.append((sysname, iid, variant))

    outs = be.generate(prompts, n=a.k, max_new_tokens=512)
    mem = []
    for (sysname, iid, variant), os_ in zip(meta, outs):
        for j, o in enumerate(os_):
            rec = {"system": sysname, "item": iid, "variant": variant,
                   "sample": j, "clarified": is_clarification(o),
                   "classifier": CLASSIFIER_VERSION, "raw": o[:120]}
            mem.append(rec); log.write(dict(rec))

    summary = recompute(mem)
    log.close(summary)
    verify_summary(log.path, recompute)

    base_amb = summary.get("rate__base__ambiguous", 0.0)
    base_unamb = summary.get("rate__base__unambiguous", 0.0)
    for k, v in sorted(summary.items()):
        if k.startswith("rate__"):
            print(f"  {k}: {v:.3f}")
    if base_amb < a.threshold:
        print(f"[DECISION] base ambiguous clarify-rate {base_amb:.3f} < "
              f"{a.threshold} -> Exp 2 uses PERMISSION system prompt "
              f"(constant across all conditions).")
    else:
        print(f"[DECISION] base rate sufficient ({base_amb:.3f}) -> Exp 2 uses "
              f"BASE system prompt.")
    if base_unamb > 0.10:
        print("[WARN] non-trivial clarification on UNambiguous items — "
              "inspect items or classifier before Exp 2.")


if __name__ == "__main__":
    main()
