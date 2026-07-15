"""
p0_pandora.py — Capability pilot (Document B §4).

One-step Pandora's Box: hold fallback F; box worth V w.p. p else 0; opening
costs c; after opening keep max(contents, F).

Oracle: E[gain from opening] = p * max(V - F, 0) - c   (one-step exact).
        OPEN iff E[gain] > 0.

Instances are seeded/deterministic. Near-boundary instances (|E[gain]| below
5% of F) are generated deliberately (20%) but EXCLUDED from the gate metric
and reported separately — the gate should measure computation, not tie-breaking.

GATE (pre-registered): optimal-match >= 70% on clear instances -> GO.
Usage:
  python p0_pandora.py --backend mock --n 100 --outdir out
  python p0_pandora.py --backend vllm --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --n 100 --outdir out
"""
import argparse, random, re, statistics, sys
from harness import (GenConfig, make_backend, RunLog, hard_assert,
                     assert_prompt_contains, burn_seeds, verify_summary)
from prompts import PANDORA_TEMPLATE

DECISION = re.compile(r"DECISION:\s*(OPEN|TAKE)", re.I)


def gen_instances(n: int, seed: int):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        F = rng.choice([50, 100, 200])
        V = rng.choice([150, 300, 600, 1000])
        p = rng.choice([0.1, 0.25, 0.5, 0.75, 0.9])
        egain_no_c = p * max(V - F, 0)
        if i % 5 == 0:  # 20% near-boundary by construction
            c = round(egain_no_c * rng.uniform(0.95, 1.05), 2)
        else:
            c = round(egain_no_c * rng.choice([0.3, 0.6, 1.5, 3.0]), 2)
        egain = egain_no_c - c
        oracle = "OPEN" if egain > 0 else "TAKE"
        near = abs(egain) < 0.05 * F
        out.append(dict(idx=i, F=F, V=V, p=p, c=c, egain=round(egain, 3),
                        oracle=oracle, near_boundary=near))
    return out


def recompute(recs):
    clear = [r for r in recs if not r["near_boundary"] and r["parsed"]]
    hard_assert(len(clear) > 0, "no clear parsed records")
    match = sum(r["match"] for r in clear) / len(clear)
    parse_rate = sum(r["parsed"] for r in recs) / len(recs)
    return {"clear_match_rate": round(match, 6),
            "parse_rate": round(parse_rate, 6),
            "n_clear": len(clear)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mock")
    ap.add_argument("--model", default="mock-model")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1701)
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--gate", type=float, default=0.70)
    a = ap.parse_args()

    cfg = GenConfig(model_id=a.model, backend=a.backend, seed=a.seed,
                    max_new_tokens=1024)
    be = make_backend(cfg)
    burn_seeds("p0_pandora", [a.seed], a.outdir)
    inst = gen_instances(a.n, a.seed)
    log = RunLog(a.outdir, f"p0_pandora_{a.backend}", cfg,
                 {"n": a.n, "gate": a.gate})

    prompts = []
    for it in inst:
        p = PANDORA_TEMPLATE.format(fallback=it["F"], v=it["V"],
                                    p=it["p"], c=it["c"])
        if a.backend == "mock":
            p += f"\nSMOKE_ORACLE={it['oracle']}"
        assert_prompt_contains(p, "PANDORA", "p0")
        prompts.append(p)

    outs = be.generate(prompts, n=1)
    mem = []
    for it, o in zip(inst, outs):
        m = DECISION.search(o[0])
        parsed = m is not None
        choice = m.group(1).upper() if m else None
        rec = {**it, "parsed": parsed, "choice": choice,
               "match": bool(parsed and choice == it["oracle"]),
               "raw_tail": o[0][-200:]}
        mem.append(rec)
        log.write(dict(rec))

    summary = recompute(mem)
    log.close(summary)
    verify_summary(log.path, recompute)

    print(f"P0 clear-instance optimal match: {summary['clear_match_rate']:.3f} "
          f"(n={summary['n_clear']}), parse rate {summary['parse_rate']:.3f}")
    hard_assert(summary["parse_rate"] >= 0.95,
                "parse rate < 95%: fix decision-format prompting before gating")
    if summary["clear_match_rate"] >= a.gate:
        print(f"[GATE] GO — {summary['clear_match_rate']:.3f} >= {a.gate}")
    else:
        print(f"[GATE] FAIL at this scale — escalate per Document B §4")
        sys.exit(1)


if __name__ == "__main__":
    main()
